"""Mine replayable decision packets and score saved world models offline.

The module is evaluation machinery.  It keeps information available before a
decision in ``admissible/`` and successor/engine information in ``oracle/``.
Nothing here is imported by the ARC host or by a solver process.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

import numpy as np

PACKET_SCHEMA = "failure-atlas-packet-v1"
CORPUS_SCHEMA = "failure-atlas-corpus-v1"
REPLAY_SCHEMA = "failure-atlas-branch-replay-v1"
BUILDER_REFRESH_SCHEMA = "builder-report-refresh-inventory-v1"
BUILDER_REPORT_MARKER = (
    "automatic world-model builder report (advisory; inspect or test it):"
)
BUILDER_REPORT_FIELDS = frozenset(
    {
        "confidence",
        "model",
        "dynamics",
        "outcome",
        "outcome_verified",
        "plan",
        "recommended_action",
        "subgoal",
        "note",
    }
)
UNKNOWN = -1
TERMINAL_STATES = frozenset({"WIN", "GAME_OVER"})


@dataclass(frozen=True)
class Action:
    action: str
    row: int | None = None
    col: int | None = None

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Action:
        return cls(
            action=str(value["action"]),
            row=value.get("row"),
            col=value.get("col"),
        )

    def to_json(self) -> dict[str, Any]:
        return {"action": self.action, "row": self.row, "col": self.col}

    def key(self) -> tuple[str, int | None, int | None]:
        return self.action, self.row, self.col


@dataclass(frozen=True)
class Observation:
    grid: tuple[tuple[int, ...], ...]
    available_actions: tuple[str, ...]
    levels_completed: int
    state: str
    full_reset: bool

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Observation:
        return cls(
            grid=tuple(tuple(int(cell) for cell in row) for row in value["grid"]),
            available_actions=tuple(str(item) for item in value["available_actions"]),
            levels_completed=int(value["levels_completed"]),
            state=str(value["state"]),
            full_reset=bool(value["full_reset"]),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "grid": [list(row) for row in self.grid],
            "available_actions": list(self.available_actions),
            "levels_completed": self.levels_completed,
            "state": self.state,
            "full_reset": self.full_reset,
        }


@dataclass(frozen=True)
class RecordedTransition:
    sequence: int
    level: int
    turn_in_level: int
    decision: Action
    before: Observation
    after: Observation


@dataclass(frozen=True)
class RecordedRun:
    game_id: str
    seed: int
    initial: Observation
    transitions: tuple[RecordedTransition, ...]


class ReplayEnvironment(Protocol):
    """Fresh public black-box execution of one action sequence."""

    def run(self, actions: Sequence[Action]) -> tuple[Observation, ...]: ...


class WorldModel:
    """The narrow Tycho world-model contract, loaded only by evaluation code."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.module = _load_world_model(self.workspace)

    def init(self, observation: Observation, level: int) -> Any:
        return self.module.init_state(np.asarray(observation.grid, dtype=np.int16), level)

    def transition(self, state: Any, action: Action) -> Any:
        return self.module.transition(state, action.to_json())

    def render(self, state: Any) -> np.ndarray:
        rendered = np.asarray(self.module.render(state))
        if rendered.ndim != 2:
            raise ValueError(f"world model rendered shape {rendered.shape}, expected 2-D")
        return rendered

    def outcome(self, state: Any) -> str:
        value = str(self.module.outcome(state))
        if value not in {"ongoing", "level_complete", "game_over"}:
            raise ValueError(f"invalid world-model outcome {value!r}")
        return value

    def actions(self, state: Any) -> tuple[Action, ...]:
        values = self.module.actions(state)
        return tuple(Action.from_json(value) for value in values)

    def variants(self, state: Any) -> tuple[np.ndarray, ...]:
        function = getattr(self.module, "observation_variants", None)
        if function is None:
            return ()
        values = tuple(np.asarray(value) for value in function(state))
        if len(values) > 5:
            raise ValueError("observation_variants returned more than 5 grids")
        return values


def mine_builder_refreshes(
    checkpoint_roots: Sequence[Path],
    *,
    shadow_root: Path | None = None,
) -> dict[str, Any]:
    """Inventory paid builder batches and nominate report-refresh shadows offline."""

    shadow_index = _builder_shadow_index(shadow_root)
    refreshes: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    total_builder_cost = 0.0

    for supplied_root in checkpoint_roots:
        root = supplied_root.resolve()
        calls = _refresh_checkpoint_calls(root)
        source_rows.append({"checkpoint_root": str(root), "call_count": len(calls)})
        previous_report: dict[str, Any] | None = None
        for builder_batch in _builder_batches(calls):
            sequence = builder_batch[0]["sequence"]
            last_builder_call = builder_batch[-1]["call_index"]
            actor = next(
                (
                    row
                    for row in calls
                    if row["call_index"] > last_builder_call
                    and row["source"] == "actor"
                    and row["sequence"] == sequence
                ),
                None,
            )
            if actor is None:
                issues.append(
                    {
                        "checkpoint_root": str(root),
                        "sequence": sequence,
                        "last_builder_call": last_builder_call,
                        "issue": "builder_batch_has_no_following_actor_checkpoint",
                    }
                )
                continue
            report = _latest_builder_report(actor["request_body"].get("messages", []))
            if report is None:
                issues.append(
                    {
                        "checkpoint_root": str(root),
                        "sequence": sequence,
                        "actor_call_index": actor["call_index"],
                        "issue": "following_actor_request_has_no_builder_report",
                    }
                )
                continue

            builder_summary = _builder_batch_summary(builder_batch)
            total_builder_cost += builder_summary["cost_usd"]
            report_commitment = commitment_json(report["fields"])
            previous_fields = previous_report["fields"] if previous_report else None
            previous_action = (
                _recommended_action(previous_fields) if previous_fields else None
            )
            action = _recommended_action(report["fields"])
            changed_fields = (
                sorted(
                    name
                    for name in set(previous_fields or {}) | set(report["fields"])
                    if (previous_fields or {}).get(name) != report["fields"].get(name)
                )
                if previous_fields is not None
                else []
            )
            if previous_report is None:
                signal = "baseline_first_report"
            elif report_commitment == previous_report["commitment"]:
                signal = "exact_content_repeat"
            elif action is None:
                signal = "no_action_recommendation"
            elif action == previous_action:
                signal = "unchanged_action_recommendation"
            else:
                signal = "changed_action_recommendation"

            shadows = shadow_index.get((str(root), actor["call_index"]), [])
            refresh = {
                "checkpoint_root": str(root),
                "sequence": sequence,
                "decision_point": actor["decision_point"],
                "actor_call_index": actor["call_index"],
                "report_message_index": report["message_index"],
                "report_commitment": report_commitment,
                "previous_report_commitment": (
                    previous_report["commitment"] if previous_report else None
                ),
                "report_fields": report["fields"],
                "changed_fields": changed_fields,
                "recommended_action": action,
                "previous_recommended_action": previous_action,
                "candidate_signal": signal,
                "builder": builder_summary,
                "immediate_actor_response": _immediate_actor_response(actor, action),
                "shadow_evidence": shadows,
                "already_shadowed": bool(shadows),
                "claim_ceiling": (
                    "checkpoint-content redundancy candidate; not causal dependence, "
                    "gameplay benefit, or permission to remove the builder"
                ),
            }
            refreshes.append(refresh)
            previous_report = {
                "commitment": report_commitment,
                "fields": report["fields"],
            }

    candidates = [
        row for row in refreshes if row["candidate_signal"] != "baseline_first_report"
    ]
    priority = {
        "exact_content_repeat": 0,
        "no_action_recommendation": 1,
        "unchanged_action_recommendation": 2,
        "changed_action_recommendation": 3,
    }
    candidates.sort(
        key=lambda row: (
            row["already_shadowed"],
            priority[row["candidate_signal"]],
            -row["builder"]["cost_usd"],
            row["checkpoint_root"],
            row["sequence"],
        )
    )
    return {
        "schema": BUILDER_REFRESH_SCHEMA,
        "model_calls_made": 0,
        "sources": source_rows,
        "refresh_count": len(refreshes),
        "candidate_count": len(candidates),
        "total_builder_cost_usd": round(total_builder_cost, 12),
        "refreshes": refreshes,
        "candidates": candidates,
        "issues": issues,
        "ranking": (
            "unshadowed first; then exact repeat, no action recommendation, "
            "unchanged recommendation, changed recommendation; then builder cost"
        ),
        "claim_ceiling": (
            "saved-checkpoint report novelty and spend; no model counterfactual, "
            "gameplay consequence, or transfer"
        ),
    }


def _refresh_checkpoint_calls(root: Path) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for call_dir in sorted((root / "calls").iterdir()):
        if not call_dir.is_dir() or not call_dir.name.isdigit():
            continue
        request = read_json(call_dir / "request.json")
        response = read_json(call_dir / "response.json")
        body = request.get("request_body")
        if not isinstance(body, dict):
            raise TypeError(f"checkpoint request body missing: {call_dir}")
        if request.get("request_payload_commitment") != commitment_json(body):
            raise ValueError(f"checkpoint request commitment is stale: {call_dir}")
        decision = request.get("decision_point")
        if not isinstance(decision, dict) or not isinstance(
            decision.get("sequence"), int
        ):
            raise TypeError(f"checkpoint decision point missing: {call_dir}")
        source = request.get("source")
        if source not in {"actor", "builder"}:
            continue
        calls.append(
            {
                "call_index": int(call_dir.name),
                "source": source,
                "sequence": int(decision["sequence"]),
                "decision_point": decision,
                "request_body": body,
                "response": response,
            }
        )
    return calls


def _builder_batches(calls: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in calls:
        if row["source"] == "builder":
            if current and row["sequence"] != current[-1]["sequence"]:
                batches.append(current)
                current = []
            current.append(row)
        elif current:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


def _latest_builder_report(messages: Any) -> dict[str, Any] | None:
    if not isinstance(messages, list):
        return None
    reports: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            continue
        for segment in message["content"].split(BUILDER_REPORT_MARKER)[1:]:
            fields = _parse_builder_report(segment)
            if fields is not None:
                reports.append({"message_index": message_index, "fields": fields})
    return reports[-1] if reports else None


def _parse_builder_report(segment: str) -> dict[str, str] | None:
    fields: dict[str, str] = {}
    for raw_line in segment.splitlines():
        line = raw_line.strip().strip("`")
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        if name not in BUILDER_REPORT_FIELDS:
            continue
        value = value.strip()
        if name == "recommended_action" and " subgoal:" in value:
            value, subgoal = value.split(" subgoal:", 1)
            fields["subgoal"] = subgoal.strip()
        if name == "note":
            # Some recorded reports sit inside a Python triple-quoted argument.
            # The wrapper is transport syntax, not builder-authored report content.
            value = value.removesuffix("''')").rstrip()
        fields[name] = value
        if name == "note":
            break
    required = BUILDER_REPORT_FIELDS - {"subgoal"}
    return fields if required <= fields.keys() else None


def _recommended_action(fields: Mapping[str, str] | None) -> str | None:
    if not fields:
        return None
    value = fields.get("recommended_action", "").strip()
    if not value or value.lower() == "none":
        return None
    return value.split()[0].upper()


def _builder_batch_summary(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cost = 0.0
    missing_cost = 0
    tokens = {name: 0 for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
    for row in batch:
        usage = row["response"].get("usage")
        usage = usage if isinstance(usage, dict) else {}
        value = usage.get("cost")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            cost += float(value)
        else:
            missing_cost += 1
        for name in tokens:
            value = usage.get(name)
            if isinstance(value, int) and not isinstance(value, bool):
                tokens[name] += value
    return {
        "call_count": len(batch),
        "cost_usd": round(cost, 12),
        "missing_cost_calls": missing_cost,
        **tokens,
    }


def _immediate_actor_response(
    actor: Mapping[str, Any], recommendation: str | None
) -> dict[str, Any]:
    tool_calls = actor["response"].get("tool_calls")
    tool_calls = tool_calls if isinstance(tool_calls, list) else []
    first = tool_calls[0].get("name") if tool_calls else None
    take_action: str | None = None
    for call in tool_calls:
        if not isinstance(call, dict) or call.get("name") != "take_action":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        if isinstance(arguments, dict) and isinstance(arguments.get("action"), str):
            take_action = arguments["action"].upper()
        break
    if recommendation is None:
        relation = "no_recommendation"
    elif take_action is None:
        relation = "not_immediate_action"
    elif take_action == recommendation:
        relation = "matched"
    else:
        relation = "diverged"
    return {
        "call_index": actor["call_index"],
        "first_tool_call": first,
        "take_action": take_action,
        "recommendation_relation": relation,
    }


def _builder_shadow_index(
    shadow_root: Path | None,
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    index: dict[tuple[str, int], list[dict[str, Any]]] = {}
    if shadow_root is None or not shadow_root.exists():
        return index
    for manifest_path in sorted(shadow_root.glob("*/packet/manifest.json")):
        manifest = read_json(manifest_path)
        source = manifest.get("source_checkpoint")
        call_index = manifest.get("call_index")
        if not isinstance(source, str) or not isinstance(call_index, int):
            continue
        result_path = manifest_path.parents[1] / "result" / "report.json"
        result = read_json(result_path) if result_path.is_file() else {}
        assessment = result.get("assessment")
        assessment = assessment if isinstance(assessment, dict) else {}
        index.setdefault((str(Path(source).resolve()), call_index), []).append(
            {
                "shadow": str(manifest_path.parents[1].resolve()),
                "judgment": assessment.get("judgment"),
                "status": result.get("status"),
                "expected_effect": result.get("expected_effect"),
                "control_recorded_response_match_rate": assessment.get(
                    "control_recorded_response_match_rate"
                ),
            }
        )
    return index


def parse_transcript(path: Path) -> RecordedRun:
    """Parse and join the host transcript, rejecting incomplete evidence."""

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    starts = [row for row in rows if row.get("type") == "start_game"]
    if len(starts) != 1:
        raise ValueError(f"expected one start_game record, found {len(starts)}")
    start = starts[0]
    decisions = {
        int(row["sequence"]): Action.from_json(row["decision"])
        for row in rows
        if row.get("type") == "decision"
    }
    results = [row for row in rows if row.get("type") == "action_result"]
    transitions: list[RecordedTransition] = []
    for expected, row in enumerate(results):
        sequence = int(row["sequence"])
        if sequence != expected:
            raise ValueError(
                f"action_result sequence is not contiguous: expected {expected}, got {sequence}"
            )
        result_action = Action.from_json(row["decision"])
        if decisions.get(sequence) != result_action:
            raise ValueError(f"decision/action_result mismatch at sequence {sequence}")
        transitions.append(
            RecordedTransition(
                sequence=sequence,
                level=int(row["level"]),
                turn_in_level=int(row["turn_in_level"]),
                decision=result_action,
                before=Observation.from_json(row["before"]),
                after=Observation.from_json(row["after"]),
            )
        )
    return RecordedRun(
        game_id=str(start["game_id"]),
        seed=int(start["seed"]),
        initial=Observation.from_json(start["initial_observation"]),
        transitions=tuple(transitions),
    )


def mine_corpus(run_dir: Path, output: Path) -> dict[str, Any]:
    """Split a run into content-committed admissible and oracle packets."""

    run_dir = run_dir.resolve()
    transcript = run_dir / "transcript.jsonl"
    recorded = parse_transcript(transcript)
    if output.exists():
        raise FileExistsError(output)
    (output / "admissible").mkdir(parents=True)
    (output / "oracle" / "recorded").mkdir(parents=True)
    (output / "models").mkdir(parents=True)
    (output / "checkpoints").mkdir(parents=True)

    transcript_commitment = commitment_bytes(transcript.read_bytes())
    checkpoint_index = _checkpoint_index(run_dir / "decision-checkpoints")
    final_workspace = _find_game_workspace(run_dir, recorded.game_id)
    packet_entries: list[dict[str, Any]] = []
    model_cache: dict[str, Path] = {}

    for transition in recorded.transitions:
        checkpoint_refs = checkpoint_index.get(transition.sequence, ())
        workspace, source, temporary = _workspace_for_packet(
            run_dir / "decision-checkpoints",
            checkpoint_refs,
            final_workspace,
        )
        try:
            model_commitment, model_dir = _bundle_model(workspace, output / "models")
        finally:
            if temporary is not None:
                temporary.cleanup()
        model_cache.setdefault(model_commitment, model_dir)
        copied_refs = _copy_checkpoint_evidence(
            run_dir / "decision-checkpoints",
            checkpoint_refs,
            output / "checkpoints",
        )
        prefix = [
            item.decision.to_json()
            for item in recorded.transitions[: transition.sequence]
        ]
        observation_prefix = [recorded.initial.to_json()]
        observation_prefix.extend(
            item.after.to_json()
            for item in recorded.transitions[: transition.sequence]
        )
        admissible = {
            "schema": PACKET_SCHEMA,
            "game_id": recorded.game_id,
            "seed": recorded.seed,
            "sequence": transition.sequence,
            "level": transition.level,
            "turn_in_level": transition.turn_in_level,
            "initial_observation": recorded.initial.to_json(),
            "action_prefix": prefix,
            "observation_prefix": observation_prefix,
            "decision_observation": transition.before.to_json(),
            "available_action_labels": list(transition.before.available_actions),
            "executed_action": transition.decision.to_json(),
            "model_commitment": model_commitment,
            "model_source": source,
            "model_temporal_validity": (
                "exact_predecision"
                if source == "decision_checkpoint"
                else "retrospective_final_candidate_not_historical"
            ),
            "checkpoint_refs": copied_refs,
            "source_transcript_commitment": transcript_commitment,
            "claim_ceiling": (
                "saved-model mechanics at this recorded decision state; exact "
                "historical attribution requires model_temporal_validity="
                "exact_predecision; not actor uptake, gameplay benefit, or transfer"
            ),
        }
        packet_id = commitment_json(admissible).split(":", 1)[1][:24]
        admissible["packet_id"] = packet_id
        oracle = {
            "schema": PACKET_SCHEMA,
            "packet_id": packet_id,
            "partition": "oracle_coach",
            "recorded_successor": transition.after.to_json(),
            "observed_outcome": observed_outcome(transition.before, transition.after),
        }
        admissible_path = output / "admissible" / f"{packet_id}.json"
        oracle_path = output / "oracle" / "recorded" / f"{packet_id}.json"
        write_json(admissible_path, admissible)
        write_json(oracle_path, oracle)
        packet_entries.append(
            {
                "packet_id": packet_id,
                "sequence": transition.sequence,
                "admissible": admissible_path.relative_to(output).as_posix(),
                "recorded_oracle": oracle_path.relative_to(output).as_posix(),
                "model_commitment": model_commitment,
            }
        )

    manifest = {
        "schema": CORPUS_SCHEMA,
        "source_run": str(run_dir),
        "game_id": recorded.game_id,
        "seed": recorded.seed,
        "source_transcript_commitment": transcript_commitment,
        "packet_count": len(packet_entries),
        "packets": packet_entries,
        "partitions": {
            "admissible": (
                "pre-decision public trace; saved artifacts are historical only when "
                "model_temporal_validity is exact_predecision"
            ),
            "oracle": "successors, local-engine branches, and evaluator scores",
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def replay_corpus(
    corpus: Path,
    output: Path,
    environment: ReplayEnvironment,
    *,
    sequences: set[int] | None = None,
    max_packets: int | None = None,
    engine_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay packet prefixes through a fresh black box for every branch."""

    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest = read_json(corpus / "manifest.json")
    entries = [
        entry
        for entry in manifest["packets"]
        if sequences is None or int(entry["sequence"]) in sequences
    ]
    if max_packets is not None:
        entries = entries[:max_packets]

    results: list[dict[str, Any]] = []
    model_cache: dict[str, WorldModel] = {}
    for entry in entries:
        packet = read_json(corpus / entry["admissible"])
        model_key = str(packet["model_commitment"])
        model = model_cache.get(model_key)
        if model is None:
            model = WorldModel(corpus / "models" / model_key.removeprefix("sha256:"))
            model_cache[model_key] = model
        result = replay_packet(packet, model, environment)
        path = output / f"{packet['packet_id']}.json"
        write_json(path, result)
        results.append(result)

    _attach_shortest_prefixes(results)
    for result in results:
        write_json(output / f"{result['packet_id']}.json", result)
    summary = _replay_summary(manifest, results, engine_identity or {})
    write_json(output / "summary.json", summary)
    return summary


def replay_packet(
    packet: Mapping[str, Any],
    model: WorldModel,
    environment: ReplayEnvironment,
) -> dict[str, Any]:
    """Parity-check one prefix, then execute every instantiated legal branch."""

    prefix = tuple(Action.from_json(value) for value in packet["action_prefix"])
    expected_before = Observation.from_json(packet["decision_observation"])
    observations = environment.run(prefix)
    parity = trace_parity(packet, observations)
    state, model_prefix = simulate_model_prefix(packet, model, observations)
    if not parity["exact"]:
        return {
            "schema": REPLAY_SCHEMA,
            "packet_id": packet["packet_id"],
            "sequence": packet["sequence"],
            "status": "invalid_engine_prefix_parity",
            "engine_prefix_parity": parity,
            "model_prefix": model_prefix,
            "branches": [],
        }
    declared = model.actions(state)
    candidates, uninstantiated = branch_candidates(
        expected_before.available_actions,
        declared,
        Action.from_json(packet["executed_action"]),
    )
    branches: list[dict[str, Any]] = []
    for action in candidates:
        branch_observations = environment.run((*prefix, action))
        branch_parity = compare_observation_sequences(observations, branch_observations[:-1])
        if not branch_parity["exact"]:
            branches.append(
                {
                    "action": action.to_json(),
                    "status": "invalid_engine_prefix_parity",
                    "engine_prefix_parity": branch_parity,
                }
            )
            continue
        actual_after = branch_observations[-1]
        predicted_state = model.transition(state, action)
        score = score_transition(
            model,
            state,
            predicted_state,
            expected_before,
            actual_after,
        )
        valid_prefix = model_prefix["status"] in {"exact", "partial_unknown"}
        branches.append(
            {
                "action": action.to_json(),
                "status": "scored" if valid_prefix else "invalid_after_model_prefix_divergence",
                "model_action_declared": action.key() in {item.key() for item in declared},
                "successor": actual_after.to_json(),
                "score": score,
            }
        )
    return {
        "schema": REPLAY_SCHEMA,
        "packet_id": packet["packet_id"],
        "sequence": packet["sequence"],
        "status": "scored",
        "engine_prefix_parity": parity,
        "model_prefix": model_prefix,
        "branch_coverage": {
            "legal_action_labels": list(expected_before.available_actions),
            "executed_branches": len(candidates),
            "uninstantiated_parameterized_actions": uninstantiated,
        },
        "branches": branches,
    }


def simulate_model_prefix(
    packet: Mapping[str, Any],
    model: WorldModel,
    observations: Sequence[Observation],
) -> tuple[Any, dict[str, Any]]:
    """Advance from the level start and retain the first canonical divergence."""

    initial = Observation.from_json(packet["initial_observation"])
    actions = tuple(Action.from_json(value) for value in packet["action_prefix"])
    state = model.init(initial, initial.levels_completed)
    initial_score = score_grid(model.render(state), np.asarray(initial.grid))
    first_divergence = None if _only_unknown_error(initial_score) else {
        "at": "initial",
        "score": initial_score,
    }
    partial = initial_score["unknown_cells"] > 0
    current = initial
    for index, action in enumerate(actions):
        predicted = model.transition(state, action)
        actual = observations[index + 1]
        terminal = observed_outcome(current, actual)
        if terminal == "ongoing" and not actual.full_reset:
            grid_score = score_grid(model.render(predicted), np.asarray(actual.grid))
            partial = partial or grid_score["unknown_cells"] > 0
            if not _only_unknown_error(grid_score) and first_divergence is None:
                first_divergence = {
                    "at": index,
                    "action": action.to_json(),
                    "score": grid_score,
                }
        if terminal != "ongoing" or actual.full_reset or action.action == "RESET":
            state = model.init(actual, actual.levels_completed)
        else:
            state = predicted
        current = actual
    status = (
        "diverged"
        if first_divergence is not None
        else ("partial_unknown" if partial else "exact")
    )
    return state, {"status": status, "first_divergence": first_divergence}


def score_transition(
    model: WorldModel,
    before_state: Any,
    predicted_state: Any,
    actual_before: Observation,
    actual_after: Observation,
) -> dict[str, Any]:
    predicted_before = model.render(before_state)
    predicted_after = model.render(predicted_state)
    actual_before_grid = np.asarray(actual_before.grid)
    actual_after_grid = np.asarray(actual_after.grid)
    outcome_expected = observed_outcome(actual_before, actual_after)
    try:
        outcome_predicted = model.outcome(predicted_state)
        outcome_error = None
    except Exception as error:  # noqa: BLE001 - evaluator preserves model failure
        outcome_predicted = None
        outcome_error = {"type": type(error).__name__, "message": str(error)}

    render_applicable = outcome_expected == "ongoing" and not actual_after.full_reset
    grid = score_grid(predicted_after, actual_after_grid) if render_applicable else None
    changes = (
        score_changes(
            predicted_before,
            predicted_after,
            actual_before_grid,
            actual_after_grid,
        )
        if render_applicable
        else None
    )
    variant = _score_variants(model, predicted_state, actual_after_grid) if render_applicable else None
    return {
        "failure": (
            outcome_predicted != outcome_expected
            or (grid is not None and not grid["canonical_exact"])
        ),
        "render_applicable": render_applicable,
        "grid": grid,
        "changed_cells": changes,
        "observation_variants": variant,
        "outcome": {
            "expected": outcome_expected,
            "predicted": outcome_predicted,
            "exact": outcome_predicted == outcome_expected,
            "error": outcome_error,
        },
    }


def score_grid(predicted: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    if predicted.shape != actual.shape:
        return {
            "shape_predicted": list(predicted.shape),
            "shape_actual": list(actual.shape),
            "canonical_exact": False,
            "claimed_cells": 0,
            "unknown_cells": int(predicted.size),
            "prediction_coverage": 0.0,
            "known_cell_accuracy": None,
            "unknown_penalized_cell_accuracy": 0.0,
            "differing_claimed_cells": None,
            "first_claimed_difference": None,
        }
    known = predicted != UNKNOWN
    matching = known & (predicted == actual)
    differing = known & (predicted != actual)
    claimed = int(known.sum())
    total = int(actual.size)
    differences = np.argwhere(differing)
    first = None
    if len(differences):
        row, col = (int(value) for value in differences[0])
        first = {
            "row": row,
            "col": col,
            "predicted": int(predicted[row, col]),
            "actual": int(actual[row, col]),
        }
    return {
        "shape_predicted": list(predicted.shape),
        "shape_actual": list(actual.shape),
        "canonical_exact": claimed == total and not bool(differing.any()),
        "claimed_cells": claimed,
        "unknown_cells": total - claimed,
        "prediction_coverage": claimed / total if total else 0.0,
        "known_cell_accuracy": float(matching.sum() / claimed) if claimed else None,
        "unknown_penalized_cell_accuracy": float(matching.sum() / total) if total else 0.0,
        "differing_claimed_cells": int(differing.sum()),
        "first_claimed_difference": first,
    }


def score_changes(
    predicted_before: np.ndarray,
    predicted_after: np.ndarray,
    actual_before: np.ndarray,
    actual_after: np.ndarray,
) -> dict[str, Any]:
    if not (
        predicted_before.shape
        == predicted_after.shape
        == actual_before.shape
        == actual_after.shape
    ):
        return {"applicable": False, "reason": "shape_mismatch"}
    known = (predicted_before != UNKNOWN) & (predicted_after != UNKNOWN)
    predicted_changed = known & (predicted_before != predicted_after)
    actual_changed = actual_before != actual_after
    true_positive = int((predicted_changed & actual_changed).sum())
    predicted_positive = int(predicted_changed.sum())
    actual_positive = int(actual_changed.sum())
    return {
        "applicable": True,
        "predicted_changed": predicted_positive,
        "actual_changed": actual_positive,
        "true_positive": true_positive,
        "precision": _ratio(true_positive, predicted_positive, empty=actual_positive == 0),
        "recall": _ratio(true_positive, actual_positive, empty=predicted_positive == 0),
        "actual_changes_hidden_by_unknown": int((actual_changed & ~known).sum()),
    }


def branch_candidates(
    legal_labels: Sequence[str],
    declared: Sequence[Action],
    recorded: Action,
) -> tuple[tuple[Action, ...], list[dict[str, Any]]]:
    """Instantiate simple actions and only grounded CLICK coordinates."""

    candidates: dict[tuple[str, int | None, int | None], Action] = {}
    uninstantiated: list[dict[str, Any]] = []
    for label in legal_labels:
        if label != "CLICK":
            action = Action(label)
            candidates[action.key()] = action
            continue
        clicks = [
            action
            for action in (*declared, recorded)
            if action.action == "CLICK"
            and type(action.row) is int
            and type(action.col) is int
            and 0 <= action.row <= 63
            and 0 <= action.col <= 63
        ]
        if not clicks:
            uninstantiated.append(
                {
                    "action": "CLICK",
                    "reason": "no model-declared or recorded public coordinate",
                }
            )
        for action in clicks:
            candidates[action.key()] = action
    return tuple(candidates.values()), uninstantiated


def trace_parity(
    packet: Mapping[str, Any], observations: Sequence[Observation]
) -> dict[str, Any]:
    expected = tuple(
        Observation.from_json(value) for value in packet["observation_prefix"]
    )
    normalized = list(observations)
    ignored: list[dict[str, Any]] = []
    if expected and normalized and expected[0].full_reset != normalized[0].full_reset:
        # The ARC SDK's online and offline reset wrappers disagree on this
        # event marker even when the complete initial grid and all other public
        # metadata match. Preserve the mismatch, but do not confuse it with a
        # mechanics divergence in the replayed action prefix.
        ignored.append(
            {
                "index": 0,
                "field": "full_reset",
                "recorded": expected[0].full_reset,
                "replayed": normalized[0].full_reset,
                "reason": "online_offline_initial_reset_marker",
            }
        )
        normalized[0] = Observation(
            grid=normalized[0].grid,
            available_actions=normalized[0].available_actions,
            levels_completed=normalized[0].levels_completed,
            state=normalized[0].state,
            full_reset=expected[0].full_reset,
        )
    result = compare_observation_sequences(expected, normalized)
    result["ignored_metadata_differences"] = ignored
    return result


def compare_observation_sequences(
    expected: Sequence[Observation], actual: Sequence[Observation]
) -> dict[str, Any]:
    first = next(
        (
            index
            for index, (left, right) in enumerate(zip(expected, actual, strict=False))
            if left != right
        ),
        None,
    )
    exact = len(expected) == len(actual) and first is None
    return {
        "exact": exact,
        "expected_length": len(expected),
        "actual_length": len(actual),
        "first_difference": first,
    }


def observed_outcome(before: Observation, after: Observation) -> str:
    if after.state.upper() == "GAME_OVER":
        return "game_over"
    if after.levels_completed > before.levels_completed or after.state.upper() == "WIN":
        return "level_complete"
    return "ongoing"


def engine_commitment(environment_files: Path, game_id: str) -> dict[str, Any]:
    """Commit to privileged local build bytes without exposing their contents."""

    short_id = game_id.split("-", 1)[0]
    roots = sorted((environment_files / short_id).glob("*/metadata.json"))
    if not roots:
        raise FileNotFoundError(f"no local build for {game_id!r} under {environment_files}")
    exact = [path for path in roots if game_id in path.read_text()]
    metadata = exact[0] if exact else roots[-1]
    build = metadata.parent
    files = []
    for path in sorted(item for item in build.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(build).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
            }
        )
    return {
        "game_id": game_id,
        "build_commitment": commitment_json(files),
        "files": files,
        "privileged_contents_exposed_to_solver": False,
    }


def commitment_json(value: Any) -> str:
    return commitment_bytes(canonical_json(value))


def commitment_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json(value) + b"\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _find_game_workspace(run_dir: Path, game_id: str) -> Path:
    candidates = [run_dir / "workspace" / game_id, run_dir / "workspace" / game_id[:4]]
    for candidate in candidates:
        if (candidate / "world_model.py").is_file():
            return candidate
    found = list((run_dir / "workspace").glob("*/world_model.py"))
    if len(found) == 1:
        return found[0].parent
    raise FileNotFoundError(f"could not find one saved world_model.py under {run_dir}")


def _checkpoint_index(root: Path) -> dict[int, tuple[dict[str, Any], ...]]:
    by_sequence: dict[int, list[dict[str, Any]]] = {}
    if not root.is_dir():
        return {}
    for request_path in sorted((root / "calls").glob("*/request.json")):
        request = read_json(request_path)
        point = request.get("decision_point") or {}
        if point.get("phase") != "decision" or type(point.get("sequence")) is not int:
            continue
        record = {
            "call_index": int(request_path.parent.name),
            "source": request.get("source"),
            "request_payload_commitment": request.get("request_payload_commitment"),
            "workspace_commitment": request.get("workspace_commitment"),
        }
        by_sequence.setdefault(point["sequence"], []).append(record)
    return {key: tuple(value) for key, value in by_sequence.items()}


def _workspace_for_packet(
    checkpoint_root: Path,
    refs: Sequence[Mapping[str, Any]],
    fallback: Path,
) -> tuple[Path, str, TemporaryDirectory[str] | None]:
    if not refs:
        return fallback, "retrospective_final_workspace", None
    call_index = int(refs[-1]["call_index"])
    temporary = TemporaryDirectory(prefix="failure-atlas-checkpoint-")
    destination = Path(temporary.name) / "workspace"
    _restore_checkpoint(checkpoint_root, call_index, destination)
    return destination, "decision_checkpoint", temporary


def _restore_checkpoint(root: Path, call_index: int, destination: Path) -> None:
    manifest = read_json(root / "calls" / f"{call_index:06d}" / "workspace.json")
    destination.mkdir(parents=True)
    for entry in manifest["directories"]:
        (destination / entry["path"]).mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        content = (root / "blobs" / "sha256" / entry["sha256"]).read_bytes()
        if len(content) != entry["size"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"corrupt checkpoint blob {entry['sha256']}")
        path = destination / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _bundle_model(workspace: Path, root: Path) -> tuple[str, Path]:
    sources = sorted(path for path in workspace.glob("*.py") if path.is_file())
    if not any(path.name == "world_model.py" for path in sources):
        raise FileNotFoundError(f"world_model.py missing from {workspace}")
    inventory = [
        {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sources
    ]
    key = commitment_json(inventory)
    destination = root / key.removeprefix("sha256:")
    if not destination.exists():
        destination.mkdir()
        for path in sources:
            shutil.copy2(path, destination / path.name)
        write_json(destination / "manifest.json", {"commitment": key, "files": inventory})
    return key, destination


def _copy_checkpoint_evidence(
    root: Path,
    refs: Sequence[Mapping[str, Any]],
    destination: Path,
) -> list[dict[str, Any]]:
    copied = []
    for ref in refs:
        call_index = int(ref["call_index"])
        source = root / "calls" / f"{call_index:06d}"
        target = destination / "calls" / f"{call_index:06d}"
        target.mkdir(parents=True, exist_ok=True)
        for name in ("request.json", "response.json", "workspace.json"):
            if (source / name).is_file():
                shutil.copy2(source / name, target / name)
        manifest = read_json(source / "workspace.json")
        for entry in manifest["files"]:
            blob_target = destination / "blobs" / "sha256" / entry["sha256"]
            if not blob_target.exists():
                blob_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / "blobs" / "sha256" / entry["sha256"], blob_target)
        copied.append({**ref, "corpus_call": f"checkpoints/calls/{call_index:06d}"})
    return copied


def _load_world_model(workspace: Path) -> Any:
    # Candidate modules use absolute sibling imports (``import wmlib``).  Clear
    # only those evaluation names before loading the next content-addressed
    # bundle; no host or solver process shares this interpreter.
    sibling_names = {path.stem for path in workspace.glob("*.py")}
    for name in sibling_names:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(workspace))
    try:
        spec = importlib.util.spec_from_file_location("world_model", workspace / "world_model.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load world model from {workspace}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["world_model"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(workspace))


def _score_variants(model: WorldModel, state: Any, actual: np.ndarray) -> dict[str, Any]:
    try:
        variants = model.variants(state)
    except Exception as error:  # noqa: BLE001 - preserve model diagnostics
        return {"count": 0, "accepted": False, "error": {"type": type(error).__name__, "message": str(error)}}
    scores = [score_grid(value, actual) for value in variants]
    return {
        "count": len(scores),
        "accepted": any(score["canonical_exact"] for score in scores),
        "best_unknown_penalized_cell_accuracy": max(
            (score["unknown_penalized_cell_accuracy"] for score in scores),
            default=None,
        ),
    }


def _ratio(numerator: int, denominator: int, *, empty: bool) -> float:
    if denominator:
        return numerator / denominator
    return 1.0 if empty else 0.0


def _only_unknown_error(score: Mapping[str, Any]) -> bool:
    """Unknown cells lower coverage but are not claimed model contradictions."""

    return (
        score["shape_predicted"] == score["shape_actual"]
        and score["differing_claimed_cells"] == 0
    )


def _failure_kind(branch: Mapping[str, Any]) -> str | None:
    if branch.get("status") != "scored":
        return None
    score = branch["score"]
    if not score["outcome"]["exact"]:
        return "termination"
    grid = score.get("grid")
    if grid and grid["differing_claimed_cells"]:
        return "claimed_render"
    if grid and grid["unknown_cells"]:
        return "unknown_coverage"
    return None


def _attach_shortest_prefixes(results: Sequence[dict[str, Any]]) -> None:
    shortest: dict[tuple[tuple[Any, ...], str], int] = {}
    for result in sorted(results, key=lambda value: int(value["sequence"])):
        for branch in result.get("branches", []):
            kind = _failure_kind(branch)
            if kind is None:
                continue
            action = Action.from_json(branch["action"])
            key = (action.key(), kind)
            shortest.setdefault(key, int(result["sequence"]))
            branch["shortest_failing_prefix"] = {
                "actions": shortest[key],
                "retained_failure_kind": kind,
                "scope": "same instantiated action among replayed packets",
            }


def _replay_summary(
    corpus: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    engine_identity: Mapping[str, Any],
) -> dict[str, Any]:
    branches = [branch for result in results for branch in result.get("branches", [])]
    scored = [branch for branch in branches if branch.get("status") == "scored"]
    failures = [branch for branch in scored if branch["score"]["failure"]]
    unknown = [
        branch
        for branch in scored
        if branch["score"].get("grid") and branch["score"]["grid"]["unknown_cells"]
    ]
    return {
        "schema": REPLAY_SCHEMA,
        "game_id": corpus["game_id"],
        "seed": corpus["seed"],
        "engine": dict(engine_identity),
        "packets_requested": len(results),
        "packets_engine_parity_valid": sum(
            result.get("engine_prefix_parity", {}).get("exact", False) for result in results
        ),
        "branches_total": len(branches),
        "branches_scored": len(scored),
        "branches_failed": len(failures),
        "branches_using_unknown": len(unknown),
        "claim_ceiling": (
            "one-step saved-model mechanics at parity-checked visited states in this "
            "local engine build; not actor uptake, gameplay benefit, or transfer"
        ),
    }
