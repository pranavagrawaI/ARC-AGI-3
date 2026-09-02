"""Evaluator-hidden, one-decision replays through the production Actor.

The replay substitutes only the consequence sink: ``TychoNooaAgent.act`` runs
normally and returns an ``ActionDecision``, but no ARC client receives it.  The
evaluator key is committed before execution and loaded only after the Actor has
finished.

The current production Python workbench can read the entire workspace without
reporting individual evidence reads.  Reports therefore distinguish exact
native-file observations from partial Python-access observations instead of
claiming support that the trace cannot establish.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from nooa.unifiedllm import UnifiedLLM
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tycho_nooa.agent import TychoNooaAgent
from tycho_nooa.models import ActionDecision, GameTurn
from tycho_nooa.modes import MODE_ENV, served_mode
from tycho_nooa.strategy import NativeToolConfig
from tycho_nooa.treatments import TREATMENT_ENV, treatment_from_env

PACKET_SCHEMA = "tycho-nooa-production-actor-replay-v1"
REPORT_SCHEMA = "tycho-nooa-production-actor-replay-report-v1"
FAILURE_CLASSES = (
    "remote_contradiction",
    "repeated_state_action",
    "multi_transition_computation",
    "goal_discrimination",
    "plan_assumption_repair",
    "evidence_sufficiency",
)

_EVIDENCE_REF = re.compile(r"^L(?P<level>\d+):T(?P<turn>\d+)$")
_TURN_PATH = re.compile(
    r"(?:^|/)level_(?P<level>\d+)/turn_(?P<turn>\d+)\.(?:json|txt)$"
)


class ProductionReplayOracle(BaseModel):
    """Evaluator-only facts; never copied into the Actor workspace or prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_purpose: Literal["decision_quality", "scripted_reachability"] = (
        "decision_quality"
    )
    failure_class: Literal[
        "remote_contradiction",
        "repeated_state_action",
        "multi_transition_computation",
        "goal_discrimination",
        "plan_assumption_repair",
        "evidence_sufficiency",
    ]
    acceptable_terminal_outcomes: tuple[ActionDecision, ...] = Field(min_length=1)
    acceptable_support_sets: tuple[tuple[str, ...], ...] = Field(min_length=1)
    known_relevant_evidence: tuple[str, ...] = ()
    known_distractor_evidence: tuple[str, ...] = ()
    required_computation: Literal["none", "python"]
    historical_failure: str = Field(min_length=1)

    @field_validator(
        "known_relevant_evidence",
        "known_distractor_evidence",
        mode="after",
    )
    @classmethod
    def validate_refs(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            _parse_ref(value)
        return values

    @field_validator("acceptable_support_sets", mode="after")
    @classmethod
    def validate_support_sets(
        cls, values: tuple[tuple[str, ...], ...]
    ) -> tuple[tuple[str, ...], ...]:
        for support_set in values:
            if not support_set:
                raise ValueError("support sets cannot be empty")
            for value in support_set:
                _parse_ref(value)
        return values


@dataclass(frozen=True)
class PreparedProductionReplay:
    path: Path
    manifest: dict[str, Any]
    oracle: ProductionReplayOracle


def prepare_production_replay_packet(
    *,
    source_workspace: Path,
    game_id: str,
    previous_turn: GameTurn,
    replay_turn: GameTurn,
    oracle: ProductionReplayOracle,
    destination: Path,
    mode: str,
    treatment: str,
) -> dict[str, Any]:
    """Freeze one real workspace boundary and a separately committed evaluator."""

    if destination.exists():
        raise FileExistsError(destination)
    if not source_workspace.is_dir():
        raise FileNotFoundError(source_workspace)
    mode_policy = served_mode(mode)
    treatment_policy = treatment_from_env(
        mode_policy,
        {TREATMENT_ENV: treatment},
    )
    if replay_turn.level != previous_turn.level:
        raise ValueError(
            "phase-one replay currently requires one within-level decision"
        )
    if replay_turn.turn_in_level != previous_turn.turn_in_level + 1:
        raise ValueError("previous_turn must immediately precede replay_turn")

    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        source = temporary / "source"
        shutil.copytree(
            source_workspace,
            source / "workspace",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        _write_json(
            source / "previous-turn.json", previous_turn.model_dump(mode="json")
        )
        _write_json(source / "replay-turn.json", replay_turn.model_dump(mode="json"))
        evaluator = temporary / "evaluator"
        evaluator.mkdir()
        key = oracle.model_dump(mode="json")
        _write_json(evaluator / "key.json", key)
        manifest = {
            "schema": PACKET_SCHEMA,
            "game_id": game_id,
            "mode": mode_policy.name,
            "treatment": treatment_policy.name,
            "workspace_commitment": _tree_commitment(source / "workspace"),
            "previous_turn_commitment": _commitment(
                previous_turn.model_dump(mode="json")
            ),
            "replay_turn_commitment": _commitment(replay_turn.model_dump(mode="json")),
            "evaluator_key_commitment": _commitment(key),
            "claim_ceiling": (
                "one production-controller decision from a frozen durable workspace; "
                "the returned ActionDecision is evaluated but never sent to ARC; exact "
                "cognitive continuation is not claimed unless the source workspace also "
                "contains the production Actor's durable event history"
            ),
        }
        _write_json(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validate_production_replay_packet(destination)
    return manifest


def validate_production_replay_packet(path: Path) -> PreparedProductionReplay:
    """Validate source and evaluator commitments without trusting path names."""

    manifest = _validate_source(path)
    oracle = _load_oracle(path, manifest)
    return PreparedProductionReplay(path=path, manifest=manifest, oracle=oracle)


async def execute_production_replay(
    packet: Path,
    destination: Path,
    *,
    llm: UnifiedLLM,
    max_tool_steps: int = 25,
) -> dict[str, Any]:
    """Run the actual production Actor and replace only ARC consequence execution."""

    manifest = _validate_source(packet)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    runtime_root = destination / "runtime"
    workspace = runtime_root / manifest["game_id"].split("-")[0]
    workspace.parent.mkdir(parents=True)
    shutil.copytree(packet / "source" / "workspace", workspace)
    previous_turn = GameTurn.model_validate(
        _read_object(packet / "source" / "previous-turn.json")
    )
    replay_turn = GameTurn.model_validate(
        _read_object(packet / "source" / "replay-turn.json")
    )
    trace_path = destination / "tool-trace.jsonl"
    before = _tree_commitment(workspace)

    with _runtime_environment(manifest["mode"], manifest["treatment"]):
        agent = TychoNooaAgent(
            llm=llm,
            game_id=manifest["game_id"],
            workspace_root=runtime_root,
            available_actions=replay_turn.available_actions,
            tool_trace_path=trace_path,
            strategy_config=NativeToolConfig(max_tool_steps=max_tool_steps),
            trigger_builder=False,
            resume_workspace=True,
            previous_turn=previous_turn,
        )
        decision = await agent.act(replay_turn)

    # The evaluator does not enter the model-facing object graph while calls are live.
    oracle = _load_oracle(packet, manifest)
    report = _evaluate(
        manifest=manifest,
        oracle=oracle,
        decision=decision,
        replay_turn=replay_turn,
        tool_records=_jsonl_records(trace_path),
        llm_records=_jsonl_records(destination / "llm-trace.jsonl"),
        workspace_before=before,
        workspace_after=_tree_commitment(workspace),
    )
    _write_json(destination / "report.json", report)
    return report


def _evaluate(
    *,
    manifest: Mapping[str, Any],
    oracle: ProductionReplayOracle,
    decision: ActionDecision,
    replay_turn: GameTurn,
    tool_records: list[dict[str, Any]],
    llm_records: list[dict[str, Any]],
    workspace_before: str,
    workspace_after: str,
) -> dict[str, Any]:
    disclosures = [(0, f"L{replay_turn.level}:T{replay_turn.turn_in_level}")]
    disclosures.extend(_native_file_disclosures(tool_records))
    python_calls = [row for row in tool_records if row.get("name") == "run_python"]
    completeness = (
        "partial_python_access" if python_calls else "exact_for_native_file_reads"
    )
    unique_refs = {ref for _, ref in disclosures}
    relevant = unique_refs & set(oracle.known_relevant_evidence)
    distractors = unique_refs & set(oracle.known_distractor_evidence)
    acquisitions = [ref for call, ref in disclosures if call > 0]
    reacquisitions = len(acquisitions) - len(set(acquisitions))
    sufficiency_call = _first_sufficiency_call(disclosures, oracle)
    terminal_record = next(
        (row for row in reversed(tool_records) if row.get("name") == "take_actions"),
        None,
    )
    terminal_call = terminal_record.get("call_index") if terminal_record else None
    exact = completeness == "exact_for_native_file_reads"
    grounding = (
        "observed"
        if exact and sufficiency_call is not None
        else "unsupported"
        if exact
        else "unobservable"
    )
    post_sufficiency = None
    if exact and sufficiency_call is not None and terminal_call is not None:
        post_sufficiency = sum(
            1
            for row in tool_records
            if sufficiency_call < int(row.get("call_index", -1)) < terminal_call
        )
    token_totals = _usage_totals(llm_records)
    observed_total = len(unique_refs)
    return {
        "schema": REPORT_SCHEMA,
        "packet_commitment": _commitment(dict(manifest)),
        "claim_ceiling": manifest["claim_ceiling"],
        "controller": "tycho_nooa.agent.TychoNooaAgent",
        "failure_class": oracle.failure_class,
        "evaluation_purpose": oracle.evaluation_purpose,
        "historical_failure": oracle.historical_failure,
        "environment_actions_executed": 0,
        "decision": decision.model_dump(mode="json"),
        "select": {
            "observation_completeness": completeness,
            "observed_relevant_disclosures": len(relevant),
            "observed_distractor_disclosures": len(distractors),
            "observed_unique_disclosures": observed_total,
            "reacquisitions": reacquisitions,
            "useful_evidence_fraction": (
                len(relevant) / observed_total if observed_total else None
            ),
            # The production Actor does not yet expose immutable evidence grants.
            "evidence_reuse_rate": None,
        },
        "compute": {
            "required": oracle.required_computation,
            "python_calls": len(python_calls),
            "python_used_unnecessarily": (
                oracle.required_computation == "none" and bool(python_calls)
            ),
            # Tool success proves execution, not semantic correctness of cognition.
            "computation_correct": None,
        },
        "commit": {
            "terminal_result_accepted": terminal_record is not None,
            "terminal_correct": decision in oracle.acceptable_terminal_outcomes,
            "grounding": grounding,
            "premature_commitment": (sufficiency_call is None if exact else None),
            "first_sufficiency_tool_call": sufficiency_call,
            "post_sufficiency_tool_calls": post_sufficiency,
            # ActionDecision has no evidence/rationale payload, so this milestone
            # cannot be separated from the accepted terminal call yet.
            "candidate_answer_formed_call": None,
            "terminal_return_call": terminal_call,
        },
        "resource": {
            "provider_calls": len(llm_records),
            **token_totals,
        },
        "workspace": {
            "commitment_before": workspace_before,
            "commitment_after": workspace_after,
        },
    }


def _native_file_disclosures(
    records: Sequence[Mapping[str, Any]],
) -> list[tuple[int, str]]:
    disclosures: list[tuple[int, str]] = []
    for record in records:
        if record.get("name") != "read_file" or record.get("status") != "complete":
            continue
        arguments = record.get("arguments")
        if not isinstance(arguments, Mapping):
            continue
        path = arguments.get("path")
        match = _TURN_PATH.search(path) if isinstance(path, str) else None
        if match is None:
            continue
        disclosures.append(
            (
                int(record["call_index"]),
                f"L{int(match['level'])}:T{int(match['turn'])}",
            )
        )
    return disclosures


def _first_sufficiency_call(
    disclosures: Sequence[tuple[int, str]],
    oracle: ProductionReplayOracle,
) -> int | None:
    observed: set[str] = set()
    for call_index, ref in sorted(disclosures):
        observed.add(ref)
        if any(
            set(support).issubset(observed)
            for support in oracle.acceptable_support_sets
        ):
            return call_index
    return None


def _usage_totals(records: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    totals: dict[str, int | float] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "cost_usd": 0.0,
    }
    for record in records:
        totals["input_tokens"] += int(record.get("prompt_tokens") or 0)
        totals["output_tokens"] += int(record.get("completion_tokens") or 0)
        totals["cached_tokens"] += int(record.get("cached_tokens") or 0)
        totals["cost_usd"] += float(record.get("cost_usd") or 0.0)
    return totals


def _validate_source(path: Path) -> dict[str, Any]:
    manifest = _read_object(path / "manifest.json")
    if manifest.get("schema") != PACKET_SCHEMA:
        raise ValueError("unsupported production replay packet schema")
    served_mode(str(manifest.get("mode")))
    checks = {
        "workspace_commitment": _tree_commitment(path / "source" / "workspace"),
        "previous_turn_commitment": _commitment(
            _read_object(path / "source" / "previous-turn.json")
        ),
        "replay_turn_commitment": _commitment(
            _read_object(path / "source" / "replay-turn.json")
        ),
    }
    for field, actual in checks.items():
        if manifest.get(field) != actual:
            raise ValueError(f"stale production replay packet: {field}")
    return manifest


def _load_oracle(path: Path, manifest: Mapping[str, Any]) -> ProductionReplayOracle:
    key = _read_object(path / "evaluator" / "key.json")
    if manifest.get("evaluator_key_commitment") != _commitment(key):
        raise ValueError("stale production replay packet: evaluator_key_commitment")
    return ProductionReplayOracle.model_validate(key)


@contextmanager
def _runtime_environment(mode: str, treatment: str):
    prior = {name: os.environ.get(name) for name in (MODE_ENV, TREATMENT_ENV)}
    os.environ[MODE_ENV] = mode
    os.environ[TREATMENT_ENV] = treatment
    try:
        yield
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _parse_ref(value: str) -> tuple[int, int]:
    match = _EVIDENCE_REF.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid evidence reference: {value!r}")
    return int(match["level"]), int(match["turn"])


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _tree_commitment(root: Path) -> str:
    if not root.is_dir():
        raise FileNotFoundError(root)
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return _commitment(records)


def _commitment(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
