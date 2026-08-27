"""Mine when evidence stops changing behavior in preserved Tycho/NOOA runs.

The miner is deliberately retrospective and zero-call.  It joins host-authoritative
transcripts, tool traces, trigger traces, exact decision checkpoints, and accounting
records.  Model-authored prose is a signal, never promoted to host observation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
DETECTOR_VERSION = "2026-08-21.1"
STAGES = (
    "candidate_objective",
    "discriminator_named",
    "discriminator_executed",
    "observation_interpreted",
    "hypothesis_revised",
    "target_encoded",
    "plan_attempted",
    "plan_valid",
    "action_followed_plan",
    "action_executed_by_host",
    "terminal_consequence",
    "knowledge_consolidated",
)


@dataclass(frozen=True)
class RunSpec:
    key: str
    path: Path
    label: str
    evidence_class: str


def default_corpus(root: Path) -> list[RunSpec]:
    runs = root / "runs"
    return [
        RunSpec(
            "ar25_goal_gate",
            runs / "tycho-nooa-goal-gate-ar25-20260819T121859Z",
            "ar25 goal gate",
            "legacy_trace",
        ),
        RunSpec(
            "ar25_symmetry_uptake",
            runs / "tycho-nooa-symmetry-uptake-ar25-20260820T101248Z",
            "ar25 symmetry uptake",
            "legacy_trace",
        ),
        RunSpec(
            "ar25_symmetry_replication",
            runs / "tycho-nooa-symmetry-replication-ar25-20260820T141506Z",
            "ar25 symmetry replication",
            "legacy_trace",
        ),
        RunSpec(
            "ls20_gemma_feasibility",
            runs / "tycho-nooa-gemma4-31b-feasibility-20260821T073930Z" / "game",
            "LS20 Gemma feasibility",
            "exact_checkpoint",
        ),
        RunSpec(
            "ls20_gemma_reasoning",
            runs / "tycho-nooa-gemma4-31b-reasoning-capture-20260821T081012Z" / "game",
            "LS20 Gemma reasoning capture",
            "exact_checkpoint",
        ),
    ]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact(value: Any, limit: int = 420) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _event(
    *,
    run_key: str,
    sequence: int,
    owner: str,
    authority: str,
    kind: str,
    text: str,
    evidence_ref: str,
    call_index: int | None = None,
) -> dict[str, Any]:
    return {
        "run_key": run_key,
        "sequence": int(sequence),
        "owner": owner,
        "authority": authority,
        "kind": kind,
        "text": text,
        "evidence_ref": evidence_ref,
        "call_index": call_index,
    }


def _sequence_index(run_path: Path) -> dict[tuple[int, int], int]:
    index: dict[tuple[int, int], int] = {}
    for row in read_jsonl(run_path / "transcript.jsonl"):
        if row.get("sequence") is None or row.get("level") is None or row.get("turn_in_level") is None:
            continue
        index[(int(row["level"]), int(row["turn_in_level"]))] = int(row["sequence"])
    return index


def _tool_sequence_maps(
    run_path: Path,
    sequence_index: dict[tuple[int, int], int] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_owner: dict[str, list[dict[str, Any]]] = {
        "actor": [],
        "builder": [],
        "scribe": [],
    }
    for owner, name in (("actor", "tool-trace.jsonl"), ("builder", "builder-tool-trace.jsonl")):
        for row_number, row in enumerate(read_jsonl(run_path / name), 1):
            row = dict(row)
            row["_trace_name"] = name
            row["_line_number"] = row_number
            effective_owner = str(row.get("source", owner))
            row["owner"] = effective_owner
            level_turn = (int(row.get("level", 0)), int(row.get("turn_in_level", 0)))
            if not isinstance(row.get("sequence"), int):
                row["sequence"] = (sequence_index or {}).get(level_turn, level_turn[1])
            by_owner.setdefault(effective_owner, []).append(row)
            if row.get("tool_call_id"):
                by_id[str(row["tool_call_id"])] = row
    return by_id, by_owner


def _checkpoint_sequences(run_path: Path) -> dict[int, int]:
    result: dict[int, int] = {}
    calls = run_path / "decision-checkpoints" / "calls"
    if not calls.exists():
        return result
    for request_path in calls.glob("*/request.json"):
        request = read_json(request_path)
        if request.get("call_index") is not None and request.get("decision_point", {}).get("sequence") is not None:
            result[int(request["call_index"])] = int(request["decision_point"]["sequence"])
    return result


def _join_llm_sequences(run_path: Path) -> tuple[list[dict[str, Any]], str]:
    """Attach each model call to a decision sequence without reading prompt text."""
    calls = read_jsonl(run_path / "llm-trace.jsonl")
    legacy_exact = _checkpoint_sequences(run_path)
    by_id, by_owner = _tool_sequence_maps(run_path, _sequence_index(run_path))
    cursors = defaultdict(int)
    last = defaultdict(int)
    direct = bool(calls) and all(isinstance(call.get("sequence"), int) for call in calls)
    alignment = (
        "exact_llm_decision_point"
        if direct
        else "legacy_call_index_checkpoint"
        if legacy_exact
        else "tool_call_join"
    )

    for call in calls:
        source = call.get("source", "actor")
        call_index = int(call.get("call_index", 0))
        sequence = int(call["sequence"]) if isinstance(call.get("sequence"), int) else None
        if sequence is None:
            sequence = legacy_exact.get(call_index)
        if sequence is None:
            tool_rows = [by_id[x] for x in call.get("tool_call_ids", []) if x in by_id]
            if tool_rows:
                sequence = int(tool_rows[0]["sequence"])
            elif call.get("tool_calls"):
                queue = by_owner.get(source, [])
                cursor = cursors[source]
                matched = None
                for expected_name in call.get("tool_calls", []):
                    while cursor < len(queue):
                        candidate = queue[cursor]
                        cursor += 1
                        if candidate.get("name") == expected_name:
                            matched = candidate
                            break
                    if matched is not None:
                        break
                cursors[source] = cursor
                sequence = int(matched["sequence"]) if matched else last[source]
            else:
                sequence = last[source]
        call["sequence"] = int(sequence)
        last[source] = int(sequence)
    return calls, alignment


def extract_run(spec: RunSpec, root: Path) -> dict[str, Any]:
    if not spec.path.exists():
        raise FileNotFoundError(spec.path)
    summary = read_json(spec.path / "summary.json")
    transcript = read_jsonl(spec.path / "transcript.jsonl")
    sequence_index = _sequence_index(spec.path)
    _, tool_by_owner = _tool_sequence_maps(spec.path, sequence_index)
    llm_calls, spend_alignment = _join_llm_sequences(spec.path)
    trigger_rows = read_jsonl(spec.path / "trigger-builder-trace.jsonl")

    events: list[dict[str, Any]] = []
    input_files: set[Path] = {spec.path / "summary.json"}
    actions: dict[int, dict[str, Any]] = {}
    turns: set[int] = set()
    max_sequence = 0
    previous_levels = 0
    terminal_sequences: list[int] = []

    transcript_path = spec.path / "transcript.jsonl"
    if transcript_path.exists():
        input_files.add(transcript_path)
    for row_number, row in enumerate(transcript, 1):
        if row.get("sequence") is None:
            continue
        sequence = int(row["sequence"])
        max_sequence = max(max_sequence, sequence)
        if row.get("type") == "turn":
            turns.add(sequence)
        elif row.get("type") == "action_result":
            turns.add(sequence)
            decision = row.get("decision") or {}
            after = row.get("after") or row.get("observation") or {}
            levels = int(after.get("levels_completed", row.get("levels_completed", previous_levels)) or 0)
            terminal = levels > previous_levels or after.get("state") in {"WON", "LEVEL_COMPLETE"}
            if terminal:
                terminal_sequences.append(sequence)
            previous_levels = max(previous_levels, levels)
            actions[sequence] = {
                "selected_action": decision.get("action"),
                "host_executed": True,
                "levels_completed": levels,
                "terminal": terminal,
                "evidence_ref": f"{rel(transcript_path, root)}:{row_number}",
            }
            events.append(_event(
                run_key=spec.key,
                sequence=sequence,
                owner="host",
                authority="host_observation",
                kind="action_result",
                text=json.dumps({"decision": decision, "levels_completed": levels, "state": after.get("state")}, sort_keys=True),
                evidence_ref=f"{rel(transcript_path, root)}:{row_number}",
            ))

    # Tool arguments are model-authored intent; results are mechanically produced evidence.
    for owner, rows in tool_by_owner.items():
        for row in rows:
            path = spec.path / str(row["_trace_name"])
            row_number = int(row["_line_number"])
            if path.exists():
                input_files.add(path)
            sequence = int(row["sequence"])
            max_sequence = max(max_sequence, sequence)
            text = json.dumps({
                "name": row.get("name"),
                "arguments": row.get("arguments"),
                "status": row.get("status"),
                "result": row.get("result"),
            }, ensure_ascii=False, sort_keys=True)
            events.append(_event(
                run_key=spec.key,
                sequence=sequence,
                owner=owner,
                authority="mixed_tool_record",
                kind="tool_call",
                text=text,
                evidence_ref=f"{rel(path, root)}:{row_number}",
                call_index=row.get("call_index"),
            ))
            if owner == "actor" and row.get("name") == "take_action" and row.get("status") == "complete":
                action = (row.get("arguments") or {}).get("action")
                actions.setdefault(sequence, {})
                actions[sequence].setdefault("selected_action", action)
                actions[sequence]["selection_evidence_ref"] = f"{rel(path, root)}:{row_number}"

    spend_by_sequence: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "actor_usd": 0.0,
            "builder_usd": 0.0,
            "scribe_usd": 0.0,
            "actor_calls": 0,
            "builder_calls": 0,
            "scribe_calls": 0,
        }
    )
    llm_path = spec.path / "llm-trace.jsonl"
    if llm_path.exists():
        input_files.add(llm_path)
    for row_number, call in enumerate(llm_calls, 1):
        sequence = int(call["sequence"])
        max_sequence = max(max_sequence, sequence)
        source = call.get("source", "actor")
        key = f"{source}_usd"
        calls_key = f"{source}_calls"
        spend_by_sequence[sequence][key] = spend_by_sequence[sequence].get(key, 0.0) + float(call.get("cost_usd") or 0.0)
        spend_by_sequence[sequence][calls_key] = spend_by_sequence[sequence].get(calls_key, 0) + 1
        authored = "\n".join(str(call.get(k) or "") for k in ("reasoning", "content"))
        if authored.strip():
            events.append(_event(
                run_key=spec.key,
                sequence=sequence,
                owner=source,
                authority="model_authored",
                kind="model_response",
                text=authored,
                evidence_ref=f"{rel(llm_path, root)}:{row_number}",
                call_index=call.get("call_index"),
            ))

    trigger_path = spec.path / "trigger-builder-trace.jsonl"
    if trigger_path.exists():
        input_files.add(trigger_path)
    consolidated_sequences: list[int] = []
    for row_number, row in enumerate(trigger_rows, 1):
        level_turn = (int(row.get("level", 0)), int(row.get("turn_in_level", 0)))
        sequence = sequence_index.get(level_turn, level_turn[1])
        max_sequence = max(max_sequence, sequence)
        events.append(_event(
            run_key=spec.key,
            sequence=sequence,
            owner="orchestrator",
            authority="host_control_trace",
            kind=str(row.get("event", "trigger")),
            text=json.dumps(row, ensure_ascii=False, sort_keys=True),
            evidence_ref=f"{rel(trigger_path, root)}:{row_number}",
        ))
        if row.get("outcome_checkpoint_required") and row.get("outcome_checkpoint_resolved"):
            consolidated_sequences.append(sequence)

    # An exact call can select an action even when the outer host never records it.
    for sequence, action in actions.items():
        action.setdefault("host_executed", False)
        action.setdefault("terminal", False)
        action.setdefault("levels_completed", 0)

    signals = detect_signals(events, terminal_sequences, consolidated_sequences)
    decisions = build_decisions(
        spec=spec,
        max_sequence=max(max_sequence, max(turns or {0})),
        turns=turns,
        actions=actions,
        signals=signals,
        events=events,
        spend_by_sequence=spend_by_sequence,
        spend_known=llm_path.exists(),
    )

    return {
        "key": spec.key,
        "label": spec.label,
        "path": rel(spec.path, root),
        "game_id": summary.get("game_id"),
        "seed": summary.get("seed"),
        "evidence_class": spec.evidence_class,
        "summary": summary,
        "spend_alignment": spend_alignment if llm_path.exists() else "unavailable",
        "input_files": [
            {"path": rel(path, root), "sha256": sha256_file(path)}
            for path in sorted(input_files)
            if path.exists()
        ],
        "events": events,
        "signals": signals,
        "decisions": decisions,
        "terminal_sequences": terminal_sequences,
        "consolidated_sequences": consolidated_sequences,
        "max_sequence": max_sequence,
    }


def detect_signals(
    events: list[dict[str, Any]],
    terminal_sequences: list[int],
    consolidated_sequences: list[int],
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []

    def add(stage: str, event: dict[str, Any], detector: str, strength: str = "trace") -> None:
        key = (stage, event["sequence"], event["evidence_ref"], detector)
        if any((s["stage"], s["sequence"], s["evidence_ref"], s["detector"]) == key for s in signals):
            return
        signals.append({
            "stage": stage,
            "sequence": event["sequence"],
            "owner": event["owner"],
            "authority": event["authority"],
            "detector": detector,
            "strength": strength,
            "snippet": compact(event["text"]),
            "evidence_ref": event["evidence_ref"],
            "call_index": event.get("call_index"),
        })

    candidate = re.compile(
        r"(?is)(?:##\s*hypotheses.{0,500}?(?:goal\s*:|win\s+condition)|goal\s*(?:is|:|may|might|could|likely|probably)|"
        r"win\s+condition.{0,160}?|"
        r"outcome\s*:.*hypotheses\s+still\s+open).{0,240}?"
        r"(?:reach|overlap|match|align|symmetr|clear|exit|target|configuration|object)",
    )
    explicit_discriminator = re.compile(
        r"(?is)(?:discriminat|distinguish|falsif|would\s+(?:confirm|support|refute|reject)|"
        r"test(?:ing)?\s+(?:the\s+)?(?:goal|objective|outcome|hypothesis))",
    )
    interpreted = re.compile(
        r"(?is)(?:this|result|observation|therefore|thus|so)\s+(?:confirms?|supports?|contradicts?|rejects?|means?)|"
        r"(?:same|congruent).{0,40}?(?:true|confirmed|established)",
    )
    revised = re.compile(
        r"(?is)(?:goal|objective|hypothesis).{0,80}?(?:confirmed|promoted|rejected|ruled\s+out)|"
        r"(?:therefore|thus).{0,80}?(?:goal|target)\s+is",
    )
    target = re.compile(
        r"(?is)(?:outcome\s*:\s*level_complete\s+if|goal\s+position|target\s+(?:bbox|position)|"
        r"displacement\s+(?:computed|=)|steps?\s+(?:down|up|left|right)\s*=)",
    )
    plan = re.compile(
        r"(?is)(?:validated\s+plan|plan\s*:\s*|route\s+(?:calculated|computed)|"
        r"need\s+to\s+move.{0,80}?then|steps?\s+(?:down|up|left|right)\s*=)",
    )
    valid = re.compile(r"(?is)(?:validated\s*=\s*yes|plan(?:ner)?_status\s*:\s*(?:found|plan_found)|canonical_replay=pass)")

    for event in events:
        text = event["text"]
        lower = text.lower()
        if event["authority"] in {"model_authored", "mixed_tool_record"} and candidate.search(text):
            add("candidate_objective", event, "candidate_goal_language")
        is_geometry_probe = (
            event["kind"] == "tool_call"
            and event["owner"] == "actor"
            and ("symmetry.canon" in lower or "symmetry.congruent" in lower or "sym.congruent" in lower)
            and "read_file" not in lower
        )
        if is_geometry_probe:
            add("discriminator_named", event, "executable_geometry_comparison")
            if '"status": "complete"' in lower:
                add("discriminator_executed", event, "completed_geometry_comparison", "mechanical")
            if re.search(r"(?is)(?:same any\?|congruent[^?]{0,30}\?)\s*(?:true|false)", text):
                add("observation_interpreted", event, "comparison_result_materialized", "mechanical")
        elif event["authority"] == "model_authored" and explicit_discriminator.search(text):
            add("discriminator_named", event, "explicit_discriminator_language")
        if event["authority"] == "model_authored" and interpreted.search(text):
            add("observation_interpreted", event, "explicit_result_interpretation")
        if event["authority"] in {"model_authored", "mixed_tool_record"} and revised.search(text):
            add("hypothesis_revised", event, "explicit_hypothesis_revision")
        if event["authority"] in {"model_authored", "mixed_tool_record", "host_control_trace"} and target.search(text):
            add("target_encoded", event, "operational_target_language")
        if event["authority"] in {"model_authored", "mixed_tool_record", "host_control_trace"} and plan.search(text):
            add("plan_attempted", event, "action_route_language")
        if valid.search(text):
            add("plan_valid", event, "explicit_plan_validation", "mechanical")
        if event["kind"] == "action_result":
            add("action_executed_by_host", event, "host_action_result", "host")
        if event["authority"] == "host_control_trace" and re.search(r'"actor_uptake"\s*:\s*true', text):
            add("action_followed_plan", event, "actor_uptake_trace", "host")

    # Host terminal evidence and its durable promotion are never inferred from prose.
    host_template = {
        "owner": "host", "authority": "host_observation", "kind": "terminal",
        "evidence_ref": "derived:host_transcript", "call_index": None,
    }
    for sequence in terminal_sequences:
        event = dict(host_template, sequence=sequence, text=f"terminal consequence at sequence {sequence}")
        add("terminal_consequence", event, "levels_completed_increment", "host")
        # A successful terminal suffix establishes that the actions cohered with a route,
        # but not that a saved world-model plan was correct.
        add("action_followed_plan", event, "terminal_suffix_completed", "posthoc")
        add("plan_valid", event, "terminal_suffix_completed", "posthoc")
    for sequence in consolidated_sequences:
        event = dict(host_template, sequence=sequence, text=f"outcome checkpoint resolved at sequence {sequence}")
        add("knowledge_consolidated", event, "resolved_outcome_checkpoint", "host")
    return sorted(signals, key=lambda row: (row["sequence"], STAGES.index(row["stage"]), row["evidence_ref"]))


def build_decisions(
    *,
    spec: RunSpec,
    max_sequence: int,
    turns: set[int],
    actions: dict[int, dict[str, Any]],
    signals: list[dict[str, Any]],
    events: list[dict[str, Any]],
    spend_by_sequence: dict[int, dict[str, Any]],
    spend_known: bool,
) -> list[dict[str, Any]]:
    by_sequence: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for signal in signals:
        by_sequence[int(signal["sequence"])].append(signal)
    cumulative_stages: set[str] = set()
    actor_spend = builder_spend = scribe_spend = 0.0
    actor_calls = builder_calls = scribe_calls = 0
    rows = []
    for sequence in range(max_sequence + 1):
        current = by_sequence.get(sequence, [])
        cumulative_stages.update(signal["stage"] for signal in current)
        spend = spend_by_sequence.get(sequence, {})
        actor_spend += float(spend.get("actor_usd", 0.0))
        builder_spend += float(spend.get("builder_usd", 0.0))
        scribe_spend += float(spend.get("scribe_usd", 0.0))
        actor_calls += int(spend.get("actor_calls", 0))
        builder_calls += int(spend.get("builder_calls", 0))
        scribe_calls += int(spend.get("scribe_calls", 0))
        action = actions.get(sequence, {})
        rows.append({
            "run_key": spec.key,
            "sequence": sequence,
            "recorded_turn": sequence in turns,
            "selected_action": action.get("selected_action"),
            "host_executed": action.get("host_executed", False),
            "terminal": action.get("terminal", False),
            "stages_new": [stage for stage in STAGES if stage in {s["stage"] for s in current}],
            "stages_seen": [stage for stage in STAGES if stage in cumulative_stages],
            "evidence_refs": sorted(
                {s["evidence_ref"] for s in current}
                | ({action["selection_evidence_ref"]} if action.get("selection_evidence_ref") else set())
                | ({action["evidence_ref"]} if action.get("evidence_ref") else set())
            ),
            "actor_calls_cumulative": actor_calls if spend_known else None,
            "builder_calls_cumulative": builder_calls if spend_known else None,
            "scribe_calls_cumulative": scribe_calls if spend_known else None,
            "actor_spend_usd_cumulative": round(actor_spend, 8) if spend_known else None,
            "builder_spend_usd_cumulative": round(builder_spend, 8) if spend_known else None,
            "scribe_spend_usd_cumulative": round(scribe_spend, 8) if spend_known else None,
        })
    return rows


def classify_run(run: dict[str, Any], horizon: int) -> dict[str, Any]:
    """Return one deduplicated earliest observable failure episode for a run."""
    signals = run["signals"]
    first_by_stage: dict[str, int] = {}
    refs_by_stage: dict[str, list[str]] = defaultdict(list)
    for signal in signals:
        first_by_stage.setdefault(signal["stage"], int(signal["sequence"]))
        refs_by_stage[signal["stage"]].append(signal["evidence_ref"])
    max_sequence = int(run["max_sequence"])
    candidates: list[dict[str, Any]] = []

    candidate_sequence = first_by_stage.get("candidate_objective")
    discriminator_sequence = first_by_stage.get("discriminator_named")
    if candidate_sequence is None and max_sequence >= horizon:
        candidates.append({
            "sequence": horizon,
            "boundary": "observation→candidate objective",
            "failure_family": "goal_mechanism_failure",
            "status": "supported",
            "reason": f"No candidate objective appeared within {horizon} recorded actions.",
            "evidence_refs": [],
        })
    elif candidate_sequence is not None:
        due = candidate_sequence + horizon
        if (discriminator_sequence is None or discriminator_sequence > due) and max_sequence >= due:
            candidates.append({
                "sequence": due,
                "boundary": "candidate objective→named discriminator",
                "failure_family": "goal_mechanism_failure",
                "status": "supported",
                "reason": f"A candidate existed at sequence {candidate_sequence}, but no objective discriminator appeared within {horizon} actions.",
                "evidence_refs": refs_by_stage["candidate_objective"][:3],
            })

    # A completed take_action tool record without a host action_result is a handoff failure,
    # not evidence that the environment rejected the action.
    for decision in run["decisions"]:
        if decision["selected_action"] and not decision["host_executed"]:
            candidates.append({
                "sequence": decision["sequence"],
                "boundary": "selected action→host execution",
                "failure_family": "handoff_failure",
                "status": "supported",
                "reason": f"The actor selected {decision['selected_action']}, but the host recorded no action_result.",
                "evidence_refs": decision["evidence_refs"],
            })
            break

    if run["terminal_sequences"]:
        terminal_sequence = min(run["terminal_sequences"])
        if not any(sequence >= terminal_sequence for sequence in run["consolidated_sequences"]):
            candidates.append({
                "sequence": terminal_sequence,
                "boundary": "terminal consequence→durable consolidation",
                "failure_family": "consolidation_failure",
                "status": "supported",
                "reason": "Host terminal evidence exists, but no resolved outcome checkpoint follows it in the live trace.",
                "evidence_refs": refs_by_stage["terminal_consequence"][:3],
            })

    if not candidates:
        pending_boundary = "candidate objective→named discriminator" if candidate_sequence is not None else "observation→candidate objective"
        return {
            "run_key": run["key"],
            "status": "censored_or_no_supported_failure",
            "boundary": pending_boundary,
            "failure_family": None,
            "sequence": None,
            "reason": "The preserved horizon ends before a configured obligation becomes overdue, or no supported failure was found.",
            "evidence_refs": refs_by_stage.get("candidate_objective", [])[:3],
            "spend": None,
        }
    episode = min(candidates, key=lambda row: row["sequence"])
    decision = run["decisions"][min(episode["sequence"], len(run["decisions"]) - 1)]
    episode.update({
        "run_key": run["key"],
        "spend": {
            "actor_usd": decision["actor_spend_usd_cumulative"],
            "builder_usd": decision["builder_spend_usd_cumulative"],
            "scribe_usd": decision["scribe_spend_usd_cumulative"],
            "actor_calls": decision["actor_calls_cumulative"],
            "builder_calls": decision["builder_calls_cumulative"],
            "scribe_calls": decision["scribe_calls_cumulative"],
        },
    })
    return episode


def positive_controls(horizon: int) -> list[dict[str, Any]]:
    """Scripted controls ensure absence in natural traces is mechanically interpretable."""
    controls = []

    def classify(name: str, stages: dict[str, int], max_sequence: int, selected_without_host: bool = False,
                 terminal_without_consolidation: bool = False) -> dict[str, Any]:
        signals = [
            {"stage": stage, "sequence": sequence, "evidence_ref": f"control:{name}:{stage}"}
            for stage, sequence in stages.items()
        ]
        decisions = [
            {"sequence": sequence, "selected_action": None, "host_executed": True,
             "actor_spend_usd_cumulative": 0.0, "builder_spend_usd_cumulative": 0.0,
             "scribe_spend_usd_cumulative": 0.0, "actor_calls_cumulative": 0,
             "builder_calls_cumulative": 0, "scribe_calls_cumulative": 0,
             "evidence_refs": []}
            for sequence in range(max_sequence + 1)
        ]
        if selected_without_host:
            decisions[-1].update(selected_action="DOWN", host_executed=False)
        run = {
            "key": name, "signals": signals, "max_sequence": max_sequence, "decisions": decisions,
            "terminal_sequences": [max_sequence] if terminal_without_consolidation else [],
            "consolidated_sequences": [],
        }
        return classify_run(run, horizon)

    full = classify(
        "full_funnel",
        {stage: min(index, 2) for index, stage in enumerate(STAGES)},
        max_sequence=2,
    )
    controls.append({"name": "complete funnel is not failed", "expected": None, "observed": full["failure_family"], "pass": full["failure_family"] is None})
    missing = classify("missing_discriminator", {"candidate_objective": 0}, max_sequence=horizon)
    controls.append({"name": "missing discriminator is detected", "expected": "goal_mechanism_failure", "observed": missing["failure_family"], "pass": missing["failure_family"] == "goal_mechanism_failure"})
    handoff = classify("selected_not_emitted", {"candidate_objective": 0}, max_sequence=1, selected_without_host=True)
    controls.append({"name": "selected action without host result is detected", "expected": "handoff_failure", "observed": handoff["failure_family"], "pass": handoff["failure_family"] == "handoff_failure"})
    terminal = classify("terminal_not_consolidated", {"candidate_objective": 0, "discriminator_named": 1}, max_sequence=2, terminal_without_consolidation=True)
    controls.append({"name": "terminal without outcome checkpoint is detected", "expected": "consolidation_failure", "observed": terminal["failure_family"], "pass": terminal["failure_family"] == "consolidation_failure"})
    return controls


def model_fidelity_findings(root: Path) -> list[dict[str, Any]]:
    findings = []
    metrics = root / "analysis" / "tool-funnel" / "20260820-latest-symmetry" / "derived-metrics.json"
    if metrics.exists():
        data = read_json(metrics)
        replay = data.get("replay", {}).get("summary", {})
        failed = replay.get("branches_failed")
        total = replay.get("branches_total")
        findings.append({
            "run_key": "ar25_symmetry_replication",
            "status": "retrospective",
            "finding": f"{failed} of {total} legal replay branches failed" if failed is not None and total is not None else "branch replay found counterfactual failures",
            "evidence_ref": rel(metrics, root),
            "claim_boundary": "The final saved model is tested retrospectively; this cannot date a failure to a historical decision.",
        })
    aar = root / "runs" / "tycho-nooa-gemma4-31b-feasibility-20260821T073930Z" / "AFTER-ACTION.md"
    if aar.exists() and re.search(r"(?is)missing\s+.*actions\(\)|actions\(\).*missing", aar.read_text(encoding="utf-8")):
        findings.append({
            "run_key": "ls20_gemma_feasibility",
            "status": "saved_model_contract_failure",
            "finding": "The saved model omitted required actions(), so replay/planning could not proceed.",
            "evidence_ref": rel(aar, root),
            "claim_boundary": "This proves a final artifact defect, not when that defect first changed actor behavior.",
        })
    return findings


def shadow_candidates(root: Path) -> list[dict[str, Any]]:
    exact_root = root / "runs" / "tycho-nooa-gemma4-31b-reasoning-capture-20260821T081012Z" / "game" / "decision-checkpoints" / "calls"
    call36 = exact_root / "000036" / "request.json"
    call60 = root / "runs" / "tycho-nooa-gemma4-31b-feasibility-20260821T073930Z" / "game" / "decision-checkpoints" / "calls" / "000060" / "request.json"
    candidates = []
    if call36.exists():
        candidates.append({
            "rank": 1,
            "run_key": "ls20_gemma_reasoning",
            "call_index": 36,
            "sequence": 2,
            "causal_state": "candidate/target present; outcome unverified; actor about to inspect a route",
            "one_factor": "Replace rhetorical goal language with one executable provisional-goal obligation: name the predicted terminal signature and choose the cheapest safe discriminator within N actions.",
            "measure": "Whether the first proposed action is an explicitly named discriminator, then action legality and goal promotion/demotion.",
            "discriminates": ["goal_mechanism_failure", "evidence_use_failure"],
            "does_not_settle": ["downstream gameplay benefit", "global builder value", "historical timeout cause"],
            "evidence_ref": rel(call36, root),
            "why": "It holds the exact request/workspace fixed at the boundary the census finds repeatedly weak; unlike report removal, it directly perturbs goal-to-probe control.",
        })
    if call60.exists():
        candidates.append({
            "rank": 2,
            "run_key": "ls20_gemma_feasibility",
            "call_index": 60,
            "sequence": 3,
            "causal_state": "candidate objective open after a costly unchanged builder refresh",
            "one_factor": "Remove or replace the current builder report.",
            "measure": "Immediate action/report dependence.",
            "discriminates": ["handoff_failure", "evidence_use_failure"],
            "does_not_settle": ["objective acquisition", "downstream gameplay benefit"],
            "evidence_ref": rel(call60, root),
            "why": "Useful secondary checkpoint, but report removal is less diagnostic of the missing goal-to-probe transition.",
        })
    return candidates


def mine(root: Path, horizon: int = 3) -> dict[str, Any]:
    controls = positive_controls(horizon)
    if not all(control["pass"] for control in controls):
        raise RuntimeError("epistemic-funnel positive control failed")
    runs = [extract_run(spec, root) for spec in default_corpus(root)]
    episodes = [classify_run(run, horizon) for run in runs]
    supported = [episode for episode in episodes if episode["status"] == "supported"]
    family_counts: dict[str, int] = defaultdict(int)
    for episode in supported:
        family_counts[str(episode["failure_family"])] += 1
    sensitivity = []
    for candidate_horizon in range(1, 7):
        classified = [classify_run(run, candidate_horizon) for run in runs]
        counts: dict[str, int] = defaultdict(int)
        for episode in classified:
            counts[episode["failure_family"] or "censored_or_no_supported_failure"] += 1
        sensitivity.append({"goal_obligation_actions": candidate_horizon, "family_counts": dict(sorted(counts.items()))})
    exact_checkpoint_calls = sum(
        len(list((root / run["path"] / "decision-checkpoints" / "calls").glob("*/request.json")))
        for run in runs
    )
    llm_trace_calls = sum(
        len(read_jsonl(root / run["path"] / "llm-trace.jsonl"))
        for run in runs
    )
    return {
        "schema": "tycho-nooa-epistemic-funnel-v2",
        "schema_version": SCHEMA_VERSION,
        "detector_version": DETECTOR_VERSION,
        "parameters": {
            "goal_obligation_actions": horizon,
            "episode_unit": "first supported failure per run-level trace",
            "null_handling": "censored when the preserved horizon ends before the obligation is overdue",
        },
        "execution": {"model_calls": 0, "arc_environment_calls": 0},
        "positive_controls": controls,
        "runs": runs,
        "failure_episodes": episodes,
        "horizon_sensitivity": sensitivity,
        "aggregate": {
            "runs": len(runs),
            "decision_sequences": sum(len(run["decisions"]) for run in runs),
            "exact_checkpoint_calls": exact_checkpoint_calls,
            "llm_trace_calls": llm_trace_calls,
            "supported_episodes": len(supported),
            "censored_or_no_supported_failure": len(episodes) - len(supported),
            "family_counts": dict(sorted(family_counts.items())),
        },
        "model_fidelity_findings": model_fidelity_findings(root),
        "shadow_candidates": shadow_candidates(root),
        "claim_ceiling": [
            "This is a census of first observable broken control boundaries in five development traces, not a population estimate across games.",
            "Model-authored hypotheses are signals; only tool/host records establish execution and consequence.",
            "Legacy ar25 spend is sequence-aligned by tool-call join; the two Gemma runs use exact decision checkpoints.",
            "Retrospective branch replay establishes saved-model counterexamples but cannot date them to historical decisions.",
        ],
    }


def write_csv(result: dict[str, Any], path: Path) -> None:
    fields = [
        "run_key", "sequence", "recorded_turn", "selected_action", "host_executed", "terminal",
        "stages_new", "stages_seen", "actor_calls_cumulative", "builder_calls_cumulative",
        "scribe_calls_cumulative", "actor_spend_usd_cumulative",
        "builder_spend_usd_cumulative", "scribe_spend_usd_cumulative", "evidence_refs",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in result["runs"]:
            for row in run["decisions"]:
                flat = dict(row)
                for key in ("stages_new", "stages_seen", "evidence_refs"):
                    flat[key] = "|".join(flat[key])
                writer.writerow({key: flat.get(key) for key in fields})


def write_json(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
