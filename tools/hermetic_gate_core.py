"""Zero-cost integrity gates for preserved Tycho-NOOA protocol evidence.

This module deliberately audits only public host messages and solver-owned
evidence.  It neither imports the ARC engine nor calls a language model.  The
gate is therefore allowed to reject a candidate cheaply; passing it only means
that the tested orchestration/evidence contracts survived.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "tycho-nooa-hermetic-gate-v1"
STREAM_FILES = {
    "transcript": "transcript.jsonl",
    "llm": "llm-trace.jsonl",
    "actor_tools": "tool-trace.jsonl",
    "builder_tools": "builder-tool-trace.jsonl",
    "triggers": "trigger-builder-trace.jsonl",
    "planner": "planner-trace.jsonl",
    "builder_planner": "builder-planner-trace.jsonl",
}


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    stream: str
    row: int | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceBundle:
    transcript: list[dict[str, Any]] = field(default_factory=list)
    llm: list[dict[str, Any]] = field(default_factory=list)
    actor_tools: list[dict[str, Any]] = field(default_factory=list)
    builder_tools: list[dict[str, Any]] = field(default_factory=list)
    triggers: list[dict[str, Any]] = field(default_factory=list)
    planner: list[dict[str, Any]] = field(default_factory=list)
    builder_planner: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    load_violations: list[Violation] = field(default_factory=list)

    def clone(self) -> TraceBundle:
        return copy.deepcopy(self)


def audit_run(run_dir: Path) -> dict[str, Any]:
    """Audit one preserved run and return a stable machine-readable report."""

    run_dir = run_dir.resolve()
    bundle = load_bundle(run_dir)
    violations = audit_bundle(bundle)
    violations.extend(audit_checkpoints(run_dir / "decision-checkpoints"))
    return report(
        violations,
        source=str(run_dir),
        metrics={
            "transcript_rows": len(bundle.transcript),
            "llm_calls": len(bundle.llm),
            "actor_tool_rows": len(bundle.actor_tools),
            "builder_tool_rows": len(bundle.builder_tools),
            "trigger_rows": len(bundle.triggers),
            "checkpoint_calls": _checkpoint_call_count(
                run_dir / "decision-checkpoints"
            ),
        },
    )


def load_bundle(run_dir: Path) -> TraceBundle:
    bundle = TraceBundle()
    for stream, filename in STREAM_FILES.items():
        rows, violations = _load_jsonl(run_dir / filename, stream)
        setattr(bundle, stream, rows)
        bundle.load_violations.extend(violations)
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        try:
            value = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise TypeError("summary root is not an object")
            bundle.summary = value
        except (OSError, json.JSONDecodeError, TypeError) as error:
            bundle.load_violations.append(
                Violation("malformed_json", str(error), "summary")
            )
    return bundle


def audit_bundle(bundle: TraceBundle) -> list[Violation]:
    violations = list(bundle.load_violations)
    violations.extend(_audit_transcript(bundle))
    violations.extend(_audit_tool_joins(bundle))
    violations.extend(_audit_action_traceability(bundle))
    violations.extend(_audit_builder_lifecycle(bundle))
    violations.extend(_audit_planner_stream(bundle.planner, "planner"))
    violations.extend(
        _audit_planner_stream(bundle.builder_planner, "builder_planner")
    )
    return _deduplicate(violations)


def report(
    violations: Iterable[Violation],
    *,
    source: str,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = list(violations)
    return {
        "schema": SCHEMA,
        "source": source,
        "status": "pass" if not rows else "fail",
        "promotion_allowed": not rows,
        "claim_ceiling": (
            "tested orchestration and evidence-integrity contracts only; not "
            "LLM understanding, gameplay benefit, or transfer"
        ),
        "metrics": metrics or {},
        "violation_count": len(rows),
        "violations": [asdict(value) for value in rows],
    }


def _audit_transcript(bundle: TraceBundle) -> list[Violation]:
    rows = bundle.transcript
    violations: list[Violation] = []
    if not rows:
        return [
            Violation(
                "missing_transcript",
                "no public host transcript was preserved",
                "transcript",
            )
        ]
    starts = [(index, row) for index, row in enumerate(rows) if row.get("type") == "start_game"]
    if len(starts) != 1:
        violations.append(
            Violation(
                "worker_restart_mid_run" if len(starts) > 1 else "missing_start_game",
                f"expected one start_game row, found {len(starts)}",
                "transcript",
                starts[1][0] if len(starts) > 1 else None,
            )
        )

    turns = _by_sequence(rows, "turn")
    decisions = _by_sequence(rows, "decision")
    results = _by_sequence(rows, "action_result")
    sequences = sorted(set(turns) | set(decisions) | set(results))
    if sequences and sequences != list(range(sequences[-1] + 1)):
        violations.append(
            Violation(
                "non_contiguous_sequence",
                f"host sequences are {sequences!r}",
                "transcript",
            )
        )

    previous_result: dict[str, Any] | None = None
    completed = -1
    for sequence in sequences:
        turn_rows = turns.get(sequence, [])
        decision_rows = decisions.get(sequence, [])
        result_rows = results.get(sequence, [])
        if len(turn_rows) != 1:
            violations.append(
                Violation(
                    "turn_count",
                    f"sequence {sequence} has {len(turn_rows)} turn rows",
                    "transcript",
                    turn_rows[0][0] if turn_rows else None,
                    {"sequence": sequence},
                )
            )
        if len(decision_rows) != 1:
            violations.append(
                Violation(
                    "decision_count",
                    f"sequence {sequence} has {len(decision_rows)} decisions",
                    "transcript",
                    decision_rows[0][0] if decision_rows else None,
                    {"sequence": sequence},
                )
            )
        if len(result_rows) != 1:
            violations.append(
                Violation(
                    "action_result_count",
                    f"sequence {sequence} has {len(result_rows)} action results",
                    "transcript",
                    result_rows[0][0] if result_rows else None,
                    {"sequence": sequence},
                )
            )
        if not (len(turn_rows) == len(decision_rows) == len(result_rows) == 1):
            continue

        turn_index, turn = turn_rows[0]
        decision_index, decision_row = decision_rows[0]
        result_index, result = result_rows[0]
        if not turn_index < decision_index < result_index:
            violations.append(
                Violation(
                    "turn_triplet_out_of_order",
                    (
                        f"sequence {sequence} row order is "
                        f"turn={turn_index}, decision={decision_index}, "
                        f"result={result_index}"
                    ),
                    "transcript",
                    min(turn_index, decision_index, result_index),
                    {"sequence": sequence},
                )
            )
        decision = decision_row.get("decision")
        observation = turn.get("observation")
        if not isinstance(decision, dict) or not isinstance(observation, dict):
            violations.append(
                Violation(
                    "malformed_turn_triplet",
                    f"sequence {sequence} lacks decision or observation objects",
                    "transcript",
                    turn_index,
                )
            )
            continue
        action = decision.get("action")
        available = observation.get("available_actions", [])
        if action not in available:
            violations.append(
                Violation(
                    "illegal_action",
                    f"sequence {sequence} committed {action!r}, available={available!r}",
                    "transcript",
                    decision_index,
                    {"sequence": sequence, "action": action},
                )
            )
        violations.extend(_coordinate_violations(decision, decision_index, sequence))
        if result.get("decision") != decision:
            violations.append(
                Violation(
                    "decision_result_mismatch",
                    f"sequence {sequence} result does not acknowledge its decision",
                    "transcript",
                    result_index,
                    {"sequence": sequence},
                )
            )
        if result.get("before") != observation:
            violations.append(
                Violation(
                    "before_observation_mismatch",
                    f"sequence {sequence} action_result.before differs from turn",
                    "transcript",
                    result_index,
                    {"sequence": sequence},
                )
            )
        if previous_result is not None:
            if turn.get("previous_decision") != previous_result.get("decision"):
                violations.append(
                    Violation(
                        "previous_decision_mismatch",
                        f"sequence {sequence} does not acknowledge the prior action",
                        "transcript",
                        turn_index,
                        {"sequence": sequence},
                    )
                )
            if observation != previous_result.get("after"):
                violations.append(
                    Violation(
                        "observation_chain_broken",
                        f"sequence {sequence} does not begin at the prior successor",
                        "transcript",
                        turn_index,
                        {"sequence": sequence},
                    )
                )
        for candidate in (observation, result.get("after")):
            if isinstance(candidate, dict):
                value = candidate.get("levels_completed")
                if isinstance(value, int):
                    if value < completed:
                        violations.append(
                            Violation(
                                "evidence_regressed",
                                f"levels_completed fell from {completed} to {value}",
                                "transcript",
                                result_index,
                                {"sequence": sequence},
                            )
                        )
                    completed = max(completed, value)
        previous_result = result

    finishes = [row for row in rows if row.get("type") == "finish"]
    if len(finishes) != 1:
        violations.append(
            Violation(
                "finish_count",
                f"expected one finish row, found {len(finishes)}",
                "transcript",
            )
        )
    if finishes:
        finish = finishes[-1]
        if finish.get("actions") != len(results):
            violations.append(
                Violation(
                    "finish_action_count_mismatch",
                    f"finish actions={finish.get('actions')!r}, results={len(results)}",
                    "transcript",
                )
            )
        if previous_result is not None and finish.get("final_observation") != previous_result.get("after"):
            violations.append(
                Violation(
                    "finish_observation_mismatch",
                    "finish does not preserve the last public successor",
                    "transcript",
                )
            )
    if bundle.summary:
        if bundle.summary.get("actions") != len(results):
            violations.append(
                Violation(
                    "summary_action_count_mismatch",
                    f"summary actions={bundle.summary.get('actions')!r}, results={len(results)}",
                    "summary",
                )
            )
        if finishes and bundle.summary.get("levels_completed") != finishes[-1].get("levels_completed"):
            violations.append(
                Violation(
                    "summary_level_mismatch",
                    "summary and finish disagree on levels_completed",
                    "summary",
                )
            )
    return violations


def _coordinate_violations(
    decision: dict[str, Any], row: int, sequence: int
) -> list[Violation]:
    action = decision.get("action")
    has_coordinates = decision.get("row") is not None or decision.get("col") is not None
    invalid = (action == "CLICK" and not (
        isinstance(decision.get("row"), int) and isinstance(decision.get("col"), int)
    )) or (action != "CLICK" and has_coordinates)
    if not invalid:
        return []
    return [
        Violation(
            "invalid_action_coordinates",
            f"sequence {sequence} coordinates do not match action {action!r}",
            "transcript",
            row,
            {"sequence": sequence},
        )
    ]


def _audit_tool_joins(bundle: TraceBundle) -> list[Violation]:
    violations: list[Violation] = []
    requested: dict[str, list[tuple[str, str, int]]] = {}
    for row_index, row in enumerate(bundle.llm):
        source = row.get("source")
        if source not in {"actor", "builder"}:
            violations.append(
                Violation(
                    "invalid_llm_source",
                    f"LLM trace source is {source!r}",
                    "llm",
                    row_index,
                )
            )
            continue
        lane = source
        ids = row.get("tool_call_ids", [])
        names = row.get("tool_calls", [])
        if len(ids) != len(names):
            violations.append(
                Violation(
                    "llm_tool_vector_mismatch",
                    "tool_call_ids and tool_calls have different lengths",
                    "llm",
                    row_index,
                )
            )
        for call_id, name in zip(ids, names, strict=False):
            if not isinstance(call_id, str) or not call_id:
                violations.append(
                    Violation(
                        "missing_tool_call_id",
                        "new trace row lacks a usable tool_call_id",
                        "llm",
                        row_index,
                    )
                )
                continue
            requested.setdefault(call_id, []).append((lane, str(name), row_index))
    for call_id, entries in requested.items():
        if len(entries) > 1:
            violations.append(
                Violation(
                    "duplicate_tool_call_id",
                    f"tool_call_id {call_id!r} was requested {len(entries)} times",
                    "llm",
                    entries[1][2],
                    {"tool_call_id": call_id},
                )
            )

    observed: dict[str, list[tuple[str, str, int]]] = {}
    for lane, stream, rows in (
        ("actor", "actor_tools", bundle.actor_tools),
        ("builder", "builder_tools", bundle.builder_tools),
    ):
        for row_index, row in enumerate(rows):
            call_id = row.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                violations.append(
                    Violation(
                        "missing_tool_call_id",
                        "tool result lacks a usable tool_call_id",
                        stream,
                        row_index,
                    )
                )
                continue
            observed.setdefault(call_id, []).append((lane, str(row.get("name")), row_index))
            candidates = requested.get(call_id, [])
            if not candidates:
                violations.append(
                    Violation(
                        "orphan_tool_result",
                        f"tool result {call_id!r} has no LLM request",
                        stream,
                        row_index,
                        {"tool_call_id": call_id},
                    )
                )
                continue
            request_lane, request_name, _ = candidates[0]
            if request_lane != lane:
                violations.append(
                    Violation(
                        "cross_lane_tool_result",
                        f"{call_id!r} requested by {request_lane}, recorded by {lane}",
                        stream,
                        row_index,
                        {"tool_call_id": call_id},
                    )
                )
            if request_name != row.get("name"):
                violations.append(
                    Violation(
                        "tool_name_mismatch",
                        f"{call_id!r} requested {request_name!r}, recorded {row.get('name')!r}",
                        stream,
                        row_index,
                        {"tool_call_id": call_id},
                    )
                )
            if row.get("status") not in {"complete", "error"}:
                violations.append(
                    Violation(
                        "invalid_tool_status",
                        f"{call_id!r} has status {row.get('status')!r}",
                        stream,
                        row_index,
                    )
                )
    for call_id, entries in observed.items():
        if len(entries) > 1:
            violations.append(
                Violation(
                    "duplicate_tool_result",
                    f"tool_call_id {call_id!r} has {len(entries)} result rows",
                    "tool_traces",
                    evidence={"tool_call_id": call_id},
                )
            )
    for call_id, entries in requested.items():
        if call_id not in observed:
            violations.append(
                Violation(
                    "missing_tool_result",
                    f"tool request {call_id!r} has no result row",
                    "tool_traces",
                    evidence={"tool_call_id": call_id, "lane": entries[0][0]},
                )
            )
    return violations


def _audit_action_traceability(bundle: TraceBundle) -> list[Violation]:
    violations: list[Violation] = []
    turns = _by_sequence(bundle.transcript, "turn")
    decisions = _by_sequence(bundle.transcript, "decision")
    commits: dict[tuple[int, int], list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(bundle.actor_tools):
        if row.get("name") == "take_action" and row.get("status") == "complete":
            key = (row.get("level"), row.get("turn_in_level"))
            commits.setdefault(key, []).append((index, row))
    for sequence, decision_rows in decisions.items():
        turn_rows = turns.get(sequence, [])
        if len(turn_rows) != 1 or len(decision_rows) != 1:
            continue
        turn = turn_rows[0][1]
        key = (turn.get("level"), turn.get("turn_in_level"))
        candidates = commits.get(key, [])
        if len(candidates) != 1:
            violations.append(
                Violation(
                    "multiple_action_commits" if len(candidates) > 1 else "action_not_traceable",
                    f"sequence {sequence} has {len(candidates)} completed take_action rows",
                    "actor_tools",
                    candidates[0][0] if candidates else None,
                    {"sequence": sequence},
                )
            )
            continue
        tool_result = candidates[0][1].get("result")
        if tool_result != decision_rows[0][1].get("decision"):
            violations.append(
                Violation(
                    "tool_decision_mismatch",
                    f"sequence {sequence} host decision differs from completed tool",
                    "actor_tools",
                    candidates[0][0],
                    {"sequence": sequence},
                )
            )
    return violations


def _audit_builder_lifecycle(bundle: TraceBundle) -> list[Violation]:
    violations: list[Violation] = []
    fired: set[tuple[int, int]] = set()
    results: set[tuple[int, int]] = set()
    errors: set[tuple[int, int]] = set()
    turns_by_location = {
        (row.get("level"), row.get("turn_in_level")): row.get("observation", {})
        for row in bundle.transcript
        if row.get("type") == "turn"
    }
    for index, row in enumerate(bundle.triggers):
        key = (row.get("level"), row.get("turn_in_level"))
        event = row.get("event")
        if (
            event == "mechanical_trigger" and row.get("fired") is True
        ) or event == "actor_request":
            fired.add(key)
        elif event in {
            "builder_result",
            "builder_error",
            "builder_deadline_exceeded",
            "builder_deadline_skipped",
        }:
            if key not in fired:
                violations.append(
                    Violation(
                        "builder_without_trigger",
                        f"{event} at {key!r} has no prior fired trigger",
                        "triggers",
                        index,
                    )
                )
            (results if event == "builder_result" else errors).add(key)
            if event == "builder_result":
                recommendation = row.get("recommended_action")
                available = turns_by_location.get(key, {}).get("available_actions", [])
                if (
                    isinstance(recommendation, dict)
                    and recommendation.get("action") not in available
                    and row.get("recommendation_parse_error") is None
                ):
                    violations.append(
                        Violation(
                            "invalid_builder_recommendation",
                            f"builder exposed unavailable action {recommendation.get('action')!r}",
                            "triggers",
                            index,
                        )
                    )
        elif event == "actor_uptake" and key not in results:
            violations.append(
                Violation(
                    "uptake_without_builder_result",
                    f"actor uptake at {key!r} has no successful builder result",
                    "triggers",
                    index,
                )
            )

    for key in fired - results - errors:
        violations.append(
            Violation(
                "trigger_without_builder_terminal",
                f"fired trigger at {key!r} has neither result nor recorded error",
                "triggers",
            )
        )

    completions: list[tuple[int, int]] = []
    for row in bundle.transcript:
        if row.get("type") != "action_result":
            continue
        before, after = row.get("before", {}), row.get("after", {})
        if after.get("levels_completed", 0) > before.get("levels_completed", 0):
            completions.append((row.get("level"), row.get("turn_in_level")))
    for level, turn in completions:
        candidate_keys = {(level, turn), (level, turn + 1)}
        checkpoints = [
            row
            for row in bundle.triggers
            if row.get("event") == "mechanical_trigger"
            and row.get("fired") is True
            and row.get("cause") == "level_complete"
            and (row.get("level"), row.get("turn_in_level")) in candidate_keys
        ]
        if not checkpoints:
            violations.append(
                Violation(
                    "missing_outcome_checkpoint",
                    f"host-confirmed completion at level {level} turn {turn} was not checkpointed",
                    "triggers",
                    evidence={"level": level, "turn_in_level": turn},
                )
            )
            continue
        checkpoint_key = (
            checkpoints[0].get("level"),
            checkpoints[0].get("turn_in_level"),
        )
        terminal = [
            row
            for row in bundle.triggers
            if row.get("event") in {"builder_result", "builder_error"}
            and (row.get("level"), row.get("turn_in_level")) == checkpoint_key
        ]
        if not terminal:
            continue
        result_rows = [row for row in terminal if row.get("event") == "builder_result"]
        if result_rows and not result_rows[-1].get("outcome_checkpoint_required"):
            violations.append(
                Violation(
                    "outcome_contract_not_declared",
                    "completion builder result did not declare the outcome checkpoint",
                    "triggers",
                    evidence={"level": level, "turn_in_level": turn},
                )
            )
        if result_rows and not result_rows[-1].get("outcome_checkpoint_resolved"):
            later_resolved = any(
                row.get("event") == "builder_result"
                and row.get("outcome_checkpoint_required") is True
                and row.get("outcome_checkpoint_resolved") is True
                for row in bundle.triggers[bundle.triggers.index(result_rows[-1]) + 1 :]
            )
            if not later_resolved:
                violations.append(
                    Violation(
                        "outcome_repair_unresolved",
                        "declared outcome checkpoint was not resolved by a later retry",
                        "triggers",
                        evidence={"level": level, "turn_in_level": turn},
                    )
                )
    return violations


def _audit_planner_stream(
    rows: list[dict[str, Any]], stream: str
) -> list[Violation]:
    if not rows:
        return []
    indices = [row.get("event_index") for row in rows]
    if indices == list(range(1, len(rows) + 1)):
        return []
    return [
        Violation(
            "planner_trace_not_monotone",
            f"{stream} event indices are {indices!r}",
            stream,
        )
    ]


def audit_checkpoints(root: Path) -> list[Violation]:
    """Verify content-addressed request/workspace evidence when it exists."""

    if not root.exists():
        return []
    violations: list[Violation] = []
    calls = sorted((root / "calls").glob("[0-9]*"))
    expected = [f"{index:06d}" for index in range(1, len(calls) + 1)]
    if [path.name for path in calls] != expected:
        violations.append(
            Violation(
                "checkpoint_sequence_not_contiguous",
                "decision checkpoint call directories are not contiguous",
                "checkpoints",
            )
        )
    for call_dir in calls:
        documents: dict[str, dict[str, Any]] = {}
        for filename in ("request.json", "response.json", "workspace.json"):
            path = call_dir / filename
            if not path.is_file():
                violations.append(
                    Violation(
                        "checkpoint_file_missing",
                        f"{call_dir.name}/{filename} is missing",
                        "checkpoints",
                        evidence={"call": call_dir.name, "file": filename},
                    )
                )
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise TypeError(f"{filename} root is not an object")
                documents[filename] = value
            except (OSError, json.JSONDecodeError, TypeError) as error:
                violations.append(
                    Violation(
                        "malformed_checkpoint",
                        str(error),
                        "checkpoints",
                        evidence={"call": call_dir.name, "file": filename},
                    )
                )

        workspace = documents.get("workspace.json")
        request = documents.get("request.json")
        response = documents.get("response.json")
        if request is not None:
            expected_index = int(call_dir.name) if call_dir.name.isdigit() else None
            if expected_index is None or request.get("call_index") != expected_index:
                violations.append(
                    Violation(
                        "checkpoint_call_index_mismatch",
                        f"checkpoint {call_dir.name} declares {request.get('call_index')!r}",
                        "checkpoints",
                        evidence={"call": call_dir.name},
                    )
                )
            if request.get("source") not in {"actor", "builder"}:
                violations.append(
                    Violation(
                        "checkpoint_source_invalid",
                        f"checkpoint {call_dir.name} source is {request.get('source')!r}",
                        "checkpoints",
                        evidence={"call": call_dir.name},
                    )
                )
        if response is not None and response.get("status") not in {"complete", "error"}:
            violations.append(
                Violation(
                    "checkpoint_response_status_invalid",
                    f"checkpoint {call_dir.name} response status is {response.get('status')!r}",
                    "checkpoints",
                    evidence={"call": call_dir.name},
                )
            )
        if workspace is None or request is None:
            continue
        if request.get("workspace_commitment") != _commitment(workspace):
            violations.append(
                Violation(
                    "workspace_commitment_mismatch",
                    f"checkpoint {call_dir.name} workspace commitment is stale",
                    "checkpoints",
                    evidence={"call": call_dir.name},
                )
            )
        body = request.get("request_body")
        if request.get("request_payload_commitment") != _commitment(body):
            violations.append(
                Violation(
                    "request_commitment_mismatch",
                    f"checkpoint {call_dir.name} request commitment is stale",
                    "checkpoints",
                    evidence={"call": call_dir.name},
                )
            )
        files = workspace.get("files")
        if not isinstance(files, list):
            violations.append(
                Violation(
                    "malformed_checkpoint",
                    f"checkpoint {call_dir.name} workspace files is not a list",
                    "checkpoints",
                    evidence={"call": call_dir.name, "file": "workspace.json"},
                )
            )
            continue
        for entry in files:
            if not isinstance(entry, dict):
                violations.append(
                    Violation(
                        "malformed_checkpoint",
                        f"checkpoint {call_dir.name} has a non-object file entry",
                        "checkpoints",
                        evidence={"call": call_dir.name, "file": "workspace.json"},
                    )
                )
                continue
            relative = entry.get("path")
            if not _safe_manifest_path(relative):
                violations.append(
                    Violation(
                        "unsafe_checkpoint_path",
                        f"checkpoint {call_dir.name} has unsafe path {relative!r}",
                        "checkpoints",
                        evidence={"call": call_dir.name, "path": relative},
                    )
                )
            digest = entry.get("sha256")
            blob = root / "blobs" / "sha256" / str(digest)
            if not blob.is_file():
                violations.append(
                    Violation(
                        "checkpoint_blob_missing",
                        f"checkpoint {call_dir.name} lacks blob {digest}",
                        "checkpoints",
                        evidence={"call": call_dir.name, "sha256": digest},
                    )
                )
                continue
            content = blob.read_bytes()
            if len(content) != entry.get("size") or hashlib.sha256(content).hexdigest() != digest:
                violations.append(
                    Violation(
                        "checkpoint_blob_corrupt",
                        f"checkpoint {call_dir.name} blob {digest} is corrupt",
                        "checkpoints",
                        evidence={"call": call_dir.name, "sha256": digest},
                    )
                )
    return violations


def run_fault_campaign() -> dict[str, Any]:
    """Generate discriminating corruptions and reduce each counterexample."""

    baseline = synthetic_bundle()
    baseline_codes = violation_codes(audit_bundle(baseline))
    cases = []
    for name, expected, mutate in _fault_mutations():
        candidate = baseline.clone()
        mutate(candidate)
        found = violation_codes(audit_bundle(candidate))
        reduced = shrink_bundle(candidate, expected)
        cases.append(
            {
                "name": name,
                "expected_violation": expected,
                "detected": expected in found,
                "all_violations": sorted(found),
                "reduced_rows": _bundle_size(reduced),
                "reduced_evidence": _compact_bundle(reduced),
            }
        )
    round_tripped = TraceBundle(**json.loads(json.dumps(asdict(baseline))))
    renamed = rename_tool_call_ids(baseline, prefix="metamorphic")
    relations = {
        "json_round_trip_preserves_result": violation_codes(audit_bundle(round_tripped))
        == baseline_codes,
        "tool_id_bijection_preserves_result": violation_codes(audit_bundle(renamed))
        == baseline_codes,
    }
    passed = not baseline_codes and all(case["detected"] for case in cases) and all(relations.values())
    return {
        "schema": SCHEMA,
        "status": "pass" if passed else "fail",
        "promotion_allowed": passed,
        "baseline_violations": sorted(baseline_codes),
        "generated_cases": len(cases),
        "cases": cases,
        "metamorphic_relations": relations,
        "claim_ceiling": "the gate detects its declared synthetic fault families",
    }


def shrink_bundle(bundle: TraceBundle, code: str) -> TraceBundle:
    """Greedily delete evidence rows while preserving one violation code."""

    candidate = bundle.clone()
    for stream in STREAM_FILES:
        rows = getattr(candidate, stream)
        index = 0
        while index < len(rows):
            trial = candidate.clone()
            del getattr(trial, stream)[index]
            if code in violation_codes(audit_bundle(trial)):
                candidate = trial
                rows = getattr(candidate, stream)
            else:
                index += 1
    return candidate


def rename_tool_call_ids(bundle: TraceBundle, *, prefix: str) -> TraceBundle:
    candidate = bundle.clone()
    ids = {
        call_id
        for row in candidate.llm
        for call_id in row.get("tool_call_ids", [])
        if isinstance(call_id, str)
    }
    mapping = {call_id: f"{prefix}-{index}" for index, call_id in enumerate(sorted(ids))}
    for row in candidate.llm:
        row["tool_call_ids"] = [mapping.get(value, value) for value in row.get("tool_call_ids", [])]
    for stream in ("actor_tools", "builder_tools"):
        for row in getattr(candidate, stream):
            row["tool_call_id"] = mapping.get(row.get("tool_call_id"), row.get("tool_call_id"))
    return candidate


def synthetic_bundle() -> TraceBundle:
    """Small valid two-action run used as the generated campaign's seed."""

    obs0 = _observation(0, levels=0)
    obs1 = _observation(1, levels=0)
    obs2 = _observation(2, levels=1)
    up = {"action": "UP", "row": None, "col": None}
    down = {"action": "DOWN", "row": None, "col": None}
    transcript = [
        {"type": "start_game", "game_id": "synthetic", "seed": 0, "initial_observation": obs0},
        {"type": "ready"},
        {"type": "turn", "sequence": 0, "level": 0, "turn_in_level": 0, "observation": obs0, "previous_decision": None},
        {"type": "decision", "sequence": 0, "decision": up},
        {"type": "action_result", "sequence": 0, "level": 0, "turn_in_level": 0, "decision": up, "before": obs0, "after": obs1},
        {"type": "turn", "sequence": 1, "level": 0, "turn_in_level": 1, "observation": obs1, "previous_decision": up},
        {"type": "decision", "sequence": 1, "decision": down},
        {"type": "action_result", "sequence": 1, "level": 0, "turn_in_level": 1, "decision": down, "before": obs1, "after": obs2},
        {"type": "finish", "actions": 2, "levels_completed": 1, "final_observation": obs2},
    ]
    llm = [
        {"call_index": 1, "source": "actor", "tool_calls": ["take_action"], "tool_call_ids": ["act-0"]},
        {"call_index": 2, "source": "actor", "tool_calls": ["take_action"], "tool_call_ids": ["act-1"]},
    ]
    actor_tools = [
        {"call_index": 1, "tool_call_id": "act-0", "level": 0, "turn_in_level": 0, "name": "take_action", "arguments": {"action": "UP"}, "status": "complete", "result": up},
        {"call_index": 2, "tool_call_id": "act-1", "level": 0, "turn_in_level": 1, "name": "take_action", "arguments": {"action": "DOWN"}, "status": "complete", "result": down},
    ]
    triggers = [
        {"event": "mechanical_trigger", "level": 0, "turn_in_level": 2, "fired": True, "cause": "level_complete"},
        {"event": "builder_result", "level": 0, "turn_in_level": 2, "outcome_checkpoint_required": True, "outcome_checkpoint_resolved": True, "recommended_action": None, "recommendation_parse_error": None},
    ]
    return TraceBundle(
        transcript=transcript,
        llm=llm,
        actor_tools=actor_tools,
        triggers=triggers,
        summary={"actions": 2, "levels_completed": 1},
    )


def _fault_mutations() -> list[tuple[str, str, Callable[[TraceBundle], None]]]:
    return [
        ("worker_restart", "worker_restart_mid_run", _mutate_worker_restart),
        ("duplicate_decision", "decision_count", _mutate_duplicate_decision),
        ("missing_action_result", "action_result_count", _mutate_missing_result),
        ("out_of_order_triplet", "turn_triplet_out_of_order", _mutate_triplet_order),
        ("illegal_action", "illegal_action", _mutate_illegal_action),
        ("evidence_regression", "evidence_regressed", _mutate_evidence_regression),
        ("duplicate_tool_id", "duplicate_tool_call_id", _mutate_duplicate_tool_id),
        ("missing_tool_result", "missing_tool_result", _mutate_missing_tool_result),
        ("cross_lane_result", "cross_lane_tool_result", _mutate_cross_lane),
        ("failed_action_as_execution", "action_not_traceable", _mutate_failed_action),
        ("malformed_builder_advice", "invalid_builder_recommendation", _mutate_builder_advice),
        ("missing_outcome_repair", "missing_outcome_checkpoint", _mutate_outcome_repair),
    ]


def _mutate_worker_restart(bundle: TraceBundle) -> None:
    bundle.transcript.insert(5, copy.deepcopy(bundle.transcript[0]))


def _mutate_duplicate_decision(bundle: TraceBundle) -> None:
    bundle.transcript.insert(4, copy.deepcopy(bundle.transcript[3]))


def _mutate_missing_result(bundle: TraceBundle) -> None:
    bundle.transcript = [
        row
        for row in bundle.transcript
        if not (row.get("type") == "action_result" and row.get("sequence") == 0)
    ]


def _mutate_triplet_order(bundle: TraceBundle) -> None:
    decision = bundle.transcript.pop(3)
    bundle.transcript.insert(2, decision)


def _mutate_illegal_action(bundle: TraceBundle) -> None:
    illegal = {"action": "TELEPORT", "row": None, "col": None}
    for row in bundle.transcript:
        if row.get("sequence") == 0 and row.get("type") in {"decision", "action_result"}:
            row["decision"] = illegal
        if row.get("sequence") == 1 and row.get("type") == "turn":
            row["previous_decision"] = illegal
    bundle.actor_tools[0]["result"] = illegal
    bundle.actor_tools[0]["arguments"] = {"action": "TELEPORT"}


def _mutate_evidence_regression(bundle: TraceBundle) -> None:
    result = next(row for row in bundle.transcript if row.get("type") == "action_result" and row.get("sequence") == 1)
    result["before"]["levels_completed"] = 2


def _mutate_duplicate_tool_id(bundle: TraceBundle) -> None:
    bundle.llm[1]["tool_call_ids"] = ["act-0"]
    bundle.actor_tools[1]["tool_call_id"] = "act-0"


def _mutate_missing_tool_result(bundle: TraceBundle) -> None:
    del bundle.actor_tools[0]


def _mutate_cross_lane(bundle: TraceBundle) -> None:
    bundle.llm[0]["source"] = "builder"


def _mutate_failed_action(bundle: TraceBundle) -> None:
    bundle.actor_tools[0]["status"] = "error"


def _mutate_builder_advice(bundle: TraceBundle) -> None:
    bundle.triggers[1]["recommended_action"] = {"action": "TELEPORT", "row": None, "col": None}


def _mutate_outcome_repair(bundle: TraceBundle) -> None:
    bundle.triggers.clear()


def violation_codes(violations: Iterable[Violation]) -> set[str]:
    return {value.code for value in violations}


def _observation(value: int, *, levels: int) -> dict[str, Any]:
    return {
        "grid": [[value]],
        "available_actions": ["UP", "DOWN"],
        "levels_completed": levels,
        "state": "NOT_FINISHED",
        "full_reset": False,
    }


def _by_sequence(
    rows: list[dict[str, Any]], message_type: str
) -> dict[int, list[tuple[int, dict[str, Any]]]]:
    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        if row.get("type") != message_type or not isinstance(row.get("sequence"), int):
            continue
        grouped.setdefault(row["sequence"], []).append((index, row))
    return grouped


def _load_jsonl(path: Path, stream: str) -> tuple[list[dict[str, Any]], list[Violation]]:
    if not path.is_file():
        return [], []
    rows: list[dict[str, Any]] = []
    violations: list[Violation] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("JSONL row is not an object")
            rows.append(value)
        except (json.JSONDecodeError, TypeError) as error:
            violations.append(Violation("malformed_json", str(error), stream, index))
    return rows, violations


def _deduplicate(values: list[Violation]) -> list[Violation]:
    unique: list[Violation] = []
    seen: set[tuple[Any, ...]] = set()
    for value in values:
        key = (value.code, value.stream, value.row, json.dumps(value.evidence, sort_keys=True))
        if key not in seen:
            unique.append(value)
            seen.add(key)
    return unique


def _commitment(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _safe_manifest_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and "." not in path.parts


def _checkpoint_call_count(root: Path) -> int:
    return len(list((root / "calls").glob("[0-9]*"))) if root.exists() else 0


def _bundle_size(bundle: TraceBundle) -> int:
    return sum(len(getattr(bundle, stream)) for stream in STREAM_FILES)


def _compact_bundle(bundle: TraceBundle) -> dict[str, list[dict[str, Any]]]:
    compact: dict[str, list[dict[str, Any]]] = {}
    for stream in STREAM_FILES:
        rows = getattr(bundle, stream)
        if rows:
            compact[stream] = [
                {
                    key: row.get(key)
                    for key in (
                        "type",
                        "event",
                        "sequence",
                        "level",
                        "turn_in_level",
                        "source",
                        "name",
                        "status",
                        "tool_call_id",
                        "tool_call_ids",
                    )
                    if key in row
                }
                for row in rows
            ]
    return compact
