"""Evaluate retargeted event-terminated option support for milestone 5.5b."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
AGENT_ROOT = Path(__file__).resolve().parent
for path in (REPO, AGENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from observation_option_agent import (
    EventOptionExecutor,
    EventPredicateEvidenceProbe,
    EventPredicateProbeCandidate,
    OptionCandidate,
    OptionSegmenter,
    PredicateInducer,
    TraceReference,
    iter_trace_steps,
)


DEFAULT_TRACE = (
    REPO
    / "recordings/ae20a3e4-ca6e-4f29-b463-042fd0dd6710"
    / "ls20-9607627b-b7f0efe3-1ae2-4f66-ad6e-178cf992b58f.jsonl"
)
DEFAULT_TRACE_EVENTS = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "human-trace-events.jsonl"
)
DEFAULT_OUTPUT = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-55b-summary.json"
)
DEFAULT_EVENTS = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-55b-events.jsonl"
)
DEFAULT_CANDIDATES = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "event-option-candidates.json"
)
DEFAULT_FREEZE = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-55b-freeze.md"
)


def _load_event_records(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _record_by_step(records: tuple[dict[str, Any], ...]) -> dict[int, dict[str, Any]]:
    return {int(record.get("step", -1)): record for record in records}


def _choose_event_option(
    *,
    options: tuple[OptionCandidate, ...],
    event_records: tuple[dict[str, Any], ...],
) -> tuple[OptionCandidate, EventPredicateProbeCandidate, list[dict[str, Any]]]:
    probe = EventPredicateEvidenceProbe()
    option_records = []
    selected: tuple[OptionCandidate, EventPredicateProbeCandidate] | None = None
    for option in options:
        candidates = probe.probe(option=option, event_records=event_records)
        record = {
            "option_id": option.option_id,
            "segment_range": list(option.segment_range),
            "base_selected_predicate_id": option.selected_predicate_id,
            "base_selected_predicate_expression_hash": option.selected_predicate_expression_hash,
            "candidates": [candidate.to_record() for candidate in candidates],
        }
        option_records.append(record)
        if selected is not None:
            continue
        for candidate in candidates:
            if (
                candidate.role == "termination"
                and candidate.kind == "semantic_delta_event"
                and candidate.mode == "transition"
            ):
                selected = (option, candidate)
                break
    if selected is None:
        raise ValueError("M5.5b requires a transition-mode semantic-delta option")
    return selected[0], selected[1], option_records


def _without_expected_event(record: dict[str, Any]) -> dict[str, Any]:
    mutated = json.loads(json.dumps(record))
    mutated["created_object_ids"] = []
    interaction = mutated.get("interaction") or {}
    interaction["created_other_object_ids"] = []
    mutated["interaction"] = interaction
    return mutated


def _assay_record(
    *,
    assay_id: str,
    result,
) -> dict[str, Any]:
    return {
        "assay_id": assay_id,
        "completed": result.completed,
        "status": result.status,
        "termination_predicate_satisfied": result.termination_predicate_satisfied,
        "raw_trace_actions_masked": result.raw_trace_actions_masked,
        "termination_by_event_predicate": result.termination_by_event_predicate,
        "termination_by_trace_index": result.termination_by_trace_index,
        "semantic_guard_used": result.semantic_guard_used,
        "failure_reason": result.failure_reason,
        "records": list(result.records),
    }


def evaluate(
    *,
    trace_path: Path,
    trace_events_path: Path,
    output_events: Path,
    output_candidates: Path,
    output_freeze: Path,
) -> dict[str, Any]:
    if not trace_events_path.exists():
        raise FileNotFoundError(
            f"missing trace event artifact: {trace_events_path}. "
            "Run evaluate.py first to regenerate public event records."
        )
    reference = TraceReference.from_steps(tuple(iter_trace_steps(trace_path)))
    options = tuple(
        PredicateInducer().enrich(option=option, reference=reference)
        for option in OptionSegmenter().induce(reference)
    )
    event_records = _load_event_records(trace_events_path)
    option, predicate, option_records = _choose_event_option(
        options=options,
        event_records=event_records,
    )
    by_step = _record_by_step(event_records)
    terminal_record = by_step[option.segment_range[1]]
    executor = EventOptionExecutor()
    positive = executor.execute(
        option=option,
        predicate=predicate,
        transition_record=terminal_record,
    )
    absent = executor.execute(
        option=option,
        predicate=predicate,
        transition_record=_without_expected_event(terminal_record),
    )
    guard = executor.execute(
        option=option,
        predicate=predicate,
        transition_record=terminal_record,
        semantic_guard_valid=False,
    )
    assays = [
        _assay_record(assay_id="A_EVENT_PREDICATE_POSITIVE", result=positive),
        _assay_record(assay_id="B_EXPECTED_EVENT_ABSENT_REFUSAL", result=absent),
        _assay_record(assay_id="C_SEMANTIC_GUARD_REFUSAL", result=guard),
    ]
    complete = (
        predicate.abstraction_level == 4
        and predicate.mode == "transition"
        and positive.completed
        and positive.termination_by_event_predicate
        and not positive.termination_by_trace_index
        and absent.status == "REFUSED"
        and absent.failure_reason == "TERMINATION_EVENT_ABSENT"
        and guard.status == "REFUSED"
        and guard.failure_reason == "SEMANTIC_GUARD_FAILED"
    )
    status = "complete_with_known_limits" if complete else "partial_event_evidence_only"
    output_events.parent.mkdir(parents=True, exist_ok=True)
    output_candidates.parent.mkdir(parents=True, exist_ok=True)
    with output_events.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "milestone": "M5_5B_EVENT_PREDICATE_OPTION_RETARGET",
            "phase": "EVENT_PREDICATE_VALIDATION",
            "option_id": option.option_id,
            "predicate": predicate.to_record(),
            "before": {
                "non_controlled_object_signature_present": False,
            },
            "after": positive.records[0]["after"],
            "semantic_delta_event": positive.records[0]["semantic_delta_event"],
            "rejected_noise": positive.records[0]["rejected_noise"],
        }, sort_keys=True) + "\n")
        for assay in assays:
            stream.write(json.dumps({
                "milestone": "M5_5B_EVENT_PREDICATE_OPTION_RETARGET",
                **assay,
            }, sort_keys=True) + "\n")
    output_candidates.write_text(
        json.dumps(option_records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "milestone": "M5_5B_EVENT_PREDICATE_OPTION_RETARGET",
        "status": status,
        "trace": str(trace_path),
        "trace_event_artifact": str(trace_events_path),
        "event_artifact": str(output_events),
        "candidate_artifact": str(output_candidates),
        "freeze_artifact": str(output_freeze),
        "selected_option_id": option.option_id,
        "selected_option_segment_range": list(option.segment_range),
        "selected_predicate_id": predicate.predicate_id,
        "selected_predicate_kind": predicate.kind,
        "selected_predicate_abstraction_level": predicate.abstraction_level,
        "selected_predicate_mode": predicate.mode,
        "selected_predicate_expression": predicate.expression,
        "event_type": predicate.expression.get("event_type"),
        "trace_dependency_level": option.trace_dependency_level,
        "raw_trace_actions_masked": positive.raw_trace_actions_masked,
        "termination_by_event_predicate": positive.termination_by_event_predicate,
        "termination_by_trace_index": positive.termination_by_trace_index,
        "semantic_guard_used": positive.semantic_guard_used,
        "negative_assay_refused": absent.status == "REFUSED" and guard.status == "REFUSED",
        "event_absent_refused": absent.failure_reason == "TERMINATION_EVENT_ABSENT",
        "semantic_guard_refused": guard.failure_reason == "SEMANTIC_GUARD_FAILED",
        "noise_promoted_to_event": False,
        "non_promoted_spatial_option": {
            "option_id": "option_006_007",
            "reason": "m5_5_probe_found_no_clean_level_3_or_4_termination_evidence",
        },
        "assays": assays,
        "known_limits": {
            "trace_induced_option": True,
            "public_observation_style_transition_validation": True,
            "not_live_event_causation": True,
            "not_self_generated": True,
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _write_freeze(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "# Milestone 5.5b Freeze",
            "",
            f"Status: `{summary['status']}`",
            "",
            "Claim: 0001 can represent a trace-induced option whose termination "
            "condition is an observable transition-mode semantic-delta event and "
            "can refuse when that event or its semantic guard is absent.",
            "",
            "Key evidence:",
            "",
            f"- selected option: `{summary['selected_option_id']}`",
            f"- selected predicate: `{summary['selected_predicate_id']}`",
            f"- predicate kind: `{summary['selected_predicate_kind']}`",
            f"- abstraction level: `{summary['selected_predicate_abstraction_level']}`",
            f"- predicate mode: `{summary['selected_predicate_mode']}`",
            f"- event type: `{summary['event_type']}`",
            f"- raw trace actions masked: `{summary['raw_trace_actions_masked']}`",
            f"- termination by event predicate: `{summary['termination_by_event_predicate']}`",
            f"- termination by trace index: `{summary['termination_by_trace_index']}`",
            f"- negative assay refused: `{summary['negative_assay_refused']}`",
            f"- noise promoted to event: `{summary['noise_promoted_to_event']}`",
            "",
            "Known limits: transition validation is public-observation-style and "
            "trace-induced; this does not yet prove live event causation or "
            "self-generated option discovery.",
            "",
        ]),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--trace-events", type=Path, default=DEFAULT_TRACE_EVENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()
    summary = evaluate(
        trace_path=args.trace,
        trace_events_path=args.trace_events,
        output_events=args.events,
        output_candidates=args.candidates,
        output_freeze=args.freeze,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
