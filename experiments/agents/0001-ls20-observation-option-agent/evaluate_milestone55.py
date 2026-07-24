"""Probe for contact/event termination evidence for milestone 5.5.

This evaluator does not upgrade the active option predicate.  It scans public
trace-event records and reports whether any level-3/4 predicate evidence is
selective enough to justify a future M5.5 contact/event upgrade.
"""

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
    EventPredicateEvidenceProbe,
    OptionCandidate,
    OptionExecutor,
    OptionSegmenter,
    PredicateInducer,
    SemanticSignature,
    TraceReference,
    iter_trace_steps,
    summarize_probe,
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
    / "milestone-55-evidence-summary.json"
)
DEFAULT_EVENTS = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-55-evidence-events.jsonl"
)
DEFAULT_CANDIDATES = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "event-predicate-candidates.json"
)
DEFAULT_FREEZE = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-55-evidence-freeze.md"
)


def _load_event_records(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _raw_actions(reference: TraceReference, option: OptionCandidate) -> tuple[int, ...]:
    return reference.segment_actions[option.segment_range[0]:option.segment_range[1]]


def _start_position(option: OptionCandidate) -> tuple[int, int]:
    raw = option.initiation_condition["actor_position"]
    return (int(raw[0]), int(raw[1]))


def _start_signature(option: OptionCandidate) -> SemanticSignature:
    return SemanticSignature.from_labels(
        option.initiation_condition.get("semantic_signature", ())
    )


def _final_signature(option: OptionCandidate) -> SemanticSignature:
    return SemanticSignature.from_labels(
        option.expected_semantic_delta.get("to_signature", ())
    )


def _execute_masked(
    *,
    option: OptionCandidate,
    reference: TraceReference,
) -> bool:
    raw = _raw_actions(reference, option)
    if not raw:
        return False
    start = _start_position(option)
    result = OptionExecutor().execute(
        option=option,
        current_position=start,
        current_signature=_start_signature(option),
        action_deltas=reference.action_model,
        blocked_edges=frozenset({(start, raw[0])}),
        final_signature=_final_signature(option),
    )
    return result.completed


def _m5_selected_option(
    *,
    reference: TraceReference,
    options: tuple[OptionCandidate, ...],
) -> OptionCandidate:
    for option in options:
        if option.predicate_abstraction_level >= 2 and _execute_masked(
            option=option,
            reference=reference,
        ):
            return option
    if not options:
        raise ValueError("M5.5 probe requires at least one option candidate")
    return options[0]


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
    selected = _m5_selected_option(reference=reference, options=options)
    event_records = _load_event_records(trace_events_path)
    probe = EventPredicateEvidenceProbe()
    option_records = []
    all_candidates = []
    for option in options:
        candidates = probe.probe(option=option, event_records=event_records)
        all_candidates.extend(candidates)
        option_records.append({
            "option_id": option.option_id,
            "segment_range": list(option.segment_range),
            "is_m5_selected_option": option.option_id == selected.option_id,
            "selected_predicate_id": option.selected_predicate_id,
            "selected_predicate_expression_hash": option.selected_predicate_expression_hash,
            "probe_summary": summarize_probe(candidates),
            "candidates": [candidate.to_record() for candidate in candidates],
        })

    selected_record = next(
        item for item in option_records if item["option_id"] == selected.option_id
    )
    all_summary = summarize_probe(tuple(all_candidates))
    selected_summary = selected_record["probe_summary"]
    status = (
        "available_event_predicate_evidence"
        if selected_summary["termination_candidate_count"] > 0
        else "available_event_predicate_evidence_elsewhere"
        if all_summary["termination_candidate_count"] > 0
        else "partial_ambiguous_evidence"
        if all_summary["rejected_event_candidate_count"] > 0
        else "no_clean_event_predicate_evidence"
    )
    recommendation = (
        "build_m5_5_contact_event_predicate_upgrade_for_m5_option"
        if status == "available_event_predicate_evidence"
        else "retarget_m5_5_to_option_with_event_evidence_or_continue_to_m6"
        if status == "available_event_predicate_evidence_elsewhere"
        else "skip_m5_5_and_continue_to_m6_self_generated_option_discovery"
    )
    output_events.parent.mkdir(parents=True, exist_ok=True)
    output_candidates.parent.mkdir(parents=True, exist_ok=True)
    with output_events.open("w", encoding="utf-8") as stream:
        for option_record in option_records:
            for candidate in option_record["candidates"]:
                stream.write(json.dumps({
                    "milestone": "M5_5_EVENT_PREDICATE_EVIDENCE_PROBE",
                    "option_id": option_record["option_id"],
                    "is_m5_selected_option": option_record["is_m5_selected_option"],
                    **candidate,
                }, sort_keys=True) + "\n")
    output_candidates.write_text(
        json.dumps(option_records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "milestone": "M5_5_EVENT_PREDICATE_EVIDENCE_PROBE",
        "status": status,
        "trace": str(trace_path),
        "trace_event_artifact": str(trace_events_path),
        "event_artifact": str(output_events),
        "candidate_artifact": str(output_candidates),
        "freeze_artifact": str(output_freeze),
        "option_count": len(options),
        "m5_selected_option_id": selected.option_id,
        "m5_selected_predicate_id": selected.selected_predicate_id,
        "m5_selected_predicate_expression_hash": selected.selected_predicate_expression_hash,
        "m5_selected_option_probe": selected_summary,
        "all_options_probe": all_summary,
        "options_with_available_event_predicates": [
            item["option_id"]
            for item in option_records
            if item["probe_summary"]["termination_candidate_count"] > 0
        ],
        "recommendation": recommendation,
        "known_limits": {
            "probe_only_no_predicate_upgrade": True,
            "uses_public_trace_event_artifact": True,
            "does_not_mutate_environment_or_private_state": True,
            "generic_animation_deltas_not_promoted_to_semantic_termination": True,
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _write_freeze(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "# Milestone 5.5 Evidence Probe Freeze",
            "",
            f"Status: `{summary['status']}`",
            "",
            "Claim: 0001 can audit public trace-event evidence for honest "
            "contact/event termination predicates before attempting a predicate upgrade.",
            "",
            "Key evidence:",
            "",
            f"- M5 selected option: `{summary['m5_selected_option_id']}`",
            "- M5 selected option termination candidates: "
            f"`{summary['m5_selected_option_probe']['termination_candidate_count']}`",
            "- all-option termination candidates: "
            f"`{summary['all_options_probe']['termination_candidate_count']}`",
            f"- recommendation: `{summary['recommendation']}`",
            "",
            "Known limits: this is a probe only; noisy generic animation deltas are "
            "not promoted to semantic termination evidence.",
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
