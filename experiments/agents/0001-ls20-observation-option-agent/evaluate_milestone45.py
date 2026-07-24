"""Evaluate predicate abstraction for milestone 4.5."""

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
    OptionCandidate,
    OptionExecutor,
    OptionSegmenter,
    PredicateInducer,
    SemanticSignature,
    TraceReference,
    iter_trace_steps,
)


DEFAULT_TRACE = (
    REPO
    / "recordings/ae20a3e4-ca6e-4f29-b463-042fd0dd6710"
    / "ls20-9607627b-b7f0efe3-1ae2-4f66-ad6e-178cf992b58f.jsonl"
)
DEFAULT_OUTPUT = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-45-summary.json"
)
DEFAULT_EVENTS = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-45-events.jsonl"
)
DEFAULT_PREDICATES = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "predicate-candidates.json"
)
DEFAULT_FREEZE = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-45-freeze.md"
)


def _raw_actions(reference: TraceReference, option: OptionCandidate) -> tuple[int, ...]:
    return reference.segment_actions[option.segment_range[0]:option.segment_range[1]]


def _execute_masked_m4_assay(
    *,
    option: OptionCandidate,
    reference: TraceReference,
) -> dict[str, Any]:
    raw = _raw_actions(reference, option)
    start = tuple(option.initiation_condition["actor_position"])
    blocked = frozenset({((int(start[0]), int(start[1])), raw[0])}) if raw else frozenset()
    result = OptionExecutor().execute(
        option=option,
        current_position=(int(start[0]), int(start[1])),
        current_signature=SemanticSignature.from_labels(
            option.initiation_condition.get("semantic_signature", ())
        ),
        action_deltas=reference.action_model,
        blocked_edges=blocked,
        final_signature=SemanticSignature.from_labels(
            option.expected_semantic_delta.get("to_signature", ())
        ),
    )
    return {
        "completed": result.completed,
        "termination_predicate_satisfied": result.termination_predicate_satisfied,
        "plan": list(result.plan),
        "plan_differs_from_raw_actions": tuple(result.plan) != raw,
        "failure_reason": result.failure_reason,
        "records": list(result.records),
    }


def _select_option(
    *,
    options: tuple[OptionCandidate, ...],
    reference: TraceReference,
) -> tuple[OptionCandidate, dict[str, Any]]:
    attempts = []
    for option in options:
        execution = _execute_masked_m4_assay(option=option, reference=reference)
        attempts.append({
            "option_id": option.option_id,
            "selected_predicate_id": option.selected_predicate_id,
            "selected_abstraction_level": option.predicate_abstraction_level,
            "completed": execution["completed"],
            "plan": execution["plan"],
            "failure_reason": execution["failure_reason"],
        })
        if execution["completed"] and option.predicate_abstraction_level >= 2:
            return option, {"attempts": attempts, "execution": execution}
    if not options:
        raise ValueError("M4.5 requires at least one option candidate")
    option = options[0]
    return option, {
        "attempts": attempts,
        "execution": _execute_masked_m4_assay(option=option, reference=reference),
    }


def evaluate(
    *,
    trace_path: Path,
    output_events: Path,
    output_predicates: Path,
    output_freeze: Path,
) -> dict[str, Any]:
    reference = TraceReference.from_steps(tuple(iter_trace_steps(trace_path)))
    base_options = OptionSegmenter().induce(reference)
    options = tuple(
        PredicateInducer().enrich(option=option, reference=reference)
        for option in base_options
    )
    chosen, selection = _select_option(options=options, reference=reference)
    output_events.parent.mkdir(parents=True, exist_ok=True)
    output_predicates.parent.mkdir(parents=True, exist_ok=True)

    predicate_records = [
        {
            "option_id": option.option_id,
            "selected_predicate_id": option.selected_predicate_id,
            "selected_predicate_expression_hash": option.selected_predicate_expression_hash,
            "selected_abstraction_level": option.predicate_abstraction_level,
            "candidates": [
                candidate.to_record()
                for candidate in option.termination_predicate_candidates
            ],
            "guard_predicates": list(option.guard_predicates),
        }
        for option in options
    ]
    output_predicates.write_text(
        json.dumps(predicate_records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with output_events.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "milestone": "M4_5_PREDICATE_ABSTRACTION",
            "test_id": "PREDICATE_SELECTION",
            "chosen_option_id": chosen.option_id,
            "selected_predicate_id": chosen.selected_predicate_id,
            "selected_predicate_expression_hash": chosen.selected_predicate_expression_hash,
            "selected_abstraction_level": chosen.predicate_abstraction_level,
            "selection_attempts": selection["attempts"],
        }, sort_keys=True) + "\n")
        stream.write(json.dumps({
            "milestone": "M4_5_PREDICATE_ABSTRACTION",
            "test_id": "MASKED_EXECUTION_VALIDATION",
            "option_id": chosen.option_id,
            **selection["execution"],
        }, sort_keys=True) + "\n")

    candidates = [
        candidate
        for option in options
        for candidate in option.termination_predicate_candidates
    ]
    level2_candidates = [
        candidate
        for candidate in candidates
        if candidate.abstraction_level >= 2
    ]
    fallback_available = any(
        candidate.kind == "exact_position"
        for candidate in candidates
    )
    selected_level = chosen.predicate_abstraction_level
    complete = (
        selected_level >= 2
        and bool(selection["execution"]["completed"])
        and fallback_available
    )
    partial_low = bool(level2_candidates) and selected_level == 1 and fallback_available
    status = (
        "complete_with_known_limits"
        if complete
        else "partial_with_low_abstraction_predicate"
        if partial_low
        else "failed"
    )
    summary = {
        "milestone": "M4_5_PREDICATE_ABSTRACTION",
        "status": status,
        "trace": str(trace_path),
        "event_artifact": str(output_events),
        "predicate_artifact": str(output_predicates),
        "freeze_artifact": str(output_freeze),
        "option_count": len(options),
        "options_with_multiple_predicates": sum(
            len(option.termination_predicate_candidates) > 1
            for option in options
        ),
        "chosen_option_id": chosen.option_id,
        "selected_predicate_id": chosen.selected_predicate_id,
        "selected_predicate_expression_hash": chosen.selected_predicate_expression_hash,
        "max_candidate_abstraction_level": max(
            (candidate.abstraction_level for candidate in candidates),
            default=0,
        ),
        "selected_abstraction_level": selected_level,
        "fallback_exact_predicate_available": fallback_available,
        "selected_predicate_validates_m4": bool(selection["execution"]["completed"]),
        "selected_predicate_terminal_selective": any(
            candidate.selected and candidate.terminal_selective
            for candidate in chosen.termination_predicate_candidates
        ),
        "partial_reason": None if complete else (
            "no_honest_level_2_plus_predicate_selected"
            if partial_low
            else "predicate_generation_or_validation_failed"
        ),
        "selection_attempts": selection["attempts"],
        "masked_execution_result": {
            key: value
            for key, value in selection["execution"].items()
            if key != "records"
        },
        "known_limits": {
            "trace_induced_predicates": True,
            "level_2_predicates_are_spatial": True,
            "not_self_discovered": True,
            "not_cross_option_planning": True,
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _write_freeze(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "# Milestone 4.5 Freeze",
            "",
            f"Status: `{summary['status']}`",
            "",
            "Claim: 0001 can generate and evaluate observable termination-predicate "
            "candidates, prefer a higher-abstraction predicate when it remains "
            "terminal-selective, and preserve an exact fallback.",
            "",
            "Key evidence:",
            "",
            f"- chosen option: `{summary['chosen_option_id']}`",
            f"- selected predicate: `{summary['selected_predicate_id']}`",
            f"- expression hash: `{summary['selected_predicate_expression_hash']}`",
            f"- selected abstraction level: `{summary['selected_abstraction_level']}`",
            f"- fallback exact predicate available: `{summary['fallback_exact_predicate_available']}`",
            f"- validates M4 masked execution: `{summary['selected_predicate_validates_m4']}`",
            f"- terminal selective: `{summary['selected_predicate_terminal_selective']}`",
            "",
            "Known limits: predicates remain trace-induced; level-2 predicates are still "
            "spatial; no self-generated options or cross-option planning yet.",
            "",
        ]),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--predicates", type=Path, default=DEFAULT_PREDICATES)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()
    summary = evaluate(
        trace_path=args.trace,
        output_events=args.events,
        output_predicates=args.predicates,
        output_freeze=args.freeze,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
