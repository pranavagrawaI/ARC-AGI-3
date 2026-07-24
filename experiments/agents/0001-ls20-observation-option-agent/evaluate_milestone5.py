"""Evaluate option reuse under route variation for milestone 5."""

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
    selected_predicate_expression_hash,
    selected_termination_expression,
)


DEFAULT_TRACE = (
    REPO
    / "recordings/ae20a3e4-ca6e-4f29-b463-042fd0dd6710"
    / "ls20-9607627b-b7f0efe3-1ae2-4f66-ad6e-178cf992b58f.jsonl"
)
DEFAULT_OUTPUT = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-5-summary.json"
)
DEFAULT_EVENTS = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-5-events.jsonl"
)
DEFAULT_FREEZE = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-5-freeze.md"
)


def _choose_option(reference: TraceReference) -> OptionCandidate:
    options = tuple(
        PredicateInducer().enrich(option=option, reference=reference)
        for option in OptionSegmenter().induce(reference)
    )
    for option in options:
        if option.predicate_abstraction_level >= 2:
            raw = _raw_actions(reference, option)
            if raw and _execute(option=option, reference=reference, blocked={(start_position(option), raw[0])})["completed"]:
                return option
    if not options:
        raise ValueError("M5 requires at least one option candidate")
    return options[0]


def _raw_actions(reference: TraceReference, option: OptionCandidate) -> tuple[int, ...]:
    return reference.segment_actions[option.segment_range[0]:option.segment_range[1]]


def start_position(option: OptionCandidate) -> tuple[int, int]:
    value = option.initiation_condition["actor_position"]
    return (int(value[0]), int(value[1]))


def _start_signature(option: OptionCandidate) -> SemanticSignature:
    return SemanticSignature.from_labels(
        option.initiation_condition.get("semantic_signature", ())
    )


def _final_signature(option: OptionCandidate) -> SemanticSignature:
    return SemanticSignature.from_labels(
        option.expected_semantic_delta.get("to_signature", ())
    )


def _start_validation(
    *,
    option: OptionCandidate,
    reference: TraceReference,
    position: tuple[int, int],
    signature: SemanticSignature,
    blocked: set[tuple[tuple[int, int], int]],
) -> dict[str, Any]:
    executor = OptionExecutor()
    initiation = position == start_position(option) and signature.compatible_with(_start_signature(option))
    initially_terminal = executor.termination_satisfied(
        option=option,
        current_position=position,
        current_signature=signature,
    )
    reachable = bool(_execute(
        option=option,
        reference=reference,
        blocked=blocked,
        position=position,
        signature=signature,
    )["plan"])
    return {
        "initiation_condition": initiation,
        "termination_condition_initially_true": initially_terminal,
        "semantic_guards": signature.compatible_with(_start_signature(option)),
        "reachable_to_termination_likely": reachable,
        "valid_start": initiation and not initially_terminal and reachable,
    }


def _execute(
    *,
    option: OptionCandidate,
    reference: TraceReference,
    blocked: set[tuple[tuple[int, int], int]],
    position: tuple[int, int] | None = None,
    signature: SemanticSignature | None = None,
    world_diffs: tuple[tuple[dict[str, Any], ...], ...] | None = None,
) -> dict[str, Any]:
    start = position or start_position(option)
    result = OptionExecutor().execute(
        option=option,
        current_position=start,
        current_signature=signature or _start_signature(option),
        action_deltas=reference.action_model,
        blocked_edges=frozenset(blocked),
        final_signature=_final_signature(option),
        world_diffs=world_diffs,
    )
    return {
        "completed": result.completed,
        "status": result.status,
        "plan": tuple(result.plan),
        "failure_reason": result.failure_reason,
        "termination_predicate_satisfied": result.termination_predicate_satisfied,
        "records": result.records,
    }


def _path(
    *,
    start: tuple[int, int],
    actions: tuple[int, ...],
    action_deltas: dict[int, tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    positions = [start]
    current = start
    for action in actions:
        delta = action_deltas[action]
        current = (current[0] + delta[0], current[1] + delta[1])
        positions.append(current)
    return tuple(positions)


def _edit_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    rows = len(left) + 1
    cols = len(right) + 1
    table = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        table[i][0] = i
    for j in range(cols):
        table[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            table[i][j] = min(
                table[i - 1][j] + 1,
                table[i][j - 1] + 1,
                table[i - 1][j - 1] + cost,
            )
    return table[-1][-1]


def _assay(
    *,
    assay_id: str,
    option: OptionCandidate,
    reference: TraceReference,
    blocked: set[tuple[tuple[int, int], int]],
    raw: tuple[int, ...],
    m4_plan: tuple[int, ...],
    invalid_signature: bool = False,
    world_drift: bool = False,
) -> dict[str, Any]:
    signature = SemanticSignature.from_labels(("semantic-mismatch",)) if invalid_signature else _start_signature(option)
    validation = _start_validation(
        option=option,
        reference=reference,
        position=start_position(option),
        signature=signature,
        blocked=blocked,
    )
    if invalid_signature:
        result = _execute(
            option=option,
            reference=reference,
            blocked=blocked,
            signature=signature,
        )
    else:
        preliminary = _execute(option=option, reference=reference, blocked=blocked)
        diffs = None
        if world_drift and preliminary["plan"]:
            diffs = tuple(
                ({"type": "unexpected_present", "label": "semantic-mismatch"},)
                if index == 0 else ()
                for index, _action in enumerate(preliminary["plan"])
            )
        result = _execute(
            option=option,
            reference=reference,
            blocked=blocked,
            world_diffs=diffs,
        )
    plan = result["plan"]
    actor_path = _path(
        start=start_position(option),
        actions=plan,
        action_deltas=reference.action_model,
    ) if plan else (start_position(option),)
    raw_path = _path(
        start=start_position(option),
        actions=raw,
        action_deltas=reference.action_model,
    ) if raw else (start_position(option),)
    m4_path = _path(
        start=start_position(option),
        actions=m4_plan,
        action_deltas=reference.action_model,
    ) if m4_plan else (start_position(option),)
    return {
        "assay_id": assay_id,
        "option_id": option.option_id,
        "selected_predicate_id": option.selected_predicate_id,
        "selected_predicate_expression_hash": selected_predicate_expression_hash(option),
        "start_validation": validation,
        "raw_action_list_masked": True,
        "completed": result["completed"],
        "status": result["status"],
        "failure_reason": result["failure_reason"],
        "termination_predicate_satisfied": result["termination_predicate_satisfied"],
        "action_sequence": list(plan),
        "action_edit_distance_from_raw": _edit_distance(plan, raw),
        "action_edit_distance_from_m4_plan": _edit_distance(plan, m4_plan),
        "action_sequence_differs_from_raw": plan != raw,
        "action_sequence_differs_from_m4": plan != m4_plan,
        "actor_path": [list(position) for position in actor_path],
        "path_differs_from_raw": actor_path != raw_path,
        "path_differs_from_m4": actor_path != m4_path,
        "events": [dict(record) for record in result["records"]],
        "refused_before_action": not result["records"] and not result["completed"],
    }


def evaluate(
    *,
    trace_path: Path,
    output_events: Path,
    output_freeze: Path,
) -> dict[str, Any]:
    reference = TraceReference.from_steps(tuple(iter_trace_steps(trace_path)))
    option = _choose_option(reference)
    raw = _raw_actions(reference, option)
    if not raw:
        raise ValueError("selected option has no raw trace baseline for comparison")
    start = start_position(option)
    m4_blocked = {(start, raw[0])}
    m4_plan = _execute(option=option, reference=reference, blocked=m4_blocked)["plan"]
    assays = [
        _assay(
            assay_id="A_ORIGINAL_M4_BLOCKED_SCRIPT",
            option=option,
            reference=reference,
            blocked=m4_blocked,
            raw=raw,
            m4_plan=m4_plan,
        ),
        _assay(
            assay_id="C_ALTERNATE_BLOCKED_EDGE",
            option=option,
            reference=reference,
            blocked={(start, raw[0]), (start, 3)},
            raw=raw,
            m4_plan=m4_plan,
        ),
        _assay(
            assay_id="D_INVALID_INITIATION_REFUSAL",
            option=option,
            reference=reference,
            blocked=m4_blocked,
            raw=raw,
            m4_plan=m4_plan,
            invalid_signature=True,
        ),
        _assay(
            assay_id="E_SEMANTIC_MISMATCH_DURING_EXECUTION",
            option=option,
            reference=reference,
            blocked=m4_blocked,
            raw=raw,
            m4_plan=m4_plan,
            world_drift=True,
        ),
    ]
    output_events.parent.mkdir(parents=True, exist_ok=True)
    with output_events.open("w", encoding="utf-8") as stream:
        for assay in assays:
            stream.write(json.dumps({
                "milestone": "M5_OPTION_REEXECUTION_ROUTE_VARIATION",
                **assay,
            }, sort_keys=True) + "\n")
    success_assays = [
        assay
        for assay in assays
        if assay["start_validation"]["valid_start"] and assay["completed"]
    ]
    route_differs_from_m4 = any(
        assay["completed"]
        and assay["path_differs_from_raw"]
        and assay["path_differs_from_m4"]
        for assay in assays
    )
    same_predicate = len({
        (
            assay["selected_predicate_id"],
            assay["selected_predicate_expression_hash"],
        )
        for assay in assays
    }) == 1
    invalid_refused = next(
        assay
        for assay in assays
        if assay["assay_id"] == "D_INVALID_INITIATION_REFUSAL"
    )["refused_before_action"]
    semantic_refused = next(
        assay
        for assay in assays
        if assay["assay_id"] == "E_SEMANTIC_MISMATCH_DURING_EXECUTION"
    )["status"] == "WORLD_DRIFT"
    complete = (
        option.predicate_abstraction_level >= 2
        and same_predicate
        and len(success_assays) >= 2
        and route_differs_from_m4
        and invalid_refused
        and semantic_refused
    )
    status = (
        "complete_with_known_limits"
        if complete
        else "partial_with_low_abstraction_predicate"
        if option.predicate_abstraction_level < 2
        else "partial"
    )
    summary = {
        "milestone": "M5_OPTION_REEXECUTION_ROUTE_VARIATION",
        "status": status,
        "trace": str(trace_path),
        "event_artifact": str(output_events),
        "freeze_artifact": str(output_freeze),
        "option_id": option.option_id,
        "selected_predicate_id": option.selected_predicate_id,
        "selected_predicate_expression_hash": selected_predicate_expression_hash(option),
        "selected_predicate_expression": selected_termination_expression(option),
        "selected_predicate_abstraction_level": option.predicate_abstraction_level,
        "same_predicate_across_assays": same_predicate,
        "trace_dependency_level": option.trace_dependency_level,
        "raw_action_list_masked": True,
        "valid_start_success_count": len(success_assays),
        "route_differs_from_raw": any(assay["path_differs_from_raw"] for assay in success_assays),
        "route_differs_from_m4": route_differs_from_m4,
        "invalid_initiation_refused": invalid_refused,
        "semantic_mismatch_refused": semantic_refused,
        "m4_plan": list(m4_plan),
        "raw_actions": list(raw),
        "assays": assays,
        "known_limits": {
            "trace_induced_option": True,
            "first_segment_bounded": True,
            "synthetic_public_observation_style_route_assays": True,
            "no_self_generated_options": True,
            "no_cross_option_planning": True,
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _write_freeze(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "# Milestone 5 Freeze",
            "",
            f"Status: `{summary['status']}`",
            "",
            "Claim: 0001 can re-execute the same trace-induced option under "
            "route variation with the same selected predicate and raw trace actions masked.",
            "",
            "Key evidence:",
            "",
            f"- option: `{summary['option_id']}`",
            f"- selected predicate: `{summary['selected_predicate_id']}`",
            f"- expression hash: `{summary['selected_predicate_expression_hash']}`",
            f"- same predicate across assays: `{summary['same_predicate_across_assays']}`",
            f"- valid-start successes: `{summary['valid_start_success_count']}`",
            f"- route differs from raw: `{summary['route_differs_from_raw']}`",
            f"- route differs from M4: `{summary['route_differs_from_m4']}`",
            f"- invalid initiation refused: `{summary['invalid_initiation_refused']}`",
            f"- semantic mismatch refused: `{summary['semantic_mismatch_refused']}`",
            "",
            "Known limits: first-segment bounded, synthetic route assays, trace-induced "
            "option, no self-generated options yet.",
            "",
        ]),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()
    summary = evaluate(
        trace_path=args.trace,
        output_events=args.events,
        output_freeze=args.freeze,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
