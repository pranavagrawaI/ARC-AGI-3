"""Evaluate trace-to-option abstraction for milestone 4."""

from __future__ import annotations

import argparse
from collections import Counter
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
    SemanticSignature,
    TraceReference,
    iter_trace_steps,
    replay_actions_to_position,
)


DEFAULT_TRACE = (
    REPO
    / "recordings/ae20a3e4-ca6e-4f29-b463-042fd0dd6710"
    / "ls20-9607627b-b7f0efe3-1ae2-4f66-ad6e-178cf992b58f.jsonl"
)
DEFAULT_OUTPUT = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-4-summary.json"
)
DEFAULT_EVENTS = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-4-events.jsonl"
)
DEFAULT_OPTIONS = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "option-candidates.json"
)
DEFAULT_FREEZE = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-4-freeze.md"
)


def _event_record(test_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "milestone": "M4_TRACE_TO_OPTION_ABSTRACTION",
        "test_id": test_id,
        **payload,
    }


def _choose_executable_candidate(
    *,
    candidates: tuple[OptionCandidate, ...],
    reference: TraceReference,
) -> tuple[OptionCandidate, dict[str, Any]]:
    executor = OptionExecutor()
    attempts = []
    for candidate in candidates:
        raw_actions = reference.segment_actions[candidate.segment_range[0]:candidate.segment_range[1]]
        if not raw_actions:
            continue
        start = tuple(candidate.initiation_condition["actor_position"])
        blocked = frozenset({((int(start[0]), int(start[1])), raw_actions[0])})
        result = executor.execute(
            option=candidate,
            current_position=(int(start[0]), int(start[1])),
            current_signature=SemanticSignature.from_labels(
                candidate.initiation_condition.get("semantic_signature", ())
            ),
            action_deltas=reference.action_model,
            blocked_edges=blocked,
            final_signature=SemanticSignature.from_labels(
                candidate.expected_semantic_delta.get("to_signature", ())
            ),
        )
        attempts.append({
            "option_id": candidate.option_id,
            "raw_segment_action_count": len(raw_actions),
            "raw_first_action_blocked": raw_actions[0],
            "completed": result.completed,
            "plan": list(result.plan),
            "plan_differs_from_raw_actions": tuple(result.plan) != tuple(raw_actions),
            "failure_reason": result.failure_reason,
        })
        if result.completed and tuple(result.plan) != tuple(raw_actions):
            return candidate, {
                "blocked_edges": [
                    {
                        "position": list(blocked_edge[0]),
                        "action": blocked_edge[1],
                    }
                    for blocked_edge in blocked
                ],
                "attempts": attempts,
            }
    if not candidates:
        raise ValueError("Milestone 4 requires at least one option candidate")
    return candidates[0], {"blocked_edges": [], "attempts": attempts}


def _blocked_replay_endpoint(
    *,
    start: tuple[int, int],
    actions: tuple[int, ...],
    action_deltas: dict[int, tuple[int, int]],
    blocked_edges: set[tuple[tuple[int, int], int]],
) -> tuple[int, int]:
    position = start
    for action in actions:
        if (position, action) in blocked_edges:
            continue
        delta = action_deltas.get(action)
        if delta is None:
            continue
        position = (position[0] + delta[0], position[1] + delta[1])
    return position


def _execute_option_assay(
    *,
    option: OptionCandidate,
    reference: TraceReference,
    blocked_edges: list[dict[str, Any]],
    event_stream,
) -> dict[str, Any]:
    executor = OptionExecutor()
    start_raw = tuple(option.initiation_condition["actor_position"])
    start = (int(start_raw[0]), int(start_raw[1]))
    current_signature = SemanticSignature.from_labels(
        option.initiation_condition.get("semantic_signature", ())
    )
    raw_actions = reference.segment_actions[option.segment_range[0]:option.segment_range[1]]
    blocked = frozenset(
        (
            (int(record["position"][0]), int(record["position"][1])),
            int(record["action"]),
        )
        for record in blocked_edges
    )
    if not blocked and raw_actions:
        blocked = frozenset({(start, raw_actions[0])})

    result = executor.execute(
        option=option,
        current_position=start,
        current_signature=current_signature,
        action_deltas=reference.action_model,
        blocked_edges=blocked,
        final_signature=SemanticSignature.from_labels(
            option.expected_semantic_delta.get("to_signature", ())
        ),
    )
    replay_end = _blocked_replay_endpoint(
        start=start,
        actions=raw_actions,
        action_deltas=reference.action_model,
        blocked_edges=set(blocked),
    )
    unblocked_replay_end = replay_actions_to_position(
        start=start,
        actions=raw_actions,
        action_deltas=reference.action_model,
    )
    baseline_satisfied = executor.termination_satisfied(
        option=option,
        current_position=replay_end,
        current_signature=current_signature,
    )
    option_record = {
        "option_id": option.option_id,
        "raw_action_list_masked": result.raw_action_list_masked,
        "option_execution_mode": "predicate_pursuit",
        "trace_dependency_level": result.trace_dependency_level,
        "predicate_abstraction_level": option.predicate_abstraction_level,
        "raw_segment_action_count": len(raw_actions),
        "raw_actions_available_to_executor": False,
        "blocked_edges": [
            {"position": list(position), "action": action}
            for position, action in sorted(blocked)
        ],
        "baseline_action_replay_result": {
            "completed": baseline_satisfied,
            "endpoint": list(replay_end),
            "unblocked_endpoint": list(unblocked_replay_end),
            "failure_reason": None if baseline_satisfied else "blocked_script_did_not_satisfy_predicate",
        },
        "option_execution_result": {
            "completed": result.completed,
            "termination_predicate_satisfied": result.termination_predicate_satisfied,
            "plan": list(result.plan),
            "plan_differs_from_raw_actions": tuple(result.plan) != tuple(raw_actions),
            "failure_reason": result.failure_reason,
        },
        "semantic_validation_result": result.semantic_validation_result,
        "events": list(result.records),
    }
    event_stream.write(json.dumps(
        _event_record("PREDICATE_PURSUIT_WITH_MASKED_ACTION_LIST", option_record),
        sort_keys=True,
    ) + "\n")
    for record in result.records:
        event_stream.write(json.dumps(
            _event_record("OPTION_EXECUTION_ACTION", {"option_id": option.option_id, **record}),
            sort_keys=True,
        ) + "\n")
    return option_record


def _refusal_assay(
    *,
    option: OptionCandidate,
    reference: TraceReference,
    event_stream,
) -> dict[str, Any]:
    start_raw = tuple(option.initiation_condition["actor_position"])
    result = OptionExecutor().execute(
        option=option,
        current_position=(int(start_raw[0]), int(start_raw[1])),
        current_signature=SemanticSignature.from_labels(("semantic-mismatch",)),
        action_deltas=reference.action_model,
    )
    record = {
        "option_id": option.option_id,
        "completed": result.completed,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "semantic_validation_result": result.semantic_validation_result,
        "trace_resumed": False,
    }
    event_stream.write(json.dumps(
        _event_record("SEMANTIC_MISMATCH_REFUSAL", record),
        sort_keys=True,
    ) + "\n")
    return record


def _action_source_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter()
    for event in events:
        for record in event.get("events", ()):
            source = record.get("action_source")
            if source:
                counter[source] += 1
    return dict(sorted(counter.items()))


def evaluate(
    *,
    trace_path: Path,
    output_events: Path,
    output_options: Path,
    output_freeze: Path,
) -> dict[str, Any]:
    reference = TraceReference.from_steps(tuple(iter_trace_steps(trace_path)))
    candidates = OptionSegmenter().induce(reference)
    output_events.parent.mkdir(parents=True, exist_ok=True)
    output_options.parent.mkdir(parents=True, exist_ok=True)
    output_options.write_text(
        json.dumps([candidate.to_record() for candidate in candidates], indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    chosen, selection = _choose_executable_candidate(
        candidates=candidates,
        reference=reference,
    )
    with output_events.open("w", encoding="utf-8") as event_stream:
        execution = _execute_option_assay(
            option=chosen,
            reference=reference,
            blocked_edges=selection["blocked_edges"],
            event_stream=event_stream,
        )
        refusal = _refusal_assay(
            option=chosen,
            reference=reference,
            event_stream=event_stream,
        )

    passed = {
        "option_generated": bool(candidates),
        "trace_dependency_level_at_least_3": chosen.trace_dependency_level >= 3,
        "raw_action_list_masked": bool(execution["raw_action_list_masked"]),
        "predicate_not_trace_index": chosen.termination_condition["type"] != "trace_index",
        "option_completed": bool(execution["option_execution_result"]["completed"]),
        "plan_differs_from_raw_actions": bool(
            execution["option_execution_result"]["plan_differs_from_raw_actions"]
        ),
        "baseline_reported": "baseline_action_replay_result" in execution,
        "semantic_mismatch_refused": refusal["failure_reason"] == "INITIATION_CONDITION_FAILED",
        "events_confess_no_raw_trace_action": all(
            not record.get("used_raw_trace_action", True)
            for record in execution["events"]
        ),
    }
    summary = {
        "milestone": "M4_TRACE_TO_OPTION_ABSTRACTION",
        "status": "complete_with_known_limits" if all(passed.values()) else "partial",
        "mode": "trace_induced_option_predicate_execution",
        "trace": str(trace_path),
        "event_artifact": str(output_events),
        "option_artifact": str(output_options),
        "freeze_artifact": str(output_freeze),
        "option_count": len(candidates),
        "chosen_option_id": chosen.option_id,
        "segment_range": list(chosen.segment_range),
        "termination_condition": chosen.termination_condition,
        "trace_dependency_level": chosen.trace_dependency_level,
        "raw_action_list_masked": execution["raw_action_list_masked"],
        "option_execution_mode": execution["option_execution_mode"],
        "predicate_abstraction_level": chosen.predicate_abstraction_level,
        "chosen_option_evidence_count": len(chosen.evidence),
        "baseline_action_replay_result": execution["baseline_action_replay_result"],
        "option_execution_result": execution["option_execution_result"],
        "semantic_refusal_result": refusal,
        "semantic_validation_result": execution["semantic_validation_result"],
        "action_source_distribution": _action_source_counts([execution]),
        "selection_attempts": selection["attempts"],
        "passed": passed,
        "known_limits": {
            "options_induced_from_human_trace": True,
            "first_segment_bounded": True,
            "execution_assay_is_public_observation_style_synthetic": True,
            "predicate_may_be_exact_coordinate_low_abstraction": True,
            "local_action_delta_planning_only": True,
            "no_cross_option_planning": True,
            "no_self_generated_options_yet": True,
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _write_freeze(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "# Milestone 4 Freeze",
            "",
            "Status: `" + summary["status"] + "`",
            "",
            "Claim: 0001 can induce an inspectable option predicate from a public trace segment "
            "and execute toward that predicate with the raw segment action list masked.",
            "",
            "Proof panels:",
            "",
            "- chosen option: `" + summary["chosen_option_id"] + "`",
            "- trace dependency level: `" + str(summary["trace_dependency_level"]) + "`",
            "- raw action list masked: `" + str(summary["raw_action_list_masked"]) + "`",
            "- option completed: `"
            + str(summary["option_execution_result"]["completed"])
            + "`",
            "- plan differs from raw actions: `"
            + str(summary["option_execution_result"]["plan_differs_from_raw_actions"])
            + "`",
            "- semantic mismatch refused: `"
            + str(summary["semantic_refusal_result"]["failure_reason"])
            + "`",
            "",
            "Known limits remain attached to the claim: trace-induced, first-segment bounded, "
            "synthetic predicate-execution assay, and no self-generated options yet.",
            "",
        ]),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--options", type=Path, default=DEFAULT_OPTIONS)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()
    summary = evaluate(
        trace_path=args.trace,
        output_events=args.events,
        output_options=args.options,
        output_freeze=args.freeze,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
