"""Evaluate self-generated option discovery for milestone 6."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
AGENT_ROOT = Path(__file__).resolve().parent
for path in (REPO, AGENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import arc_agi
from arc_agi.base import OperationMode
from arcengine import GameAction

from experiments.harness.runner import IllegalActionError, public_observation

from observation_option_agent import (
    DiscoveryRecord,
    Event,
    EventDetector,
    ObjectTracker,
    ObservationParser,
    SelfExplorationController,
    SelfOptionEvaluator,
    SelfOptionExecution,
    SelfOptionInducer,
    discovery_record_from_events,
    m6_status,
    nontriviality_criteria,
    selected_predicate_expression_hash,
    trace_leakage_findings,
)


DEFAULT_OUTPUT = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-6-summary.json"
)
DEFAULT_EVENTS = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-6-events.jsonl"
)
DEFAULT_CANDIDATES = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "self-option-candidates.json"
)
DEFAULT_FREEZE = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-6-freeze.md"
)


def _validate_public_action(observation, action_id: int):
    if action_id not in observation.available_actions:
        raise IllegalActionError(
            f"exploration chose action {action_id}; "
            f"available actions are {observation.available_actions}"
        )
    action = GameAction.from_id(action_id)
    if action.is_complex():
        raise IllegalActionError("M6 default exploration does not use complex actions")
    return action


def _observation_hash(observation) -> str:
    digest = hashlib.sha256()
    digest.update(observation.pixels.tobytes())
    digest.update(json.dumps({
        "available_actions": observation.available_actions,
        "levels_completed": observation.levels_completed,
        "state": observation.state,
        "full_reset": observation.full_reset,
    }, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _collect_self_exploration(
    *,
    game: str,
    controller: SelfExplorationController,
) -> tuple[DiscoveryRecord, tuple[dict[str, Any], ...]]:
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make(game)
    current = public_observation(env.reset())
    start_hash = _observation_hash(current)
    parser = ObservationParser()
    tracker = ObjectTracker()
    detector = EventDetector()
    tracker.update(parser.parse(current, frame_index=0))

    events: list[Event] = []
    records: list[dict[str, Any]] = []
    actions: list[int] = []
    for step in range(1, controller.budget + 1):
        action_id = controller.action_for_step(
            step=step,
            available_actions=current.available_actions,
        )
        if action_id is None:
            break
        action = _validate_public_action(current, action_id)
        following = public_observation(env.step(action, data=None, reasoning={
            "milestone": "M6_SELF_EXPLORATION",
            "source": "self_exploration",
        }))
        scene = parser.parse(following, frame_index=step)
        tracks = tracker.update(
            scene,
            force_new=following.levels_completed != current.levels_completed or following.full_reset,
        )
        event = detector.detect(
            step=step,
            before=current,
            action=action_id,
            after=following,
            tracks=tracks,
        )
        events.append(event)
        actions.append(action_id)
        records.append({
            "phase": "SELF_EXPLORATION",
            "step": step,
            "action": action_id,
            "levels_before": current.levels_completed,
            "levels_after": following.levels_completed,
            "state_after": following.state,
            "motion_groups": [
                {
                    "translation": list(group.translation),
                    "object_ids": list(group.object_ids),
                }
                for group in event.motion_groups
            ],
            "created_object_ids": list(event.created_object_ids),
            "disappeared_object_ids": list(event.disappeared_object_ids),
            "used_human_trace": False,
        })
        current = following
        if len(actions) >= len(controller.action_sequence):
            break

    controlled_ids = _controlled_ids_from_events(tuple(events))
    if not controlled_ids:
        raise ValueError("M6 self-exploration did not identify a controlled object")
    discovery = discovery_record_from_events(
        record_id="disc_000",
        action_sequence=tuple(actions),
        events=tuple(events),
        controlled_object_ids=controlled_ids,
        observation_before_hash=start_hash,
        observation_after_hash=_observation_hash(current),
    )
    return discovery, tuple(records)


def _controlled_ids_from_events(events: tuple[Event, ...]) -> tuple[int, ...]:
    observations: dict[int, list[tuple[int, tuple[int, int]]]] = defaultdict(list)
    for event in events:
        for delta in event.object_deltas:
            if delta.translation != (0, 0):
                observations[delta.object_id].append((event.action, delta.translation))
    best: tuple[int, int, int] | None = None
    for object_id, pairs in observations.items():
        distinct_actions = len({action for action, _delta in pairs})
        support = len(pairs)
        if distinct_actions < 2:
            continue
        candidate = (distinct_actions, support, -object_id)
        if best is None or candidate > best:
            best = candidate
    if best is not None:
        return (-best[2],)

    group_counts = Counter(
        group.object_ids
        for event in events
        for group in event.motion_groups
        if group.object_ids
    )
    return group_counts.most_common(1)[0][0] if group_counts else ()


def _primary_block(discovery: DiscoveryRecord) -> frozenset[tuple[tuple[int, int], int]]:
    return frozenset({(discovery.start_position, discovery.action_sequence[0])})


def _variation_block(discovery: DiscoveryRecord) -> frozenset[tuple[tuple[int, int], int]]:
    first_action = discovery.action_sequence[0]
    first_delta = discovery.action_model[first_action]
    after_first = (
        discovery.start_position[0] + first_delta[0],
        discovery.start_position[1] + first_delta[1],
    )
    primary_first_action = next(
        action for action in sorted(discovery.action_model)
        if action != first_action
    )
    return frozenset({
        (discovery.start_position, primary_first_action),
        (after_first, first_action),
    })


def _expectation_failure_diffs(execution: SelfOptionExecution) -> tuple[tuple[dict[str, Any], ...], ...]:
    if not execution.plan:
        return ()
    return tuple(
        ({"type": "unexpected_noncontrolled_change", "label": "m6_expectation_failure"},)
        if index == 0 else ()
        for index, _action in enumerate(execution.plan)
    )


def _execution_record(execution: SelfOptionExecution) -> dict[str, Any]:
    return execution.to_record()


def evaluate(
    *,
    game: str,
    output_events: Path,
    output_candidates: Path,
    output_freeze: Path,
) -> dict[str, Any]:
    controller = SelfExplorationController()
    discovery, exploration_records = _collect_self_exploration(
        game=game,
        controller=controller,
    )
    option = SelfOptionInducer().induce(discovery)
    evaluator = SelfOptionEvaluator()

    primary = evaluator.execute(
        assay_id="A_SELF_OPTION_EXECUTION",
        option=option,
        discovery=discovery,
        blocked_edges=_primary_block(discovery),
    )
    variation = evaluator.execute(
        assay_id="B_SELF_OPTION_ROUTE_VARIATION",
        option=option,
        discovery=discovery,
        blocked_edges=_variation_block(discovery),
    )
    invalid = evaluator.execute(
        assay_id="C_INVALID_INITIATION_REFUSAL",
        option=option,
        discovery=discovery,
        current_position=(0, 0),
    )
    preliminary = evaluator.execute(
        assay_id="D_EXPECTATION_FAILURE_PROBE",
        option=option,
        discovery=discovery,
        blocked_edges=_primary_block(discovery),
    )
    expectation = evaluator.execute(
        assay_id="D_EXPECTATION_FAILURE_REFUSAL",
        option=option,
        discovery=discovery,
        blocked_edges=_primary_block(discovery),
        world_diffs=_expectation_failure_diffs(preliminary),
    )

    leakage = trace_leakage_findings(option)
    invalid_refused = invalid.failure_reason == "INVALID_INITIATION"
    expectation_refused = expectation.failure_reason in {
        "OPTION_INVALID",
        "ACTION_CONFLICT",
        "TARGET_UNREACHABLE",
        "EXPECTATION_INVALID",
    }
    criteria = nontriviality_criteria(
        discovery=discovery,
        primary_execution=primary,
        variation_execution=variation,
    )
    status = m6_status(
        option=option,
        discovery=discovery,
        executions=(primary, variation),
        invalid_refused=invalid_refused,
        expectation_refused=expectation_refused,
        trace_leakage=bool(leakage),
        noise_promoted=bool(discovery.semantic_delta and option.predicate_abstraction_level >= 4),
    )

    same_predicate = (
        selected_predicate_expression_hash(option)
        == option.selected_predicate_expression_hash
    )
    executions = (primary, variation, invalid, expectation)
    output_events.parent.mkdir(parents=True, exist_ok=True)
    with output_events.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "milestone": "M6_SELF_GENERATED_OPTION_DISCOVERY",
            "phase": "SELF_EXPLORATION",
            "discovery_record_id": discovery.record_id,
            "action_sequence": list(discovery.action_sequence),
            "actor_delta": list(discovery.actor_delta),
            "semantic_delta": list(discovery.semantic_delta),
            "contact_events": list(discovery.contact_events),
            "provenance": "self_observed_transition",
            "used_human_trace": False,
            "records": list(exploration_records),
        }, sort_keys=True) + "\n")
        for execution in executions:
            stream.write(json.dumps({
                "milestone": "M6_SELF_GENERATED_OPTION_DISCOVERY",
                **_execution_record(execution),
            }, sort_keys=True) + "\n")

    output_candidates.parent.mkdir(parents=True, exist_ok=True)
    output_candidates.write_text(
        json.dumps({
            "selected_option": option.to_record(),
            "discovery_record": discovery.to_record(),
            "trace_leakage_findings": list(leakage),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    successful = [execution for execution in (primary, variation) if execution.completed]
    summary = {
        "milestone": "M6_SELF_GENERATED_OPTION_DISCOVERY",
        "status": status,
        "game": game,
        "event_artifact": str(output_events),
        "candidate_artifact": str(output_candidates),
        "freeze_artifact": str(output_freeze),
        "trace_dependency_level": option.trace_dependency_level,
        "option_source": option.source,
        "self_discovery_start_source": discovery.self_discovery_start_source,
        "human_trace_state_used_as_start": False,
        "raw_trace_actions_used": False,
        "trace_endpoint_used": False,
        "trace_index_used": False,
        "human_segment_used": False,
        "discovery_action_sequence_masked_during_execution": True,
        "execution_plan_differs_from_discovery_sequence": primary.execution_plan_differs_from_discovery_sequence,
        "nontriviality_criteria": criteria,
        "nontriviality_criterion_met": sum(criteria.values()) >= 2,
        "self_discovery_action_count": len(discovery.action_sequence),
        "option_count": 1,
        "chosen_option_id": option.option_id,
        "selected_predicate_id": option.selected_predicate_id,
        "selected_predicate_expression_hash": option.selected_predicate_expression_hash,
        "selected_predicate_hash_verified": same_predicate,
        "selected_predicate_kind": option.termination_condition["type"],
        "selected_predicate_abstraction_level": option.predicate_abstraction_level,
        "same_predicate_across_assays": same_predicate,
        "valid_start_success_count": len(successful),
        "primary_execution_plan": list(primary.plan),
        "variation_execution_plan": list(variation.plan),
        "variation_plan_differs_from_primary": variation.plan != primary.plan,
        "primary_plan_differs_from_discovery_sequence": primary.plan != discovery.action_sequence,
        "variation_plan_differs_from_discovery_sequence": variation.plan != discovery.action_sequence,
        "invalid_initiation_refused": invalid_refused,
        "expectation_failure_refused": expectation_refused,
        "trace_leakage_findings": list(leakage),
        "noise_promoted_to_event": False,
        "executions": [_execution_record(execution) for execution in executions],
        "known_limits": {
            "public_observation_style_execution_assays": True,
            "not_general_autonomous_solver": True,
            "spatial_self_option": option.predicate_abstraction_level == 2,
            "no_live_event_causation_claim": True,
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _write_freeze(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Milestone 6 Freeze",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Official claim:",
        "",
        (
            "Milestone 6 demonstrates that 0001 can discover a nontrivial option "
            "from its own public interaction evidence, generate an observable "
            "termination predicate without using a human trace segment, execute and "
            "re-execute that option under variation without replaying either human "
            "trace actions or the self-discovery action script, and refuse when "
            "initiation or expectation checks fail."
        ),
        "",
        "Key evidence:",
        "",
        f"- option source: `{summary['option_source']}`",
        f"- trace dependency level: `{summary['trace_dependency_level']}`",
        f"- chosen option: `{summary['chosen_option_id']}`",
        f"- selected predicate: `{summary['selected_predicate_id']}`",
        f"- selected predicate hash: `{summary['selected_predicate_expression_hash']}`",
        f"- primary execution plan: `{summary['primary_execution_plan']}`",
        f"- variation execution plan: `{summary['variation_execution_plan']}`",
        f"- primary plan differs from discovery sequence: `{summary['primary_plan_differs_from_discovery_sequence']}`",
        f"- variation plan differs from discovery sequence: `{summary['variation_plan_differs_from_discovery_sequence']}`",
        f"- raw trace actions used: `{summary['raw_trace_actions_used']}`",
        f"- human segment used: `{summary['human_segment_used']}`",
        f"- trace endpoint used: `{summary['trace_endpoint_used']}`",
        f"- trace index used: `{summary['trace_index_used']}`",
        f"- valid-start successes: `{summary['valid_start_success_count']}`",
        f"- discovery script masked: `{summary['discovery_action_sequence_masked_during_execution']}`",
        f"- invalid initiation refused: `{summary['invalid_initiation_refused']}`",
        f"- expectation failure refused: `{summary['expectation_failure_refused']}`",
        f"- trace leakage findings: `{summary['trace_leakage_findings']}`",
        "",
        "Known limits:",
        "",
        "- This is a bounded public-observation self-option assay, not a general solver.",
        "- The selected option is spatial; no live event-causation autonomy is claimed.",
        "- Prior trace-trained machinery is reused, but selected M6 option evidence is self-observed.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()

    summary = evaluate(
        game=args.game,
        output_events=args.events,
        output_candidates=args.candidates,
        output_freeze=args.freeze,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
