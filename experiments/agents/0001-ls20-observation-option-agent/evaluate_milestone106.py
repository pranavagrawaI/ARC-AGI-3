"""Evaluate M10.6 live option-state binding and replanning."""

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

import arc_agi
from arc_agi.base import OperationMode

from experiments.harness.runner import public_observation

from evaluate_milestone10 import _m10_candidate  # noqa: E402
from evaluate_milestone9 import _build_graph_context  # noqa: E402
from observation_option_agent import (  # noqa: E402
    AttemptMemory,
    LearnedProgressModel,
    LiveCandidateActuator,
    LiveCandidateGenerator,
    LiveObservationStateClassifier,
    ProgressHypothesisSet,
    SelectionPolicy,
    forbidden_live_replanning_findings,
    m106_status,
    observation_hash,
    outcome_record_from_candidate,
)


ARTIFACT_DIR = REPO / "experiments/artifacts/0001-ls20-observation-option-agent"
DEFAULT_OUTPUT = ARTIFACT_DIR / "milestone-106-summary.json"
DEFAULT_EVENTS = ARTIFACT_DIR / "milestone-106-events.jsonl"
DEFAULT_CLASSIFICATIONS = ARTIFACT_DIR / "live-state-classifications.json"
DEFAULT_REPLANNING = ARTIFACT_DIR / "live-replanning-ledger.json"
DEFAULT_ACTUATION = ARTIFACT_DIR / "live-candidate-actuation.json"
DEFAULT_FREEZE = ARTIFACT_DIR / "milestone-106-freeze.md"


def evaluate(
    *,
    game: str,
    action_budget: int,
    max_replan_cycles: int,
    output_events: Path,
    output_classifications: Path,
    output_replanning: Path,
    output_actuation: Path,
    output_freeze: Path,
) -> dict[str, Any]:
    graph_context = _build_graph_context(game=game)
    graph = graph_context["graph"]
    option_a = graph_context["option_a"]
    option_b = graph_context["option_b"]
    options = (option_a, option_b)
    action_plans = {
        option_a.option_id: tuple(graph_context["heldout_a"].action_plan),
        option_b.option_id: tuple(graph_context["heldout_b"].action_plan),
    }

    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make(game)
    current = public_observation(env.reset())
    current_position = tuple(
        int(value) for value in option_a.initiation_condition["actor_position"]
    )
    current_position_source = "option_state_binding"
    classifier = LiveObservationStateClassifier()
    generator = LiveCandidateGenerator()
    actuator = LiveCandidateActuator()
    actuator.start(current)
    selection_policy = SelectionPolicy(
        progress_model=_learned_progress_model()
    )
    attempt_memory = AttemptMemory()

    classifications = []
    selections = []
    actuations = []
    events = []
    stop_reason = None
    actions_used = 0
    cycles_completed = 0
    candidate_generation_from_live_state = False
    learned_progress_used = False
    selected_candidate_sent_to_actuator = False

    for cycle in range(max_replan_cycles):
        binding = _classify_and_record(
            cycle=cycle,
            classifier=classifier,
            graph=graph,
            options=options,
            current=current,
            current_position=current_position,
            current_position_source=current_position_source,
            classifications=classifications,
            events=events,
        )
        if current.state in {"WIN", "GAME_OVER"}:
            stop_reason = "TERMINAL_STATE"
            break
        if not binding.replannable:
            stop_reason = binding.classification
            break

        candidates = generator.generate(
            graph=graph,
            binding=binding,
            options=options,
            action_plans=action_plans,
            max_depth=2,
        )
        candidate_generation_from_live_state = True
        if not candidates:
            stop_reason = "NO_APPLICABLE_LIVE_CANDIDATES"
            break
        selection = selection_policy.select(
            round_index=cycle,
            state_class_id=binding.state_class_id or "",
            candidates=candidates,
            attempt_memory=attempt_memory,
        )
        learned_progress_used = True
        selections.append(selection.to_record())
        events.append({
            "milestone": "M10_6_LIVE_OPTION_STATE_BINDING",
            "phase": "LIVE_REPLAN_SELECTION",
            "cycle": cycle,
            "used_learned_progress_weights": True,
            "ledger": selection.to_record(),
        })
        selected = _selected_candidate(selection)
        if selected is None:
            stop_reason = "NO_APPLICABLE_LIVE_CANDIDATES"
            break
        remaining = action_budget - actions_used
        if remaining <= 0:
            stop_reason = "ACTION_BUDGET_EXHAUSTED"
            break
        result, current = actuator.execute_first_link(
            env=env,
            current_observation=current,
            candidate=selected,
            action_plans=action_plans,
            current_controlled_position=current_position,
            step_offset=actions_used,
            remaining_budget=remaining,
        )
        selected_candidate_sent_to_actuator = (
            selected_candidate_sent_to_actuator
            or result.selected_candidate_sent_to_actuator
        )
        actions_used += result.actions_executed
        current_position = result.controlled_position
        current_position_source = result.controlled_position_source
        cycles_completed += 1 if result.actions_executed else 0
        actuations.append({
            "cycle": cycle,
            **result.to_record(),
        })
        events.append({
            "milestone": "M10_6_LIVE_OPTION_STATE_BINDING",
            "phase": "LIVE_ACTUATION",
            "cycle": cycle,
            "actuation": result.to_record(),
        })
        if result.stop_reason is not None:
            stop_reason = result.stop_reason
            break
        if actions_used >= action_budget:
            stop_reason = "ACTION_BUDGET_EXHAUSTED"
            break
    if stop_reason is None:
        _classify_and_record(
            cycle=len(classifications),
            classifier=classifier,
            graph=graph,
            options=options,
            current=current,
            current_position=current_position,
            current_position_source=current_position_source,
            classifications=classifications,
            events=events,
        )
    if stop_reason is None:
        stop_reason = "ACTION_BUDGET_EXHAUSTED" if actions_used >= action_budget else "MAX_REPLAN_CYCLES"

    valid_stop_reason = stop_reason in {
        "NO_APPLICABLE_LIVE_CANDIDATES",
        "UNKNOWN_STATE_CLASS",
        "AMBIGUOUS_STATE_CLASS",
        "SEMANTIC_GUARD_VIOLATION",
        "CONTROLLED_ENTITY_LOST",
        "ACTION_NOT_AVAILABLE",
        "ACTION_BUDGET_EXHAUSTED",
        "TERMINAL_STATE",
        "MAX_REPLAN_CYCLES",
    }
    post_execution_classified = (
        len(classifications) >= 2
        and any(item["cycle"] > 0 for item in classifications)
    )
    classified_failures = {
        "UNKNOWN_STATE_CLASS",
        "AMBIGUOUS_STATE_CLASS",
        "SEMANTIC_GUARD_VIOLATION",
        "CONTROLLED_ENTITY_LOST",
    }
    live_unclassified = cycles_completed == 0 and any(
        item["classification"] in classified_failures
        for item in classifications
    )
    expected_candidate_missing = any(
        item["classification"] == "MATCHED_KNOWN_STATE_CLASS"
        and item["applicable_options"]
        for item in classifications
    ) and not candidate_generation_from_live_state
    selected_not_executed = any(
        selection.get("selected_candidate_id")
        for selection in selections
    ) and not selected_candidate_sent_to_actuator

    audit_payload = {
        "classifications": classifications,
        "selections": selections,
        "actuations": actuations,
    }
    findings = forbidden_live_replanning_findings(audit_payload)
    status = m106_status(
        classification_count=len(classifications),
        post_execution_classified=post_execution_classified,
        candidate_generation_from_live_state=candidate_generation_from_live_state,
        learned_progress_used=learned_progress_used,
        candidate_sent_to_actuator=selected_candidate_sent_to_actuator,
        actions_executed=actions_used,
        replan_cycles_completed=cycles_completed,
        valid_stop_reason=valid_stop_reason,
        counterfactual_only=False,
        selected_candidate_not_executed=selected_not_executed,
        trace_leakage=bool(findings),
        oracle_signal=False,
        live_observation_unclassified=live_unclassified,
        no_live_candidate_generation=expected_candidate_missing,
    )

    _write_jsonl(output_events, events)
    _write_json(output_classifications, {"classifications": classifications})
    _write_json(output_replanning, {"selection_rounds": selections})
    _write_json(output_actuation, {"actuations": actuations})
    summary = {
        "milestone": "M10_6_LIVE_OPTION_STATE_BINDING_AND_REPLANNING",
        "status": status,
        "contract_status": status,
        "game": game,
        "action_budget": action_budget,
        "max_replan_cycles": max_replan_cycles,
        "live_actions_executed": actions_used,
        "replan_cycles_attempted": len(selections),
        "replan_cycles_completed": cycles_completed,
        "post_execution_state_classified": post_execution_classified,
        "final_state_classification_result": classifications[-1]["classification"],
        "final_state_class_id": classifications[-1]["state_class_id"],
        "controlled_position_source": current_position_source,
        "learned_progress_used_for_live_selection": learned_progress_used,
        "selected_candidate_sent_to_actuator": selected_candidate_sent_to_actuator,
        "counterfactual_fixture_selection_only": False,
        "live_candidate_generation_from_classified_state": candidate_generation_from_live_state,
        "stop_reason": stop_reason,
        "levels_completed": current.levels_completed,
        "final_state": current.state,
        "forbidden_source_findings": list(findings),
        "known_limits": {
            "bounded_live_replanning_loop": True,
            "not_final_objective_inference": True,
            "not_full_ls20_solver": True,
            "option_graph_remains_small": True,
        },
        "artifacts": {
            "events": str(output_events),
            "classifications": str(output_classifications),
            "replanning": str(output_replanning),
            "actuation": str(output_actuation),
            "freeze": str(output_freeze),
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _classify_and_record(
    *,
    cycle: int,
    classifier: LiveObservationStateClassifier,
    graph,
    options,
    current,
    current_position,
    current_position_source,
    classifications,
    events,
):
    binding = classifier.classify(
        observation_hash=observation_hash(current),
        graph=graph,
        options=options,
        controlled_position=current_position,
        controlled_position_source=current_position_source,
        semantic_guards_valid=True,
    )
    classifications.append({
        "cycle": cycle,
        **binding.to_record(),
    })
    events.append({
        "milestone": "M10_6_LIVE_OPTION_STATE_BINDING",
        "phase": "LIVE_STATE_CLASSIFICATION",
        "cycle": cycle,
        "binding": binding.to_record(),
    })
    return binding


def _learned_progress_model() -> LearnedProgressModel:
    priors = ProgressHypothesisSet.from_m9_priors()
    positive = _m10_candidate(
        candidate_id="m106_positive_feature_source",
        new_option_enabled=True,
        new_state_class=True,
        action_cost=6,
    )
    noise = _m10_candidate(
        candidate_id="m106_noise_feature_source",
        volatile_visual_delta=True,
        action_cost=1,
    )
    positive_outcome = outcome_record_from_candidate(
        outcome_id="m106_positive_progress_outcome",
        candidate=positive,
        observed_features={
            "level_completed": False,
            "new_option_enabled": True,
            "new_state_class": True,
            "prediction_validated": True,
            "stable_semantic_delta": False,
            "guard_violation": False,
            "action_conflict_or_next_not_applicable": False,
        },
        stop_reason="PUBLIC_PROGRESS_CONFIRMED",
        provenance=({"kind": "m106_progress_evidence_fixture"},),
    )
    noise_outcome = outcome_record_from_candidate(
        outcome_id="m106_noise_outcome",
        candidate=noise,
        observed_features={
            "level_completed": False,
            "new_option_enabled": False,
            "new_state_class": False,
            "prediction_validated": False,
            "stable_semantic_delta": False,
            "guard_violation": False,
            "action_conflict_or_next_not_applicable": True,
        },
        stop_reason="NO_STABLE_PROGRESS",
        provenance=({"kind": "m106_noise_suppression_fixture"},),
    )
    learned, _updates = priors.update_from_outcomes((
        positive_outcome,
        noise_outcome,
    ))
    return LearnedProgressModel(learned)


def _selected_candidate(selection) -> Any | None:
    for candidate in selection.candidates:
        if candidate.selected:
            return candidate
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_freeze(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Milestone 10.6 Behavior Freeze",
        "",
        f"Status: `{summary['status']}`",
        f"Contract status: `{summary['contract_status']}`",
        "",
        "Lab-book line:",
        "",
        "```text",
        "M10.6:",
        "  I can act, re-locate myself in my option graph, and choose again.",
        "```",
        "",
        "Key evidence:",
        "",
        f"- live actions executed: `{summary['live_actions_executed']}`",
        f"- replan cycles completed: `{summary['replan_cycles_completed']}`",
        f"- post-execution state classified: `{summary['post_execution_state_classified']}`",
        f"- final classification: `{summary['final_state_classification_result']}`",
        f"- final state class: `{summary['final_state_class_id']}`",
        f"- controlled position source: `{summary['controlled_position_source']}`",
        f"- learned progress used: `{summary['learned_progress_used_for_live_selection']}`",
        f"- candidate sent to actuator: `{summary['selected_candidate_sent_to_actuator']}`",
        f"- live candidate generation from classified state: `{summary['live_candidate_generation_from_classified_state']}`",
        f"- stop reason: `{summary['stop_reason']}`",
        f"- levels completed: `{summary['levels_completed']}`",
        f"- forbidden source findings: `{summary['forbidden_source_findings']}`",
        "",
        "Known limits:",
        "",
        "- This is a bounded live replanning loop over the existing small option graph.",
        "- This is not final objective inference.",
        "- This is not a full LS20 solver.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--action-budget", type=int, default=200)
    parser.add_argument("--max-replan-cycles", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--classifications", type=Path, default=DEFAULT_CLASSIFICATIONS)
    parser.add_argument("--replanning", type=Path, default=DEFAULT_REPLANNING)
    parser.add_argument("--actuation", type=Path, default=DEFAULT_ACTUATION)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()

    summary = evaluate(
        game=args.game,
        action_budget=args.action_budget,
        max_replan_cycles=args.max_replan_cycles,
        output_events=args.events,
        output_classifications=args.classifications,
        output_replanning=args.replanning,
        output_actuation=args.actuation,
        output_freeze=args.freeze,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
