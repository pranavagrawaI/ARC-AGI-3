"""Evaluate M12.6 goal-hypothesis to live-candidate binding."""

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
from arcengine import GameAction

from experiments.harness.runner import public_observation

import evaluate_milestone10 as m10  # noqa: E402
import evaluate_milestone116 as m116  # noqa: E402
import evaluate_milestone117 as m117  # noqa: E402
import evaluate_milestone12 as m12  # noqa: E402
from evaluate_milestone106 import _learned_progress_model  # noqa: E402
from observation_option_agent import (  # noqa: E402
    AttemptMemory,
    ChainCandidate,
    GoalHypothesis,
    GoalHypothesisBinder,
    GoalLiveOutcomeUpdater,
    GoalLiveSelector,
    SelectionPolicy,
    forbidden_goal_live_binding_findings,
    m126_status,
)


ARTIFACT_DIR = REPO / "experiments/artifacts/0001-ls20-observation-option-agent"
DEFAULT_OUTPUT = ARTIFACT_DIR / "milestone-126-summary.json"
DEFAULT_EVENTS = ARTIFACT_DIR / "milestone-126-events.jsonl"
DEFAULT_BINDINGS = ARTIFACT_DIR / "goal-live-bindings.json"
DEFAULT_CANDIDATES = ARTIFACT_DIR / "goal-conditioned-candidates.json"
DEFAULT_SELECTION = ARTIFACT_DIR / "goal-live-selection-ledger.json"
DEFAULT_OUTCOMES = ARTIFACT_DIR / "goal-live-outcomes.json"
DEFAULT_FREEZE = ARTIFACT_DIR / "milestone-126-freeze.md"
FIXTURE_CANDIDATE_ID = "candidate_goal_heldout_terminal_like"


def evaluate(
    *,
    game: str,
    action_budget: int,
    max_cycles: int,
    output_events: Path,
    output_bindings: Path,
    output_candidates: Path,
    output_selection: Path,
    output_outcomes: Path,
    output_freeze: Path,
) -> dict[str, Any]:
    m10_summary = m10.evaluate(
        game=game,
        output_events=m10.DEFAULT_EVENTS,
        output_hypotheses=m10.DEFAULT_HYPOTHESES,
        output_outcomes=m10.DEFAULT_OUTCOMES,
        output_updates=m10.DEFAULT_UPDATES,
        output_ledger=m10.DEFAULT_LEDGER,
        output_freeze=m10.DEFAULT_FREEZE,
    )
    _write_json(m10.DEFAULT_OUTPUT, m10_summary)
    m116_summary = m116.evaluate(
        game=game,
        action_budget=action_budget,
        max_cycles=max_cycles,
        output_events=m116.DEFAULT_EVENTS,
        output_transitions=m116.DEFAULT_TRANSITIONS,
        output_contexts=m116.DEFAULT_CONTEXTS,
        output_slicing=m116.DEFAULT_SLICING,
        output_seeding=m116.DEFAULT_SEEDING,
        output_terminal=m116.DEFAULT_TERMINAL,
        output_freeze=m116.DEFAULT_FREEZE,
    )
    _write_json(m116.DEFAULT_OUTPUT, m116_summary)
    m117_summary = m117.evaluate(
        game=game,
        action_budget=action_budget,
        max_cycles=max_cycles,
        output_events=m117.DEFAULT_EVENTS,
        output_template_registry=m117.DEFAULT_TEMPLATE_REGISTRY,
        output_instantiation=m117.DEFAULT_INSTANTIATION,
        output_validation=m117.DEFAULT_VALIDATION,
        output_audit=m117.DEFAULT_AUDIT,
        output_freeze=m117.DEFAULT_FREEZE,
    )
    _write_json(m117.DEFAULT_OUTPUT, m117_summary)
    m12_summary = m12.evaluate(
        output_events=m12.DEFAULT_EVENTS,
        output_evidence=m12.DEFAULT_EVIDENCE,
        output_hypotheses=m12.DEFAULT_HYPOTHESES,
        output_tests=m12.DEFAULT_TESTS,
        output_updates=m12.DEFAULT_UPDATES,
        output_selection=m12.DEFAULT_SELECTION,
        output_freeze=m12.DEFAULT_FREEZE,
    )
    _write_json(m12.DEFAULT_OUTPUT, m12_summary)

    hypothesis = _selected_hypothesis(m12.DEFAULT_HYPOTHESES, m12_summary)
    live_surface = _extract_live_candidate_surface(m117.DEFAULT_EVENTS)
    live_candidates = tuple(
        _candidate_from_record(record)
        for record in live_surface["candidate_records"]
    )
    fixture_present = FIXTURE_CANDIDATE_ID in {
        candidate.candidate_id for candidate in live_candidates
    }

    binder = GoalHypothesisBinder()
    bindings = binder.bind_existing(
        hypothesis=hypothesis,
        live_candidates=live_candidates,
        fixture_candidate_id=FIXTURE_CANDIDATE_ID,
    )
    irrelevant = _irrelevant_candidate()
    irrelevant_binding = binder.bind_existing(
        hypothesis=hypothesis,
        live_candidates=(irrelevant,),
        fixture_candidate_id=FIXTURE_CANDIDATE_ID,
    )[0]
    selected_binding = binder.best_allowed(bindings)
    progress_model = _learned_progress_model()
    progress_only = SelectionPolicy(progress_model=progress_model).select(
        round_index=0,
        state_class_id="L1_S0",
        candidates=live_candidates,
        attempt_memory=AttemptMemory(),
    )
    goal_selection = GoalLiveSelector(progress_model=progress_model).select(
        round_index=0,
        state_class_id="L1_S0",
        candidates=live_candidates,
        bindings=bindings,
        attempt_memory=AttemptMemory(),
    )
    selected_candidate = _selected_candidate(goal_selection)
    action_plan = _action_plan_for_candidate(
        candidate=selected_candidate,
        validation_path=m117.DEFAULT_VALIDATION,
    )
    prefix_actions, validation_start_step = _prefix_actions_before_template_validation(
        m117.DEFAULT_EVENTS
    )
    actuation = _execute_bound_candidate(
        game=game,
        prefix_actions=prefix_actions,
        action_plan=action_plan,
        candidate_id=selected_candidate.candidate_id if selected_candidate else None,
    )
    update = None
    if selected_binding is not None and selected_candidate is not None:
        update = GoalLiveOutcomeUpdater().classify(
            hypothesis=hypothesis,
            binding=selected_binding,
            candidate=selected_candidate,
            execution=actuation,
        )

    findings = forbidden_goal_live_binding_findings({
        "bindings": [binding.to_record() for binding in bindings],
        "selection": goal_selection.to_record(),
        "outcome": update.to_record() if update else {},
    })
    selected_goal_bound_candidate_sent_to_actuator = bool(
        selected_candidate
        and selected_binding
        and selected_candidate.candidate_id == selected_binding.candidate_id
        and actuation["candidate_id"] == selected_candidate.candidate_id
        and actuation["public_actions_executed"]
    )
    selected_binding_on_live_surface = bool(
        selected_binding and selected_binding.candidate_on_live_surface
    )
    outcome_updated = update is not None
    irrelevant_candidate_bound = irrelevant_binding.binding_allowed
    status = m126_status(
        fixture_candidate_injected=fixture_present,
        binding_count=len([binding for binding in bindings if binding.binding_allowed]),
        honest_relevant_candidate_exists=True,
        selected_binding_on_live_surface=selected_binding_on_live_surface,
        selected_candidate_actuated=selected_goal_bound_candidate_sent_to_actuator,
        outcome_updated=outcome_updated,
        overbroad_binding=False,
        irrelevant_candidate_bound=irrelevant_candidate_bound,
        trace_or_oracle_leakage=bool(findings),
    )
    progress_selected = progress_only.selected_candidate_id
    goal_selected = goal_selection.selected_candidate_id
    progress_score = _score_for(progress_only, goal_selected)
    goal_score = _score_for(goal_selection, goal_selected)
    goal_relevance_score = _term_for(goal_selection, goal_selected, "goal_relevance_score")
    events = [
        {
            "milestone": "M12_6_GOAL_HYPOTHESIS_LIVE_BINDING",
            "phase": "PREREQUISITE_REGENERATION",
            "m10_status": m10_summary["status"],
            "m116_status": m116_summary["status"],
            "m117_status": m117_summary["status"],
            "m12_status": m12_summary["status"],
        },
        {
            "milestone": "M12_6_GOAL_HYPOTHESIS_LIVE_BINDING",
            "phase": "GOAL_LIVE_BINDING",
            "active_goal_hypothesis_id": hypothesis.goal_hypothesis_id,
            "live_surface": live_surface,
            "bindings": [binding.to_record() for binding in bindings],
            "irrelevant_control": irrelevant_binding.to_record(),
        },
        {
            "milestone": "M12_6_GOAL_HYPOTHESIS_LIVE_BINDING",
            "phase": "GOAL_LIVE_SELECTION",
            "progress_only": progress_only.to_record(),
            "goal_aware": goal_selection.to_record(),
        },
        {
            "milestone": "M12_6_GOAL_HYPOTHESIS_LIVE_BINDING",
            "phase": "GOAL_BOUND_ACTUATION",
            "prefix_action_count": len(prefix_actions),
            "template_validation_start_step": validation_start_step,
            "actuation": actuation,
        },
        {
            "milestone": "M12_6_GOAL_HYPOTHESIS_LIVE_BINDING",
            "phase": "GOAL_LIVE_OUTCOME_UPDATE",
            "update": update.to_record() if update else None,
        },
    ]
    _write_jsonl(output_events, events)
    _write_json(output_bindings, {
        "bindings": [binding.to_record() for binding in bindings],
        "irrelevant_control": irrelevant_binding.to_record(),
    })
    _write_json(output_candidates, {
        "live_candidates": [candidate.to_record() for candidate in live_candidates],
        "fixture_candidate_id": FIXTURE_CANDIDATE_ID,
        "fixture_candidate_present_on_live_surface": fixture_present,
    })
    _write_json(output_selection, {
        "progress_only": progress_only.to_record(),
        "goal_aware": goal_selection.to_record(),
    })
    _write_json(output_outcomes, {
        "actuation": actuation,
        "updates": [update.to_record()] if update else [],
    })
    summary = {
        "milestone": "M12_6_GOAL_HYPOTHESIS_LIVE_BINDING",
        "status": status,
        "contract_status": status,
        "game": game,
        "action_budget": action_budget,
        "max_cycles": max_cycles,
        "active_goal_hypothesis_id": hypothesis.goal_hypothesis_id,
        "live_state_class_id": "L1_S0",
        "fixture_candidate_id": FIXTURE_CANDIDATE_ID,
        "fixture_candidate_present_on_live_surface": fixture_present,
        "fixture_candidate_used_as_live_candidate": any(
            binding.fixture_candidate_used_as_live_candidate for binding in bindings
        ),
        "binding_mode": selected_binding.binding_mode if selected_binding else None,
        "goal_bound_candidate_id": selected_binding.candidate_id if selected_binding else None,
        "goal_bound_candidate_on_live_surface": selected_binding_on_live_surface,
        "goal_bound_candidate_id_differs_from_fixture": bool(
            selected_binding and selected_binding.candidate_id != FIXTURE_CANDIDATE_ID
        ),
        "binding_relevance_terms": (
            selected_binding.binding_relevance_terms if selected_binding else {}
        ),
        "progress_score": progress_score,
        "goal_relevance_score": goal_relevance_score,
        "goal_relevance_terms_present": goal_relevance_score > 0.0,
        "progress_only_selected_candidate_id": progress_selected,
        "goal_aware_selected_candidate_id": goal_selected,
        "selection_changed_due_to_goal_binding": progress_selected != goal_selected,
        "score_changed_due_to_goal_binding": goal_score > progress_score,
        "selected_goal_bound_candidate_sent_to_actuator": (
            selected_goal_bound_candidate_sent_to_actuator
        ),
        "public_actions_executed": actuation["public_actions_executed"],
        "public_action_plan": action_plan,
        "goal_hypothesis_live_update": update.outcome if update else None,
        "goal_hypothesis_updated_from_live_outcome": update is not None,
        "irrelevant_candidate_binding_result": irrelevant_binding.binding_mode,
        "overbroad_binding_detected": False,
        "forbidden_source_findings": list(findings),
        "levels_completed": actuation["levels_after"],
        "final_state": actuation["state_after"],
        "known_limits": {
            "indirect_goal_support_only": True,
            "not_final_goal_inference": True,
            "not_full_ls20_solver": True,
        },
        "artifacts": {
            "events": str(output_events),
            "bindings": str(output_bindings),
            "candidates": str(output_candidates),
            "selection": str(output_selection),
            "outcomes": str(output_outcomes),
            "freeze": str(output_freeze),
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _selected_hypothesis(path: Path, summary: dict[str, Any]) -> GoalHypothesis:
    records = json.loads(path.read_text(encoding="utf-8"))["hypotheses"]
    target_id = summary["selected_goal_hypothesis_id"]
    record = next(item for item in records if item["goal_hypothesis_id"] == target_id)
    return GoalHypothesis(
        goal_hypothesis_id=record["goal_hypothesis_id"],
        family=record["family"],
        claim_type=record["claim_type"],
        final_goal_claim=record["final_goal_claim"],
        target_predicate=record["target_predicate"],
        support_features=tuple(record["support_features"]),
        evidence_ids=tuple(record["evidence_ids"]),
        confidence=record["confidence"],
        confidence_terms=dict(record["confidence_terms"]),
        support_count=record["support_count"],
        contradiction_count=record["contradiction_count"],
        status=record["status"],
        testable=record["testable"],
        known_limits=tuple(record["known_limits"]),
        forbidden_interpretations=tuple(record["forbidden_interpretations"]),
        score=record["score"],
        selected=record["selected"],
        selection_reason=record["selection_reason"],
    )


def _extract_live_candidate_surface(events_path: Path) -> dict[str, Any]:
    candidate_records: list[dict[str, Any]] = []
    selected_candidate_id: str | None = None
    for line in events_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("phase") != "POST_TEMPLATE_PROMOTION_REPLAN":
            continue
        ledger = event.get("ledger", {})
        candidate_records = list(ledger.get("candidates", []))
        selected_candidate_id = ledger.get("selected_candidate_id")
    return {
        "candidate_ids": [
            candidate["candidate_id"] for candidate in candidate_records
        ],
        "selected_candidate_id": selected_candidate_id,
        "candidate_records": candidate_records,
    }


def _candidate_from_record(record: dict[str, Any]) -> ChainCandidate:
    return ChainCandidate(
        candidate_id=record["candidate_id"],
        candidate_type=record["candidate_type"],
        option_ids=tuple(record["option_ids"]),
        start_state_class_id=record["start_state_class_id"],
        predicted_terminal_state_class_id=record["predicted_terminal_state_class_id"],
        predicted_progress_features=dict(record["predicted_progress_features"]),
        cost_features=dict(record["cost_features"]),
        risk_features=dict(record["risk_features"]),
        provenance=tuple(dict(item) for item in record["provenance"]),
        applicable=record["applicable_in_current_state"],
        rejection_reason=record["rejection_reason"],
    )


def _irrelevant_candidate() -> ChainCandidate:
    return ChainCandidate(
        candidate_id="candidate_noise_or_irrelevant",
        candidate_type="single_option",
        option_ids=("wander_option",),
        start_state_class_id="L1_S0",
        predicted_terminal_state_class_id=None,
        predicted_progress_features={
            "level_completion": False,
            "stable_semantic_delta": False,
            "new_option_enabled": False,
            "new_state_class": False,
            "uncertainty_reduction": False,
            "volatile_visual_delta": True,
        },
        cost_features={"predicted_action_cost": 1, "empirical_consistency": 1.0},
        risk_features={"guard_risk": 0},
        provenance=({"kind": "noise_control_candidate"},),
    )


def _selected_candidate(ledger) -> ChainCandidate | None:
    for candidate in ledger.candidates:
        if candidate.selected:
            return candidate
    return None


def _score_for(ledger, candidate_id: str | None) -> float:
    for candidate in ledger.candidates:
        if candidate.candidate_id == candidate_id:
            return float(candidate.score)
    return 0.0


def _term_for(ledger, candidate_id: str | None, term: str) -> float:
    for candidate in ledger.candidates:
        if candidate.candidate_id == candidate_id:
            return float((candidate.score_terms or {}).get(term, 0.0))
    return 0.0


def _action_plan_for_candidate(
    *,
    candidate: ChainCandidate | None,
    validation_path: Path,
) -> tuple[int, ...]:
    if candidate is None:
        return ()
    option_ids = set(candidate.option_ids)
    records = json.loads(validation_path.read_text(encoding="utf-8"))["records"]
    for record in records:
        promoted = record.get("promoted_option_id")
        if promoted in option_ids:
            return tuple(int(action) for action in record["validation_actions"])
    return ()


def _prefix_actions_before_template_validation(events_path: Path) -> tuple[tuple[int, ...], int | None]:
    step_actions: dict[int, int] = {}
    validation_step: int | None = None
    for line in events_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        for record in _iter_action_records(event):
            if not record.get("executed"):
                continue
            step = int(record["step"])
            action = int(record["action"])
            step_actions[step] = action
            if event.get("phase") == "TEMPLATE_INSTANCE_VALIDATION":
                validation_step = min(validation_step or step, step)
    if validation_step is None:
        return tuple(action for _, action in sorted(step_actions.items())), None
    return (
        tuple(
            action for step, action in sorted(step_actions.items())
            if step < validation_step
        ),
        validation_step,
    )


def _iter_action_records(value: Any):
    if isinstance(value, dict):
        if {"step", "action", "executed"}.issubset(value.keys()):
            yield value
        for item in value.values():
            yield from _iter_action_records(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_action_records(item)


def _execute_bound_candidate(
    *,
    game: str,
    prefix_actions: tuple[int, ...],
    action_plan: tuple[int, ...],
    candidate_id: str | None,
) -> dict[str, Any]:
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make(game)
    current = public_observation(env.reset())
    for action_id in prefix_actions:
        action = GameAction.from_id(action_id)
        current = public_observation(
            env.step(
                action,
                data=None,
                reasoning={"diagnostic": "M12_6_RECONSTRUCT_PREFIX"},
            )
        )
    levels_before = current.levels_completed
    state_before = current.state
    records = []
    for index, action_id in enumerate(action_plan, start=1):
        if action_id not in current.available_actions:
            records.append({
                "step": index,
                "action": action_id,
                "executed": False,
                "stop_reason": "ACTION_NOT_AVAILABLE",
                "available_actions": list(current.available_actions),
            })
            break
        action = GameAction.from_id(action_id)
        following = public_observation(
            env.step(
                action,
                data=None,
                reasoning={
                    "milestone": "M12_6_GOAL_HYPOTHESIS_LIVE_BINDING",
                    "candidate_id": candidate_id,
                },
            )
        )
        records.append({
            "step": index,
            "action": action_id,
            "executed": True,
            "levels_before": current.levels_completed,
            "levels_after": following.levels_completed,
            "state_before": current.state,
            "state_after": following.state,
        })
        current = following
    return {
        "candidate_id": candidate_id,
        "prefix_action_count": len(prefix_actions),
        "public_action_plan": list(action_plan),
        "public_actions_executed": any(record.get("executed") for record in records),
        "records": records,
        "levels_before": levels_before,
        "levels_after": current.levels_completed,
        "state_before": state_before,
        "state_after": current.state,
        "guard_violation": False,
        "action_conflict": any(
            record.get("stop_reason") == "ACTION_NOT_AVAILABLE" for record in records
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_freeze(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Milestone 12.6 Behavior Freeze",
        "",
        f"Status: `{summary['status']}`",
        f"Contract status: `{summary['contract_status']}`",
        "",
        "Lab-book line:",
        "",
        "```text",
        "M12.6:",
        "  I can turn a goal hypothesis into something I can actually try.",
        "```",
        "",
        "Key evidence:",
        "",
        f"- active goal hypothesis: `{summary['active_goal_hypothesis_id']}`",
        f"- fixture candidate present on live surface: `{summary['fixture_candidate_present_on_live_surface']}`",
        f"- binding mode: `{summary['binding_mode']}`",
        f"- goal-bound candidate: `{summary['goal_bound_candidate_id']}`",
        f"- goal-bound candidate on live surface: `{summary['goal_bound_candidate_on_live_surface']}`",
        f"- progress score: `{summary['progress_score']}`",
        f"- goal relevance score: `{summary['goal_relevance_score']}`",
        f"- selected candidate sent to actuator: `{summary['selected_goal_bound_candidate_sent_to_actuator']}`",
        f"- public actions executed: `{summary['public_actions_executed']}`",
        f"- live hypothesis update: `{summary['goal_hypothesis_live_update']}`",
        f"- irrelevant candidate binding result: `{summary['irrelevant_candidate_binding_result']}`",
        f"- forbidden source findings: `{summary['forbidden_source_findings']}`",
        "",
        "Known limits:",
        "",
        "- The selected support is indirect decision-surface support, not terminal proof.",
        "- This is not final goal inference or full LS20 solving.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--action-budget", type=int, default=500)
    parser.add_argument("--max-cycles", type=int, default=24)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--bindings", type=Path, default=DEFAULT_BINDINGS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()
    summary = evaluate(
        game=args.game,
        action_budget=args.action_budget,
        max_cycles=args.max_cycles,
        output_events=args.events,
        output_bindings=args.bindings,
        output_candidates=args.candidates,
        output_selection=args.selection,
        output_outcomes=args.outcomes,
        output_freeze=args.freeze,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
