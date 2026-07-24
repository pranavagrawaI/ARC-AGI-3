"""Run the M10.5 live-unleash diagnostic.

This is deliberately a diagnostic, not a new capability milestone. It checks how
far the current M6-M10 stack can be driven against the public LS20 environment
before an integration boundary stops it.
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

import arc_agi
from arc_agi.base import OperationMode
from arcengine import GameAction

from experiments.harness.runner import public_observation

from evaluate_milestone10 import _m10_candidate  # noqa: E402
from evaluate_milestone9 import _build_graph_context  # noqa: E402
from observation_option_agent import (  # noqa: E402
    AttemptMemory,
    LearnedProgressModel,
    ProgressHypothesisSet,
    SelectionPolicy,
    generate_chain_candidates,
    outcome_record_from_candidate,
)


ARTIFACT_DIR = REPO / "experiments/artifacts/0001-ls20-observation-option-agent"
DEFAULT_OUTPUT = ARTIFACT_DIR / "milestone-105-summary.json"
DEFAULT_EVENTS = ARTIFACT_DIR / "milestone-105-events.jsonl"
DEFAULT_LEDGER = ARTIFACT_DIR / "milestone-105-diagnostic-ledger.json"
DEFAULT_FREEZE = ARTIFACT_DIR / "milestone-105-freeze.md"
DEFAULT_BUDGET = 200


def evaluate(
    *,
    game: str,
    budget: int,
    output_events: Path,
    output_ledger: Path,
    output_freeze: Path,
) -> dict[str, Any]:
    graph_context = _build_graph_context(game=game)
    state_0 = graph_context["state_0"]
    candidates = generate_chain_candidates(
        graph=graph_context["graph"],
        start_state_class_id=state_0.state_class_id,
        option_a=graph_context["option_a"],
        option_b=graph_context["option_b"],
    )
    m9_selection = SelectionPolicy().select(
        round_index=0,
        state_class_id=state_0.state_class_id,
        candidates=candidates,
        attempt_memory=AttemptMemory(),
    )
    live_candidate = _selected_candidate(m9_selection)
    live_plan = _plan_for_candidate(
        candidate_id=live_candidate.candidate_id,
        graph_context=graph_context,
    )

    live_records, live_final = _execute_public_plan(
        game=game,
        plan=live_plan[:budget],
    )

    learned_selection = _m10_learned_selection(state_class_id=state_0.state_class_id)
    integration_bottlenecks = _integration_bottlenecks(
        live_plan=live_plan,
        live_records=live_records,
        learned_selection_id=learned_selection.selected_candidate_id,
    )
    levels_completed = int(live_final["levels_completed"])
    status = (
        "diagnostic_reached_public_completion"
        if levels_completed > 0
        else "diagnostic_complete_with_bottleneck"
    )

    events = [
        {
            "diagnostic": "M10_5_LIVE_UNLEASH",
            "phase": "M9_EXECUTABLE_SELECTION",
            "ledger": m9_selection.to_record(),
        },
        {
            "diagnostic": "M10_5_LIVE_UNLEASH",
            "phase": "M10_LEARNED_PROGRESS_SELECTION",
            "ledger": learned_selection.to_record(),
            "live_executable": False,
            "reason": "M10 held-out progress candidates are diagnostic fixtures, not live option graph candidates.",
        },
        {
            "diagnostic": "M10_5_LIVE_UNLEASH",
            "phase": "LIVE_PUBLIC_PLAN_ATTEMPT",
            "candidate_id": live_candidate.candidate_id,
            "plan": list(live_plan),
            "records": live_records,
        },
        {
            "diagnostic": "M10_5_LIVE_UNLEASH",
            "phase": "BOTTLENECK_CLASSIFICATION",
            "bottlenecks": integration_bottlenecks,
        },
    ]
    _write_jsonl(output_events, events)

    ledger = {
        "candidate_count": len(candidates),
        "executable_candidate_ids": [candidate.candidate_id for candidate in candidates],
        "m9_selected_executable_candidate_id": live_candidate.candidate_id,
        "m9_selected_executable_plan": list(live_plan),
        "m10_learned_selected_candidate_id": learned_selection.selected_candidate_id,
        "m10_learned_candidate_is_live_executable": (
            learned_selection.selected_candidate_id
            in {candidate.candidate_id for candidate in candidates}
        ),
        "live_public_records": live_records,
        "integration_bottlenecks": integration_bottlenecks,
    }
    output_ledger.parent.mkdir(parents=True, exist_ok=True)
    output_ledger.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "diagnostic": "M10_5_LIVE_UNLEASH",
        "status": status,
        "game": game,
        "budget": budget,
        "mode": "diagnostic_not_capability_claim",
        "live_public_environment_used": True,
        "live_public_actions_attempted": len(live_records),
        "live_executable_selected_candidate_id": live_candidate.candidate_id,
        "live_executable_plan": list(live_plan),
        "m10_learned_selected_candidate_id": learned_selection.selected_candidate_id,
        "m10_learned_selection_live_executable": ledger[
            "m10_learned_candidate_is_live_executable"
        ],
        "final_state": live_final["state"],
        "levels_completed": levels_completed,
        "first_level_completed": levels_completed >= 1,
        "unused_action_budget": max(0, budget - len(live_records)),
        "integration_bottlenecks": integration_bottlenecks,
        "interpretation": _interpretation(
            levels_completed=levels_completed,
            bottlenecks=integration_bottlenecks,
        ),
        "known_limits": {
            "not_new_milestone": True,
            "m10_progress_learning_not_integrated_into_live_candidate_generator": True,
            "no_live_observation_to_option_graph_state_classifier": True,
            "derived_continuation_option_not_discovered_live_in_same_rollout": True,
            "not_full_ls20_solver": True,
        },
        "artifacts": {
            "events": str(output_events),
            "diagnostic_ledger": str(output_ledger),
            "freeze": str(output_freeze),
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _selected_candidate(ledger) -> Any:
    for candidate in ledger.candidates:
        if candidate.selected:
            return candidate
    raise RuntimeError("no selected candidate in diagnostic ledger")


def _plan_for_candidate(
    *,
    candidate_id: str,
    graph_context: dict[str, Any],
) -> tuple[int, ...]:
    plans = {
        "option_000": tuple(graph_context["heldout_a"].action_plan),
        "option_001": tuple(graph_context["heldout_b"].action_plan),
        "chain_000": (
            tuple(graph_context["heldout_a"].action_plan)
            + tuple(graph_context["heldout_b"].action_plan)
        ),
    }
    return plans[candidate_id]


def _execute_public_plan(
    *,
    game: str,
    plan: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make(game)
    current = public_observation(env.reset())
    records: list[dict[str, Any]] = []
    for index, action_id in enumerate(plan, start=1):
        if current.state in {"WIN", "GAME_OVER"}:
            break
        if action_id not in current.available_actions:
            records.append({
                "step": index,
                "action": action_id,
                "phase": "LIVE_PUBLIC_PLAN_ATTEMPT",
                "executed": False,
                "stop_reason": "ACTION_NOT_AVAILABLE",
                "available_actions": list(current.available_actions),
                "levels_before": current.levels_completed,
                "state_before": current.state,
            })
            break
        action = GameAction.from_id(action_id)
        if action.is_complex():
            records.append({
                "step": index,
                "action": action_id,
                "phase": "LIVE_PUBLIC_PLAN_ATTEMPT",
                "executed": False,
                "stop_reason": "COMPLEX_ACTION_NOT_SUPPORTED_BY_DIAGNOSTIC",
                "levels_before": current.levels_completed,
                "state_before": current.state,
            })
            break
        following = public_observation(
            env.step(
                action,
                data=None,
                reasoning={
                    "diagnostic": "M10_5_LIVE_UNLEASH",
                    "source": "current_option_stack_plan",
                },
            )
        )
        records.append({
            "step": index,
            "action": action_id,
            "phase": "LIVE_PUBLIC_PLAN_ATTEMPT",
            "executed": True,
            "levels_before": current.levels_completed,
            "levels_after": following.levels_completed,
            "state_before": current.state,
            "state_after": following.state,
            "available_actions_before": list(current.available_actions),
            "available_actions_after": list(following.available_actions),
        })
        current = following
    return records, {
        "state": current.state,
        "levels_completed": current.levels_completed,
        "available_actions": list(current.available_actions),
    }


def _m10_learned_selection(*, state_class_id: str):
    priors = ProgressHypothesisSet.from_m9_priors()
    positive = _m10_candidate(
        candidate_id="observed_positive_feature_source",
        new_option_enabled=True,
        new_state_class=True,
        action_cost=6,
    )
    noise = _m10_candidate(
        candidate_id="observed_noise_feature_source",
        volatile_visual_delta=True,
        action_cost=1,
    )
    positive_outcome = outcome_record_from_candidate(
        outcome_id="m105_positive_progress_outcome",
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
        provenance=({"kind": "m105_current_stack_progress_fixture"},),
    )
    noise_outcome = outcome_record_from_candidate(
        outcome_id="m105_noise_outcome",
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
        provenance=({"kind": "m105_noise_suppression_fixture"},),
    )
    learned, _updates = priors.update_from_outcomes((positive_outcome, noise_outcome))
    candidates = (
        _m10_candidate(
            candidate_id="candidate_stable_prior",
            stable_semantic_delta=True,
            action_cost=5,
        ),
        _m10_candidate(
            candidate_id="heldout_positive_feature",
            new_option_enabled=True,
            new_state_class=True,
            action_cost=4,
        ),
        _m10_candidate(
            candidate_id="heldout_noise_feature",
            volatile_visual_delta=True,
            action_cost=1,
        ),
    )
    return SelectionPolicy(progress_model=LearnedProgressModel(learned)).select(
        round_index=1,
        state_class_id=state_class_id,
        candidates=candidates,
        attempt_memory=AttemptMemory(),
    )


def _integration_bottlenecks(
    *,
    live_plan: tuple[int, ...],
    live_records: list[dict[str, Any]],
    learned_selection_id: str | None,
) -> list[str]:
    bottlenecks = [
        "NO_LIVE_OBSERVATION_TO_OPTION_GRAPH_STATE_CLASSIFIER",
        "NO_LIVE_REPLANNER_AFTER_SELECTED_CHAIN_PLAN",
        "M10_LEARNED_PROGRESS_CANDIDATES_NOT_WIRED_TO_LIVE_OPTION_ACTUATOR",
    ]
    if learned_selection_id and learned_selection_id.startswith("heldout_"):
        bottlenecks.append("LEARNED_SELECTION_IS_COUNTERFACTUAL_FIXTURE")
    if len(live_records) < len(live_plan):
        bottlenecks.append("LIVE_PLAN_STOPPED_BEFORE_SYNTHETIC_PLAN_EXHAUSTED")
    return bottlenecks


def _interpretation(
    *,
    levels_completed: int,
    bottlenecks: list[str],
) -> str:
    if levels_completed > 0:
        return (
            "The current option stack reached public completion during the "
            "diagnostic, but the run still lacks an integrated live replanner."
        )
    return (
        "The current stack can select and execute a short public action plan, "
        "but it does not yet have the live state binding needed to continue "
        "M10 policy selection from fresh LS20 observations. First bottleneck: "
        f"{bottlenecks[0]}."
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
        "# M10.5 Live-Unleash Diagnostic",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Question:",
        "",
        (
            "What happens if the current M6-M10 option machinery is driven "
            "against the public LS20 environment?"
        ),
        "",
        "Result:",
        "",
        f"- live public actions attempted: `{summary['live_public_actions_attempted']}`",
        f"- selected live-executable candidate: `{summary['live_executable_selected_candidate_id']}`",
        f"- live plan: `{summary['live_executable_plan']}`",
        f"- final state: `{summary['final_state']}`",
        f"- levels completed: `{summary['levels_completed']}`",
        f"- M10 learned selection: `{summary['m10_learned_selected_candidate_id']}`",
        f"- M10 selection live-executable: `{summary['m10_learned_selection_live_executable']}`",
        "",
        "Bottlenecks:",
        "",
        *[f"- `{item}`" for item in summary["integration_bottlenecks"]],
        "",
        "Interpretation:",
        "",
        summary["interpretation"],
        "",
        "Known limits:",
        "",
        "- This is a diagnostic, not a new capability claim.",
        "- It uses the public environment for the selected action plan attempt.",
        "- It does not add a live observation-to-option-state classifier.",
        "- It does not make M10 a full LS20 solver.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()

    summary = evaluate(
        game=args.game,
        budget=args.budget,
        output_events=args.events,
        output_ledger=args.ledger,
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
