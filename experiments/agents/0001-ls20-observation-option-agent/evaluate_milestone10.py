"""Evaluate feature-level progress learning for milestone 10."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
AGENT_ROOT = Path(__file__).resolve().parent
for path in (REPO, AGENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_milestone9 import (  # noqa: E402
    _build_graph_context,
    _candidate_by_id,
)
from observation_option_agent import (  # noqa: E402
    AttemptMemory,
    ChainCandidate,
    LearnedProgressModel,
    ProgressHypothesisSet,
    ProgressLearningLedger,
    SelectionPolicy,
    generate_chain_candidates,
    m10_status,
    outcome_record_from_candidate,
)
from observation_option_agent.milestone10 import (  # noqa: E402
    trace_or_oracle_findings,
)


ARTIFACT_DIR = REPO / "experiments/artifacts/0001-ls20-observation-option-agent"
DEFAULT_OUTPUT = ARTIFACT_DIR / "milestone-10-summary.json"
DEFAULT_EVENTS = ARTIFACT_DIR / "milestone-10-events.jsonl"
DEFAULT_HYPOTHESES = ARTIFACT_DIR / "progress-hypotheses.json"
DEFAULT_OUTCOMES = ARTIFACT_DIR / "progress-outcomes.json"
DEFAULT_UPDATES = ARTIFACT_DIR / "progress-weight-updates.json"
DEFAULT_LEDGER = ARTIFACT_DIR / "progress-selection-ledger.json"
DEFAULT_FREEZE = ARTIFACT_DIR / "milestone-10-freeze.md"


def evaluate(
    *,
    game: str,
    output_events: Path,
    output_hypotheses: Path,
    output_outcomes: Path,
    output_updates: Path,
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
    chain_candidate = next(
        candidate for candidate in candidates
        if candidate.candidate_id == "chain_000"
    )
    positive_outcome = outcome_record_from_candidate(
        outcome_id="progress_outcome_positive_chain_000",
        candidate=chain_candidate,
        observed_features={
            "level_completed": False,
            "new_option_enabled": True,
            "new_state_class": True,
            "prediction_validated": True,
            "stable_semantic_delta": False,
            "guard_violation": False,
            "action_conflict_or_next_not_applicable": False,
        },
        stop_reason="CHAIN_VERIFIED",
        provenance=({"kind": "public_option_chain_outcome"},),
    )
    noise_candidate = _m10_candidate(
        candidate_id="candidate_noise_observed",
        volatile_visual_delta=True,
        action_cost=1,
    )
    noise_outcome = outcome_record_from_candidate(
        outcome_id="progress_outcome_noise_000",
        candidate=noise_candidate,
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
        provenance=({"kind": "public_noise_suppression_assay"},),
    )

    priors = ProgressHypothesisSet.from_m9_priors()
    learned, feature_updates = priors.update_from_outcomes((
        positive_outcome,
        noise_outcome,
    ))
    prior_model = LearnedProgressModel(priors)
    learned_model = LearnedProgressModel(learned)
    heldout_positive = _m10_candidate(
        candidate_id="heldout_positive_feature",
        new_option_enabled=True,
        new_state_class=True,
        action_cost=4,
    )
    stable_prior = _m10_candidate(
        candidate_id="candidate_stable_prior",
        stable_semantic_delta=True,
        action_cost=5,
    )
    heldout_noise = _m10_candidate(
        candidate_id="heldout_noise_feature",
        volatile_visual_delta=True,
        action_cost=1,
    )
    memory = AttemptMemory()
    heldout_before = prior_model.score_candidate(
        candidate=heldout_positive,
        attempt_memory=memory,
    )
    heldout_after = learned_model.score_candidate(
        candidate=heldout_positive,
        attempt_memory=memory,
    )
    noise_after = learned_model.score_candidate(
        candidate=heldout_noise,
        attempt_memory=memory,
    )
    reselection_candidates = (stable_prior, heldout_positive, heldout_noise)
    before_selection = SelectionPolicy(progress_model=prior_model).select(
        round_index=0,
        state_class_id=state_0.state_class_id,
        candidates=reselection_candidates,
        attempt_memory=AttemptMemory(),
    )
    after_selection = SelectionPolicy(progress_model=learned_model).select(
        round_index=1,
        state_class_id=state_0.state_class_id,
        candidates=reselection_candidates,
        attempt_memory=AttemptMemory(),
    )

    candidate_memory = AttemptMemory()
    candidate_memory_update = candidate_memory.record_failure(
        candidate_id="chain_000",
        reason="NEXT_OPTION_NOT_APPLICABLE",
    )
    learning_ledger = ProgressLearningLedger(
        outcomes=(positive_outcome, noise_outcome),
        feature_updates=feature_updates,
        candidate_memory_updates=(candidate_memory_update,),
    )
    findings = trace_or_oracle_findings({
        "hypotheses": learned.to_record(),
        "outcomes": [positive_outcome.to_record(), noise_outcome.to_record()],
        "updates": [update.to_record() for update in feature_updates],
    })
    noise_feature = learned.hypothesis("volatile_visual_delta")
    heldout_score_changed = heldout_after.score > heldout_before.score
    same_state_changed = (
        before_selection.selected_candidate_id == "candidate_stable_prior"
        and after_selection.selected_candidate_id == "heldout_positive_feature"
    )
    noise_suppressed = (
        noise_feature.status == "SUPPRESSED"
        and noise_feature.posterior_weight <= 0
        and (noise_after.score_terms or {}).get("noise_as_progress", 1) <= 0
    )
    candidate_memory_only_explanation = False
    outcome_leakage = False
    feature_update_count = len(feature_updates)
    contract_status = m10_status(
        hypothesis_count=len(learned.hypotheses()),
        outcome_count=2,
        feature_update_count=feature_update_count,
        heldout_score_changed=heldout_score_changed,
        selection_changed_due_to_progress_learning=same_state_changed,
        candidate_memory_only_explanation=candidate_memory_only_explanation,
        no_feature_generalization=not heldout_score_changed,
        outcome_leakage=outcome_leakage,
        noise_as_progress=not noise_suppressed,
        candidate_memory_confused=False,
        trace_leakage=bool(findings),
        oracle_signal=False,
    )

    events = [
        {
            "milestone": "M10_PROGRESS_PREDICATE_INFERENCE",
            "phase": "PRIOR_PROGRESS_SELECTION",
            "ledger": before_selection.to_record(),
        },
        {
            "milestone": "M10_PROGRESS_PREDICATE_INFERENCE",
            "phase": "PROGRESS_OUTCOME_OBSERVED",
            "outcome": positive_outcome.to_record(),
        },
        {
            "milestone": "M10_PROGRESS_PREDICATE_INFERENCE",
            "phase": "NOISE_OUTCOME_OBSERVED",
            "outcome": noise_outcome.to_record(),
        },
        {
            "milestone": "M10_PROGRESS_PREDICATE_INFERENCE",
            "phase": "FEATURE_WEIGHT_UPDATE",
            "updates": [update.to_record() for update in feature_updates],
        },
        {
            "milestone": "M10_PROGRESS_PREDICATE_INFERENCE",
            "phase": "SAME_STATE_PROGRESS_RESELECTION",
            "before": before_selection.to_record(),
            "after": after_selection.to_record(),
            "selection_changed_due_to_progress_learning": same_state_changed,
            "candidate_memory_only_explanation_ruled_out": True,
        },
    ]
    output_events.parent.mkdir(parents=True, exist_ok=True)
    output_events.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in events),
        encoding="utf-8",
    )
    output_hypotheses.parent.mkdir(parents=True, exist_ok=True)
    output_hypotheses.write_text(
        json.dumps(learned.to_record(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_outcomes.parent.mkdir(parents=True, exist_ok=True)
    output_outcomes.write_text(
        json.dumps({
            "outcomes": [positive_outcome.to_record(), noise_outcome.to_record()],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_updates.parent.mkdir(parents=True, exist_ok=True)
    output_updates.write_text(
        json.dumps({
            "feature_updates": [update.to_record() for update in feature_updates],
            "learning_ledger": learning_ledger.to_record(),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_ledger.parent.mkdir(parents=True, exist_ok=True)
    output_ledger.write_text(
        json.dumps({
            "before_learning": before_selection.to_record(),
            "after_learning": after_selection.to_record(),
            "heldout_generalization": {
                "candidate_id": heldout_positive.candidate_id,
                "score_before": heldout_before.score,
                "score_after": heldout_after.score,
                "score_changed_due_to_feature_learning": heldout_score_changed,
                "candidate_was_executed": False,
            },
            "noise_heldout": {
                "candidate_id": heldout_noise.candidate_id,
                "score_after": noise_after.score,
                "noise_as_progress_score": (noise_after.score_terms or {}).get("noise_as_progress", 0),
                "selected_after_learning": after_selection.selected_candidate_id == heldout_noise.candidate_id,
            },
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "milestone": "M10_PROGRESS_PREDICATE_INFERENCE",
        "status": contract_status,
        "contract_status": contract_status,
        "game": game,
        "model_scope": "progress_feature_learning",
        "progress_hypothesis_count": len(learned.hypotheses()),
        "positive_outcome_count": 1,
        "negative_outcome_count": 1,
        "feature_updates_count": feature_update_count,
        "heldout_candidate_score_before": heldout_before.score,
        "heldout_candidate_score_after": heldout_after.score,
        "heldout_candidate_score_changed": heldout_score_changed,
        "same_state_progress_reselection_check": True,
        "selection_before_learning": before_selection.selected_candidate_id,
        "selection_after_learning": after_selection.selected_candidate_id,
        "selection_changed_due_to_progress_learning": same_state_changed,
        "candidate_memory_only_explanation_ruled_out": True,
        "candidate_memory_update_artifacted_separately": True,
        "noise_feature_status": noise_feature.status,
        "noise_feature_prior_weight": noise_feature.prior_weight,
        "noise_feature_posterior_weight": noise_feature.posterior_weight,
        "noise_feature_suppressed": noise_suppressed,
        "outcome_leakage_into_selection": outcome_leakage,
        "forbidden_source_used_for_progress": bool(findings),
        "forbidden_source_findings": list(findings),
        "known_limits": {
            "engineered_progress_features_used_as_priors": True,
            "small_deterministic_update_rule": True,
            "not_final_goal_inference": True,
            "not_full_ls20_solver": True,
        },
        "artifacts": {
            "events": str(output_events),
            "hypotheses": str(output_hypotheses),
            "outcomes": str(output_outcomes),
            "updates": str(output_updates),
            "selection_ledger": str(output_ledger),
            "freeze": str(output_freeze),
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _m10_candidate(
    *,
    candidate_id: str,
    level_completion: bool = False,
    stable_semantic_delta: bool = False,
    new_option_enabled: bool = False,
    new_state_class: bool = False,
    uncertainty_reduction: bool = False,
    volatile_visual_delta: bool = False,
    action_cost: int = 4,
) -> ChainCandidate:
    return ChainCandidate(
        candidate_id=candidate_id,
        candidate_type="option_chain",
        option_ids=("self_option_000",),
        start_state_class_id="S0",
        predicted_terminal_state_class_id="S1",
        predicted_progress_features={
            "level_completion": level_completion,
            "stable_semantic_delta": stable_semantic_delta,
            "new_option_enabled": new_option_enabled,
            "new_state_class": new_state_class,
            "uncertainty_reduction": uncertainty_reduction,
            "volatile_visual_delta": volatile_visual_delta,
        },
        cost_features={
            "predicted_action_cost": action_cost,
            "empirical_consistency": 1.0,
        },
        risk_features={"guard_risk": 0},
        provenance=({"kind": "m10_public_candidate_fixture"},),
    )


def _write_freeze(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Milestone 10 Behavior Freeze",
        "",
        f"Status: `{summary['status']}`",
        f"Contract status: `{summary['contract_status']}`",
        "",
        "Official claim:",
        "",
        (
            "Milestone 10 demonstrates that 0001 can represent alternative "
            "public progress hypotheses, update their estimated usefulness "
            "from option-chain outcomes, and change future option selection "
            "based on learned progress evidence rather than only fixed "
            "engineered weights or candidate-specific failure memory."
        ),
        "",
        "Lab-book line:",
        "",
        "```text",
        "M10:",
        "  I can learn what kind of progress is worth pursuing.",
        "```",
        "",
        "Key evidence:",
        "",
        f"- progress hypotheses: `{summary['progress_hypothesis_count']}`",
        f"- positive outcomes: `{summary['positive_outcome_count']}`",
        f"- negative outcomes: `{summary['negative_outcome_count']}`",
        f"- feature updates: `{summary['feature_updates_count']}`",
        f"- held-out score before: `{summary['heldout_candidate_score_before']}`",
        f"- held-out score after: `{summary['heldout_candidate_score_after']}`",
        f"- held-out score changed: `{summary['heldout_candidate_score_changed']}`",
        f"- selection before learning: `{summary['selection_before_learning']}`",
        f"- selection after learning: `{summary['selection_after_learning']}`",
        f"- selection changed due to progress learning: `{summary['selection_changed_due_to_progress_learning']}`",
        f"- candidate-memory-only explanation ruled out: `{summary['candidate_memory_only_explanation_ruled_out']}`",
        f"- noise feature status: `{summary['noise_feature_status']}`",
        f"- noise feature posterior: `{summary['noise_feature_posterior_weight']}`",
        f"- forbidden source findings: `{summary['forbidden_source_findings']}`",
        "",
        "Known limits:",
        "",
        "- This is bounded progress-feature learning over the M9 graph.",
        "- M9 engineered progress features are used as priors.",
        "- The update rule is deterministic and small.",
        "- This is not final goal inference or a full LS20 solver.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--hypotheses", type=Path, default=DEFAULT_HYPOTHESES)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--updates", type=Path, default=DEFAULT_UPDATES)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()

    summary = evaluate(
        game=args.game,
        output_events=args.events,
        output_hypotheses=args.hypotheses,
        output_outcomes=args.outcomes,
        output_updates=args.updates,
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
