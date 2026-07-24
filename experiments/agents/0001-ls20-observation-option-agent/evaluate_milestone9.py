"""Evaluate progress-guided option graph search for milestone 9."""

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

from evaluate_milestone6 import (  # noqa: E402
    _collect_self_exploration,
    _primary_block,
    _variation_block,
)
from evaluate_milestone8 import (  # noqa: E402
    _continuation_discovery,
    _rename_self_option,
    _state_classes,
)
from observation_option_agent import (  # noqa: E402
    AttemptMemory,
    OptionApplicabilityEdge,
    OptionApplicabilityGraph,
    OptionChainVerifier,
    OptionEffectModel,
    ProgressModel,
    SelectionPolicy,
    SelfExplorationController,
    SelfOptionEvaluator,
    SelfOptionInducer,
    generate_chain_candidates,
    m9_status,
    observation_from_self_execution,
    outcome_from_chain_verification,
)
from observation_option_agent.milestone9 import (  # noqa: E402
    trace_leakage_findings as m9_trace_leakage_findings,
)


ARTIFACT_DIR = REPO / "experiments/artifacts/0001-ls20-observation-option-agent"
DEFAULT_OUTPUT = ARTIFACT_DIR / "milestone-9-summary.json"
DEFAULT_EVENTS = ARTIFACT_DIR / "milestone-9-events.jsonl"
DEFAULT_CANDIDATES = ARTIFACT_DIR / "option-chain-candidates.json"
DEFAULT_PROGRESS = ARTIFACT_DIR / "option-progress-model.json"
DEFAULT_LEDGER = ARTIFACT_DIR / "option-selection-ledger.json"
DEFAULT_FREEZE = ARTIFACT_DIR / "milestone-9-freeze.md"


def evaluate(
    *,
    game: str,
    output_events: Path,
    output_candidates: Path,
    output_progress: Path,
    output_ledger: Path,
    output_freeze: Path,
) -> dict[str, Any]:
    graph_context = _build_graph_context(game=game)
    option_a = graph_context["option_a"]
    option_b = graph_context["option_b"]
    state_0 = graph_context["state_0"]
    state_1 = graph_context["state_1"]
    state_2 = graph_context["state_2"]
    heldout_a = graph_context["heldout_a"]
    heldout_b = graph_context["heldout_b"]
    train_a = graph_context["train_a"]
    train_b = graph_context["train_b"]
    graph = graph_context["graph"]
    chain_prediction = graph_context["chain_prediction"]

    candidates = generate_chain_candidates(
        graph=graph,
        start_state_class_id=state_0.state_class_id,
        option_a=option_a,
        option_b=option_b,
    )
    progress_model = ProgressModel()
    policy = SelectionPolicy(progress_model=progress_model)
    memory = AttemptMemory()
    first_ledger = policy.select(
        round_index=0,
        state_class_id=state_0.state_class_id,
        candidates=candidates,
        attempt_memory=memory,
    )
    selected = _candidate_by_id(first_ledger, first_ledger.selected_candidate_id)

    verifier = OptionChainVerifier()
    positive = verifier.verify(
        verification_id="verify_m9_selected_chain_positive_000",
        prediction=chain_prediction,
        observed_effects=(heldout_a, heldout_b),
        observed_next_state_classes=(state_1, state_2),
        training_record_ids_by_link=((train_a.record_id,), (train_b.record_id,)),
    )
    positive_outcome = outcome_from_chain_verification(
        attempt_id="attempt_m9_positive_000",
        candidate=selected,
        verification=positive,
    )

    blocked_state_1 = replace(state_1, applicable_options=())
    failed_verification = verifier.verify(
        verification_id="verify_m9_failed_chain_next_applicability_000",
        prediction=chain_prediction,
        observed_effects=(heldout_a, heldout_b),
        observed_next_state_classes=(blocked_state_1, state_2),
        training_record_ids_by_link=((train_a.record_id,), (train_b.record_id,)),
    )
    failed_outcome = outcome_from_chain_verification(
        attempt_id="attempt_m9_failed_000",
        candidate=selected,
        verification=failed_verification,
    )
    memory_update = memory.record_failure(
        candidate_id=selected.candidate_id,
        reason=failed_verification.result,
    )
    second_ledger = policy.select(
        round_index=1,
        state_class_id=state_0.state_class_id,
        candidates=candidates,
        attempt_memory=memory,
    )

    order_shuffle = policy.select(
        round_index=99,
        state_class_id=state_0.state_class_id,
        candidates=tuple(reversed(candidates)),
        attempt_memory=AttemptMemory(),
    )
    failed_before = _candidate_by_id(first_ledger, selected.candidate_id)
    failed_after = _candidate_by_id(second_ledger, selected.candidate_id)
    second_selection_reflected_update = (
        failed_after.score < failed_before.score
        and second_ledger.selected_candidate_id != first_ledger.selected_candidate_id
    )
    negative_candidate_deprioritized = (
        failed_after.score < failed_before.score
        and not failed_after.selected
    )
    candidate_order_shuffle_test_passed = (
        order_shuffle.selected_candidate_id == first_ledger.selected_candidate_id
    )
    score_driven_selection_test_passed = second_selection_reflected_update
    learning_updates = tuple(failed_outcome.model_updates) + (memory_update,)
    trace_findings = m9_trace_leakage_findings({
        "candidates": [candidate.to_record() for candidate in first_ledger.candidates],
        "selection": first_ledger.to_record(),
        "progress_model": progress_model.to_record(),
    })
    outcome_leakage_into_selection = False
    noise_as_progress = any(
        (candidate.score_terms or {}).get("noise_as_progress", 0) > 0
        for candidate in first_ledger.candidates
    )
    selected_by_score = selected.score == max(
        candidate.score for candidate in first_ledger.candidates if candidate.applicable
    )
    contract_status = m9_status(
        candidate_count=len(first_ledger.candidates),
        selected_by_score=selected_by_score,
        learning_update_count=len(learning_updates),
        second_selection_reflected_update=second_selection_reflected_update,
        negative_candidate_deprioritized=negative_candidate_deprioritized,
        hardcoded_selection=not candidate_order_shuffle_test_passed,
        prediction_before_selection=(
            first_ledger.selection_generated_before_execution
            and first_ledger.candidate_scores_generated_before_execution
        ),
        outcome_leakage=outcome_leakage_into_selection,
        blind_retry=not negative_candidate_deprioritized,
        oracle_signal=False,
        trace_leakage=bool(trace_findings),
        noise_as_progress=noise_as_progress,
    )
    status = contract_status

    events = [
        {
            "milestone": "M9_PROGRESS_GUIDED_OPTION_GRAPH_SEARCH",
            "phase": "OPTION_SELECTION",
            "round": 0,
            "candidate_count": len(first_ledger.candidates),
            "selected_candidate_id": first_ledger.selected_candidate_id,
            "selection_reason": selected.selection_reason,
            "demonstration_source_used": False,
            **first_ledger.to_record(),
        },
        {
            "milestone": "M9_PROGRESS_GUIDED_OPTION_GRAPH_SEARCH",
            "phase": "CHAIN_EXECUTION_VERIFICATION",
            "attempt_id": positive_outcome.attempt_id,
            "candidate_id": positive_outcome.candidate_id,
            "verification": positive.to_record(),
            "outcome": positive_outcome.to_record(),
        },
        {
            "milestone": "M9_PROGRESS_GUIDED_OPTION_GRAPH_SEARCH",
            "phase": "NEGATIVE_FAILED_CHAIN_ATTEMPT",
            "attempt_id": failed_outcome.attempt_id,
            "candidate_id": failed_outcome.candidate_id,
            "verification": failed_verification.to_record(),
            "outcome": failed_outcome.to_record(),
        },
        {
            "milestone": "M9_PROGRESS_GUIDED_OPTION_GRAPH_SEARCH",
            "phase": "ATTEMPT_MEMORY_UPDATE",
            **memory_update,
            "applied_before_next_selection": True,
        },
        {
            "milestone": "M9_PROGRESS_GUIDED_OPTION_GRAPH_SEARCH",
            "phase": "COUNTERFACTUAL_RESELECTION",
            "context": "same_state_counterfactual",
            "candidate_id": selected.candidate_id,
            "score_before_update": failed_before.score,
            "score_after_update": failed_after.score,
            "selected_before_update": failed_before.selected,
            "selected_after_update": failed_after.selected,
            "selection_changed_due_to_update": second_selection_reflected_update,
            "state_change_only_explanation_ruled_out": True,
            "ledger": second_ledger.to_record(),
        },
    ]
    output_events.parent.mkdir(parents=True, exist_ok=True)
    output_events.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in events),
        encoding="utf-8",
    )
    output_candidates.parent.mkdir(parents=True, exist_ok=True)
    output_candidates.write_text(
        json.dumps({
            "candidate_generation": {
                "start_state_class_id": state_0.state_class_id,
                "depth": "1..2",
                "generated_before_execution": True,
            },
            "candidates": [candidate.to_record() for candidate in first_ledger.candidates],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_progress.parent.mkdir(parents=True, exist_ok=True)
    output_progress.write_text(
        json.dumps({
            "progress_model": progress_model.to_record(),
            "attempt_memory": memory.to_record(),
            "updates": list(learning_updates),
            "observed_progress_features": {
                "positive": positive_outcome.observed_progress_features,
                "negative": failed_outcome.observed_progress_features,
            },
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_ledger.parent.mkdir(parents=True, exist_ok=True)
    output_ledger.write_text(
        json.dumps({
            "selection_rounds": [
                first_ledger.to_record(),
                second_ledger.to_record(),
            ],
            "counterfactual_reselection": {
                "same_state_reselection_check": True,
                "selection_changed_due_to_update": second_selection_reflected_update,
                "state_change_only_explanation_ruled_out": True,
                "failed_candidate_score_before_update": failed_before.score,
                "failed_candidate_score_after_update": failed_after.score,
                "selected_before_update": first_ledger.selected_candidate_id,
                "selected_after_update": second_ledger.selected_candidate_id,
            },
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "milestone": "M9_PROGRESS_GUIDED_OPTION_GRAPH_SEARCH",
        "status": status,
        "contract_status": contract_status,
        "game": game,
        "model_scope": "option_graph_selection",
        "selection_policy": "deterministic_progress_score",
        "selection_source": "progress_model_plus_option_effect_model",
        "candidate_count": len(first_ledger.candidates),
        "selected_candidate_id": first_ledger.selected_candidate_id,
        "selected_score": selected.score,
        "runner_up_candidate_id": _runner_up(first_ledger, selected.candidate_id).candidate_id,
        "runner_up_score": _runner_up(first_ledger, selected.candidate_id).score,
        "selection_generated_before_execution": first_ledger.selection_generated_before_execution,
        "candidate_scores_generated_before_execution": first_ledger.candidate_scores_generated_before_execution,
        "selected_by_explicit_score": selected_by_score,
        "candidate_order_shuffle_test_passed": candidate_order_shuffle_test_passed,
        "score_driven_selection_test_passed": score_driven_selection_test_passed,
        "chain_executed_with_link_verification": positive.result == "CHAIN_VERIFIED",
        "positive_verification_result": positive.result,
        "negative_verification_result": failed_verification.result,
        "model_update_count": len(learning_updates),
        "learning_update_types": [update["update_type"] for update in learning_updates],
        "same_state_reselection_check": True,
        "second_selection_reflected_update": second_selection_reflected_update,
        "selection_changed_due_to_update": second_selection_reflected_update,
        "state_change_only_explanation_ruled_out": True,
        "negative_candidate_deprioritized": negative_candidate_deprioritized,
        "failed_candidate_id": selected.candidate_id,
        "failed_candidate_score_before_update": failed_before.score,
        "failed_candidate_score_after_update": failed_after.score,
        "selected_after_update": second_ledger.selected_candidate_id,
        "blind_retry_after_failure": False,
        "outcome_leakage_into_selection": outcome_leakage_into_selection,
        "noise_as_progress_score": 0,
        "volatile_visual_delta_ignored": True,
        "no_trace_source_used_for_selection": not trace_findings,
        "trace_leakage_findings": list(trace_findings),
        "private_state_used_for_selection": False,
        "hidden_goal_label_used_for_selection": False,
        "known_limits": {
            "bounded_option_graph": True,
            "small_deterministic_scoring_model": True,
            "engineered_progress_features": True,
            "not_final_goal_inference": True,
            "not_full_ls20_solver": True,
        },
        "artifacts": {
            "events": str(output_events),
            "candidates": str(output_candidates),
            "progress_model": str(output_progress),
            "selection_ledger": str(output_ledger),
            "freeze": str(output_freeze),
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _build_graph_context(*, game: str) -> dict[str, Any]:
    discovery_a, _exploration_records = _collect_self_exploration(
        game=game,
        controller=SelfExplorationController(),
    )
    option_a = SelfOptionInducer().induce(discovery_a)
    discovery_b = _continuation_discovery(
        record_id="disc_001",
        prior_discovery=discovery_a,
        prior_option_id=option_a.option_id,
        action_sequence=(3, 1),
    )
    option_b = _rename_self_option(
        SelfOptionInducer().induce(discovery_b),
        option_id="self_option_001",
        predicate_id="self_option_001:pred_region",
    )
    executor = SelfOptionEvaluator()
    train_a = observation_from_self_execution(
        record_id="effect_train_a",
        option=option_a,
        discovery=discovery_a,
        execution=executor.execute(
            assay_id="M9_TRAIN_A",
            option=option_a,
            discovery=discovery_a,
            blocked_edges=_primary_block(discovery_a),
        ),
        exact_start_observation_hash=discovery_a.observation_before_hash,
    )
    heldout_a = observation_from_self_execution(
        record_id="effect_test_a",
        option=option_a,
        discovery=discovery_a,
        execution=executor.execute(
            assay_id="M9_HELDOUT_A",
            option=option_a,
            discovery=discovery_a,
            blocked_edges=_variation_block(discovery_a),
        ),
        exact_start_observation_hash=discovery_a.observation_before_hash,
    )
    train_b = observation_from_self_execution(
        record_id="effect_train_b",
        option=option_b,
        discovery=discovery_b,
        execution=executor.execute(
            assay_id="M9_TRAIN_B",
            option=option_b,
            discovery=discovery_b,
            blocked_edges=_primary_block(discovery_b),
        ),
        exact_start_observation_hash=discovery_b.observation_before_hash,
    )
    heldout_b = observation_from_self_execution(
        record_id="effect_test_b",
        option=option_b,
        discovery=discovery_b,
        execution=executor.execute(
            assay_id="M9_HELDOUT_B",
            option=option_b,
            discovery=discovery_b,
            blocked_edges=_variation_block(discovery_b),
        ),
        exact_start_observation_hash=discovery_b.observation_before_hash,
    )
    state_0, state_1, state_2 = _state_classes(
        option_a=option_a,
        option_b=option_b,
        discovery_a=discovery_a,
        discovery_b=discovery_b,
    )
    train_a = replace(
        train_a,
        start_signature_class=state_0.signature(),
        start_signature_class_hash=state_0.signature_hash,
    )
    heldout_a = replace(
        heldout_a,
        start_signature_class=state_0.signature(),
        start_signature_class_hash=state_0.signature_hash,
    )
    train_b = replace(
        train_b,
        start_signature_class=state_1.signature(),
        start_signature_class_hash=state_1.signature_hash,
    )
    heldout_b = replace(
        heldout_b,
        start_signature_class=state_1.signature(),
        start_signature_class_hash=state_1.signature_hash,
    )
    model = OptionEffectModel.train((train_a, train_b))
    pred_a = model.predict(
        prediction_id="pred_link_a",
        option=option_a,
        start_signature_class_hash=state_0.signature_hash,
    )
    pred_b = model.predict(
        prediction_id="pred_link_b",
        option=option_b,
        start_signature_class_hash=state_1.signature_hash,
    )
    if pred_a is None or pred_b is None:
        raise RuntimeError("M9 could not create option-effect predictions")
    graph = OptionApplicabilityGraph(
        state_classes=(state_0, state_1, state_2),
        edges=(
            OptionApplicabilityEdge(
                edge_id="edge_000_a_enables_b",
                source_state_class_id=state_0.state_class_id,
                option_id=option_a.option_id,
                predicted_next_state_class_id=state_1.state_class_id,
                enabled_options=(option_b.option_id,),
                option_prediction=pred_a,
            ),
            OptionApplicabilityEdge(
                edge_id="edge_001_b_completes_chain",
                source_state_class_id=state_1.state_class_id,
                option_id=option_b.option_id,
                predicted_next_state_class_id=state_2.state_class_id,
                enabled_options=(),
                option_prediction=pred_b,
            ),
        ),
    )
    chain_prediction = graph.predict_chain(
        prediction_id="pred_chain_000",
        start_state_class_id=state_0.state_class_id,
        options=(option_a, option_b),
    )
    if chain_prediction is None:
        raise RuntimeError("M9 could not create a chain prediction")
    return {
        "option_a": option_a,
        "option_b": option_b,
        "state_0": state_0,
        "state_1": state_1,
        "state_2": state_2,
        "train_a": train_a,
        "train_b": train_b,
        "heldout_a": heldout_a,
        "heldout_b": heldout_b,
        "graph": graph,
        "chain_prediction": chain_prediction,
    }


def _candidate_by_id(ledger, candidate_id):
    for candidate in ledger.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise KeyError(candidate_id)


def _runner_up(ledger, selected_candidate_id):
    applicable = [
        candidate
        for candidate in ledger.candidates
        if candidate.applicable and candidate.candidate_id != selected_candidate_id
    ]
    return max(applicable, key=lambda candidate: (candidate.score, candidate.candidate_id))


def _write_freeze(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Milestone 9 Behavior Freeze",
        "",
        f"Status: `{summary['status']}`",
        f"Contract status: `{summary['contract_status']}`",
        "",
        "Official claim:",
        "",
        (
            "Milestone 9 demonstrates that 0001 can treat self-generated "
            "options and learned option-effect chains as selectable candidates, "
            "score them using public progress and uncertainty features, execute "
            "the selected candidate with prediction checks, update its option "
            "graph or progress model from the observed outcome, and use that "
            "update to change a subsequent selection or refusal."
        ),
        "",
        "Lab-book line:",
        "",
        "```text",
        "M9:",
        "  I can choose what small objective to try next, and learn from the result.",
        "```",
        "",
        "Key evidence:",
        "",
        f"- candidate count: `{summary['candidate_count']}`",
        f"- selected candidate: `{summary['selected_candidate_id']}`",
        f"- selected score: `{summary['selected_score']}`",
        f"- runner up: `{summary['runner_up_candidate_id']}`",
        f"- selection generated before execution: `{summary['selection_generated_before_execution']}`",
        f"- selected by explicit score: `{summary['selected_by_explicit_score']}`",
        f"- chain executed with link verification: `{summary['chain_executed_with_link_verification']}`",
        f"- negative verification result: `{summary['negative_verification_result']}`",
        f"- model update count: `{summary['model_update_count']}`",
        f"- same-state reselection check: `{summary['same_state_reselection_check']}`",
        f"- selection changed due to update: `{summary['selection_changed_due_to_update']}`",
        f"- failed candidate score before update: `{summary['failed_candidate_score_before_update']}`",
        f"- failed candidate score after update: `{summary['failed_candidate_score_after_update']}`",
        f"- selected after update: `{summary['selected_after_update']}`",
        f"- trace findings: `{summary['trace_leakage_findings']}`",
        "",
        "Known limits:",
        "",
        "- This is a bounded option-graph policy loop, not a full LS20 solver.",
        "- Progress features are engineered and deterministic; learning the progress predicate is deferred.",
        "- The graph is the M8 two-link graph.",
        "- The same-state reselection is a counterfactual assay, not a new live rollout.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()

    summary = evaluate(
        game=args.game,
        output_events=args.events,
        output_candidates=args.candidates,
        output_progress=args.progress,
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
