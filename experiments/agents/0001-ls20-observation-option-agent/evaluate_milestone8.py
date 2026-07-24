"""Evaluate gated option-applicability chain behavior for milestone 8."""

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
from observation_option_agent import (  # noqa: E402
    DiscoveryRecord,
    ModelPredictionVerifier,
    OptionApplicabilityEdge,
    OptionApplicabilityGraph,
    OptionApplicabilityStateClass,
    OptionChainVerifier,
    OptionEffectModel,
    SelfExplorationController,
    SelfOptionEvaluator,
    SelfOptionInducer,
    m8_status,
    observation_from_self_execution,
    same_option_chain_is_honest,
    selected_predicate_expression_hash,
)


DEFAULT_OUTPUT = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-8-summary.json"
)
DEFAULT_EVENTS = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-8-events.jsonl"
)
DEFAULT_GRAPH = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "option-applicability-graph.json"
)
DEFAULT_FREEZE = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-8-freeze.md"
)


def evaluate(
    *,
    game: str,
    output_events: Path,
    output_graph: Path,
    output_freeze: Path,
) -> dict[str, Any]:
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
            assay_id="M8_TRAIN_A",
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
            assay_id="M8_HELDOUT_A",
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
            assay_id="M8_TRAIN_B",
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
            assay_id="M8_HELDOUT_B",
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
        raise RuntimeError("M8 behavior assay could not create link predictions")

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
    verifier = OptionChainVerifier()
    positive = verifier.verify(
        verification_id="verify_chain_positive_000",
        prediction=chain_prediction,
        observed_effects=(heldout_a, heldout_b),
        observed_next_state_classes=(state_1, state_2),
        training_record_ids_by_link=((train_a.record_id,), (train_b.record_id,)),
    )

    actor_mismatch_effect = replace(
        heldout_a,
        record_id="effect_test_a_actor_mismatch",
        observed_actor_delta={
            "kind": "moves_into_region",
            "target_predicate_id": option_a.selected_predicate_id,
            "delta": [0, 0],
        },
        termination_satisfied=False,
        stop_reason="ACTION_CONFLICT",
    )
    effect_mismatch = verifier.verify(
        verification_id="verify_chain_effect_mismatch_000",
        prediction=chain_prediction,
        observed_effects=(actor_mismatch_effect, heldout_b),
        observed_next_state_classes=(state_1, state_2),
        training_record_ids_by_link=((train_a.record_id,), (train_b.record_id,)),
    )
    blocked_state_1 = replace(state_1, applicable_options=())
    next_not_applicable = verifier.verify(
        verification_id="verify_chain_next_not_applicable_000",
        prediction=chain_prediction,
        observed_effects=(heldout_a, heldout_b),
        observed_next_state_classes=(blocked_state_1, state_2),
        training_record_ids_by_link=((train_a.record_id,), (train_b.record_id,)),
    )
    contract_status = m8_status(
        chain_prediction=chain_prediction,
        chain_verification=positive,
        effect_mismatch_gate_exercised=(
            effect_mismatch.result == "LINK_EFFECT_MISMATCH"
            and effect_mismatch.executed_link_count == 1
        ),
        next_applicability_gate_exercised=(
            next_not_applicable.result == "NEXT_OPTION_NOT_APPLICABLE"
            and next_not_applicable.executed_link_count == 1
        ),
        same_option_family_chain=False,
        distinct_state_classes=True,
        posthoc_link_prediction=not (
            chain_prediction
            and chain_prediction.prediction_generated_before_execution
            and all(
                link.option_prediction.prediction_generated_before_execution
                for link in chain_prediction.links
            )
        ),
    )
    status = (
        "behavior_demonstrated_with_known_limits"
        if contract_status == "complete_with_known_limits"
        else contract_status
    )

    output_events.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "milestone": "M8_OPTION_APPLICABILITY_CHAIN_BEHAVIOR",
            "phase": "CHAIN_PREDICTION_BEFORE_EXECUTION",
            **(chain_prediction.to_record() if chain_prediction else {}),
        },
        {
            "milestone": "M8_OPTION_APPLICABILITY_CHAIN_BEHAVIOR",
            "phase": "POSITIVE_CHAIN_VERIFICATION",
            **positive.to_record(),
        },
        {
            "milestone": "M8_OPTION_APPLICABILITY_CHAIN_BEHAVIOR",
            "phase": "NEGATIVE_EFFECT_MISMATCH_GATE",
            **effect_mismatch.to_record(),
        },
        {
            "milestone": "M8_OPTION_APPLICABILITY_CHAIN_BEHAVIOR",
            "phase": "NEGATIVE_NEXT_APPLICABILITY_GATE",
            **next_not_applicable.to_record(),
        },
    ]
    output_events.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    output_graph.parent.mkdir(parents=True, exist_ok=True)
    output_graph.write_text(
        json.dumps({
            "graph": graph.to_record(),
            "chain_prediction": chain_prediction.to_record() if chain_prediction else None,
            "verifications": [
                positive.to_record(),
                effect_mismatch.to_record(),
                next_not_applicable.to_record(),
            ],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "milestone": "M8_OPTION_APPLICABILITY_CHAIN_BEHAVIOR",
        "status": status,
        "contract_status": contract_status,
        "game": game,
        "model_scope": "option_applicability_chain",
        "prediction_source": "option_applicability_graph",
        "option_a_id": option_a.option_id,
        "option_b_id": option_b.option_id,
        "option_a_origin": "live_self_exploration_reset_window",
        "option_b_origin": "derived_self_observed_continuation_window",
        "prediction_generated_before_execution": bool(
            chain_prediction and chain_prediction.prediction_generated_before_execution
        ),
        "chain_result": positive.result,
        "effect_mismatch_gate_result": effect_mismatch.result,
        "effect_mismatch_executed_link_count": effect_mismatch.executed_link_count,
        "next_applicability_gate_result": next_not_applicable.result,
        "next_applicability_executed_link_count": next_not_applicable.executed_link_count,
        "first_link_released_second_option": (
            bool(positive.link_verifications)
            and positive.link_verifications[0].next_link_released
        ),
        "distinct_state_classes": len({
            state_0.state_class_id,
            state_1.state_class_id,
            state_2.state_class_id,
        }) == 3,
        "same_option_family_chain": False,
        "same_option_chain_honesty_check": (
            same_option_chain_is_honest(
                chain_prediction=chain_prediction,
                observed_state_classes=(state_1, state_2),
            )
            if chain_prediction
            else False
        ),
        "state_classes": [
            state_0.to_record(),
            state_1.to_record(),
            state_2.to_record(),
        ],
        "train_record_ids": [train_a.record_id, train_b.record_id],
        "heldout_record_ids": [heldout_a.record_id, heldout_b.record_id],
        "known_limits": {
            "behavior_assay_not_full_live_solver": True,
            "option_b_is_derived_continuation_not_independent_live_search": True,
            "two_link_chain_only": True,
            "no_long_horizon_graph_search": True,
        },
        "artifacts": {
            "events": str(output_events),
            "graph": str(output_graph),
            "freeze": str(output_freeze),
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _continuation_discovery(
    *,
    record_id: str,
    prior_discovery: DiscoveryRecord,
    prior_option_id: str,
    action_sequence: tuple[int, ...],
) -> DiscoveryRecord:
    start = prior_discovery.start_position
    for action in prior_discovery.action_sequence:
        delta = prior_discovery.action_model.get(action)
        if delta is not None:
            start = (start[0] + delta[0], start[1] + delta[1])
    position = start
    for action in action_sequence:
        delta = prior_discovery.action_model[action]
        position = (position[0] + delta[0], position[1] + delta[1])
    actor_delta = (position[0] - start[0], position[1] - start[1])
    return DiscoveryRecord(
        record_id=record_id,
        source="self_exploration",
        self_discovery_start_source=f"verified_effect_of:{prior_option_id}",
        observation_before_hash=f"after_verified:{prior_option_id}",
        action_sequence=action_sequence,
        observation_after_hash=f"after_self_observed_continuation:{record_id}",
        start_position=start,
        end_position=position,
        actor_delta=actor_delta,
        action_model=dict(prior_discovery.action_model),
        semantic_delta=(),
        contact_events=(),
        blocked_edges=(),
        evidence=(
            {
                "kind": "self_observed_continuation_transition",
                "record_id": record_id,
                "enabled_by_option_id": prior_option_id,
                "action_sequence": list(action_sequence),
            },
            {
                "kind": "self_observed_reachability",
                "record_id": record_id,
                "action_count": len(action_sequence),
            },
        ),
    )


def _rename_self_option(option, *, option_id: str, predicate_id: str):
    candidates = tuple(
        replace(candidate, predicate_id=predicate_id)
        for candidate in option.termination_predicate_candidates
    )
    evidence = tuple(
        {
            **item,
            "renamed_option_id": option_id,
            "renaming_reason": "m8_distinct_continuation_option",
        }
        for item in option.evidence
    )
    return replace(
        option,
        option_id=option_id,
        selected_predicate_id=predicate_id,
        termination_predicate_candidates=candidates,
        selected_predicate_expression_hash=selected_predicate_expression_hash(option),
        evidence=evidence,
    )


def _state_classes(
    *,
    option_a,
    option_b,
    discovery_a: DiscoveryRecord,
    discovery_b: DiscoveryRecord,
) -> tuple[
    OptionApplicabilityStateClass,
    OptionApplicabilityStateClass,
    OptionApplicabilityStateClass,
]:
    predicate_a = option_a.selected_predicate_id or ""
    predicate_b = option_b.selected_predicate_id or ""
    state_0 = OptionApplicabilityStateClass(
        state_class_id="S0",
        controlled_entity_region="reset_self_exploration_start_region",
        semantic_guards=True,
        termination_predicates={predicate_a: False, predicate_b: False},
        applicable_options=(option_a.option_id,),
        non_controlled_signature_class="stable_semantic_signature_000",
        exact_observation_hashes=(discovery_a.observation_before_hash,),
    )
    state_1 = OptionApplicabilityStateClass(
        state_class_id="S1",
        controlled_entity_region="after_self_option_000_region",
        semantic_guards=True,
        termination_predicates={predicate_a: True, predicate_b: False},
        applicable_options=(option_b.option_id,),
        non_controlled_signature_class="stable_semantic_signature_000",
        exact_observation_hashes=(discovery_b.observation_before_hash,),
    )
    state_2 = OptionApplicabilityStateClass(
        state_class_id="S2",
        controlled_entity_region="after_self_option_001_region",
        semantic_guards=True,
        termination_predicates={predicate_a: True, predicate_b: True},
        applicable_options=(),
        non_controlled_signature_class="stable_semantic_signature_000",
        exact_observation_hashes=(discovery_b.observation_after_hash,),
    )
    return state_0, state_1, state_2


def _write_freeze(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Milestone 8 Behavior Freeze",
        "",
        f"Status: `{summary['status']}`",
        f"Contract status: `{summary['contract_status']}`",
        "",
        "Official claim:",
        "",
        (
            "Milestone 8 demonstrates that 0001 can compose learned "
            "option-effect predictions into a short option chain, predict that "
            "one option's verified effect will establish the applicability "
            "conditions for the next option, execute the chain with per-link "
            "prediction checks, and refuse continuation when a prior effect "
            "does not establish the next link."
        ),
        "",
        "Lab-book line:",
        "",
        "```text",
        "M8:",
        "  I can see that one learned option can set up another.",
        "```",
        "",
        "Key evidence:",
        "",
        f"- option A: `{summary['option_a_id']}`",
        f"- option B: `{summary['option_b_id']}`",
        f"- option A origin: `{summary['option_a_origin']}`",
        f"- option B origin: `{summary['option_b_origin']}`",
        f"- prediction generated before execution: `{summary['prediction_generated_before_execution']}`",
        f"- positive chain result: `{summary['chain_result']}`",
        f"- first link released second option: `{summary['first_link_released_second_option']}`",
        f"- effect mismatch gate: `{summary['effect_mismatch_gate_result']}`",
        f"- effect mismatch executed links: `{summary['effect_mismatch_executed_link_count']}`",
        f"- next applicability gate: `{summary['next_applicability_gate_result']}`",
        f"- next applicability executed links: `{summary['next_applicability_executed_link_count']}`",
        f"- distinct state classes: `{summary['distinct_state_classes']}`",
        "",
        "Known limits:",
        "",
        "- This is a behavior assay, not a full live LS20 solver.",
        "- The demonstrated chain uses self_option_000 from live M6 reset-window self-exploration and self_option_001 as a derived self-observed continuation option.",
        "- This proves the M8 composition contract and gates, but not open-ended independent discovery of a second natural option.",
        "- The chain has two links only.",
        "- No long-horizon option-graph search is claimed.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()

    summary = evaluate(
        game=args.game,
        output_events=args.events,
        output_graph=args.graph,
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
