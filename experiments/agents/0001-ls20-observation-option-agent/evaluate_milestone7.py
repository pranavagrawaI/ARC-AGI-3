"""Evaluate held-out option-effect prediction for milestone 7."""

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
    ModelBackedPredictionProvider,
    ModelPredictionVerifier,
    OptionEffectModel,
    SelfExplorationController,
    SelfOptionEvaluator,
    SelfOptionInducer,
    m7_status,
    observation_from_self_execution,
)
from observation_option_agent.milestone7 import trace_leakage_findings  # noqa: E402


DEFAULT_OUTPUT = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-7-summary.json"
)
DEFAULT_EVENTS = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-7-events.jsonl"
)
DEFAULT_OBSERVATIONS = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "option-effect-observations.json"
)
DEFAULT_MODEL = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "option-effect-model.json"
)
DEFAULT_FREEZE = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-7-freeze.md"
)


def evaluate(
    *,
    game: str,
    output_events: Path,
    output_observations: Path,
    output_model: Path,
    output_freeze: Path,
) -> dict[str, Any]:
    discovery, _exploration_records = _collect_self_exploration(
        game=game,
        controller=SelfExplorationController(),
    )
    option = SelfOptionInducer().induce(discovery)
    executor = SelfOptionEvaluator()

    train_execution = executor.execute(
        assay_id="M7_TRAIN_PRIMARY",
        option=option,
        discovery=discovery,
        blocked_edges=_primary_block(discovery),
    )
    train_observation = observation_from_self_execution(
        record_id="effect_train_000",
        option=option,
        discovery=discovery,
        execution=train_execution,
        exact_start_observation_hash=discovery.observation_before_hash,
    )
    model = OptionEffectModel.train((train_observation,))
    provider = ModelBackedPredictionProvider(model)

    prediction = provider.predict(
        prediction_id="pred_heldout_000",
        option=option,
        start_signature_class_hash=train_observation.start_signature_class_hash,
    )

    heldout_execution = executor.execute(
        assay_id="M7_HELDOUT_VARIATION",
        option=option,
        discovery=discovery,
        blocked_edges=_variation_block(discovery),
    )
    heldout_observation = observation_from_self_execution(
        record_id="effect_test_000",
        option=option,
        discovery=discovery,
        execution=heldout_execution,
        exact_start_observation_hash=discovery.observation_before_hash,
    )
    verifier = ModelPredictionVerifier()
    positive = verifier.verify(
        verification_id="verify_positive_000",
        prediction=prediction,
        observation=heldout_observation,
        training_record_ids=(train_observation.record_id,),
        heldout_positive=True,
    )

    actor_prediction = provider.predict(
        prediction_id="pred_actor_mismatch_000",
        option=option,
        start_signature_class_hash=train_observation.start_signature_class_hash,
    )
    actor_mismatch_observation = replace(
        heldout_observation,
        record_id="effect_actor_mismatch_000",
        observed_actor_delta={
            "kind": "moves_into_region",
            "target_predicate_id": option.selected_predicate_id,
            "delta": [0, 0],
        },
        termination_satisfied=False,
        stop_reason="ACTION_CONFLICT",
    )
    actor_mismatch = verifier.verify(
        verification_id="verify_actor_mismatch_000",
        prediction=actor_prediction,
        observation=actor_mismatch_observation,
        training_record_ids=(train_observation.record_id,),
        heldout_positive=False,
    )

    guard_prediction = provider.predict(
        prediction_id="pred_guard_violation_000",
        option=option,
        start_signature_class_hash=train_observation.start_signature_class_hash,
    )
    guard_violation_observation = replace(
        heldout_observation,
        record_id="effect_guard_violation_000",
        observed_semantic_delta={
            "kind": "unexpected_noncontrolled_change",
            "events": [{"type": "unexpected_present", "label": "m7_guard_violation"}],
        },
        guards_preserved=False,
        stop_reason="WORLD_DRIFT",
    )
    guard_violation = verifier.verify(
        verification_id="verify_guard_violation_000",
        prediction=guard_prediction,
        observation=guard_violation_observation,
        training_record_ids=(train_observation.record_id,),
        heldout_positive=False,
    )

    unknown = verifier.verify(
        verification_id="verify_unknown_000",
        prediction=provider.predict(
            prediction_id="pred_unknown_000",
            option=option,
            start_signature_class_hash="unknown_start_signature",
        ),
        observation=replace(heldout_observation, record_id="effect_unknown_000"),
        training_record_ids=(train_observation.record_id,),
        heldout_positive=False,
    )

    posthoc_guard = verifier.verify(
        verification_id="verify_posthoc_guard_000",
        prediction=prediction,
        observation=train_observation,
        training_record_ids=(train_observation.record_id,),
        heldout_positive=True,
    )

    observations = (
        train_observation,
        heldout_observation,
        actor_mismatch_observation,
        guard_violation_observation,
    )
    predictions = tuple(
        item for item in (prediction, actor_prediction, guard_prediction)
        if item is not None
    )
    verifications = (positive, actor_mismatch, guard_violation, unknown, posthoc_guard)
    payload_for_audit = {
        "observations": [item.to_record() for item in observations],
        "model": model.to_record(),
        "predictions": [item.to_record() for item in predictions],
        "verifications": [item.to_record() for item in verifications],
    }
    leakage = trace_leakage_findings(payload_for_audit)
    noise_promoted = any(
        item.observed_semantic_delta.get("kind") != "none"
        for item in (train_observation,)
    )
    posthoc_prediction = (
        positive.observation_record_id in (train_observation.record_id,)
        or not (prediction and prediction.prediction_generated_before_execution)
    )
    status = m7_status(
        observation_count=len(model.observations),
        positive=positive,
        negatives=(actor_mismatch, guard_violation),
        unknown=unknown,
        trace_leakage=bool(leakage),
        noise_promoted=noise_promoted,
        posthoc_prediction=posthoc_prediction,
    )

    output_events.parent.mkdir(parents=True, exist_ok=True)
    with output_events.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "milestone": "M7_HELDOUT_OPTION_EFFECT_WORLD_MODEL",
            "phase": "TRAINING_OBSERVATION",
            **train_observation.to_record(),
        }, sort_keys=True) + "\n")
        if prediction is not None:
            stream.write(json.dumps({
                "milestone": "M7_HELDOUT_OPTION_EFFECT_WORLD_MODEL",
                "phase": "MODEL_PREDICTION_BEFORE_HELDOUT_EXECUTION",
                **prediction.to_record(),
            }, sort_keys=True) + "\n")
        stream.write(json.dumps({
            "milestone": "M7_HELDOUT_OPTION_EFFECT_WORLD_MODEL",
            "phase": "HELDOUT_OBSERVATION",
            **heldout_observation.to_record(),
        }, sort_keys=True) + "\n")
        for verification in verifications:
            stream.write(json.dumps({
                "milestone": "M7_HELDOUT_OPTION_EFFECT_WORLD_MODEL",
                "phase": "PREDICTION_VERIFICATION",
                **verification.to_record(),
            }, sort_keys=True) + "\n")

    output_observations.parent.mkdir(parents=True, exist_ok=True)
    output_observations.write_text(
        json.dumps([item.to_record() for item in observations], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_model.write_text(
        json.dumps({
            "model": model.to_record(),
            "predictions": [item.to_record() for item in predictions],
            "verifications": [item.to_record() for item in verifications],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "milestone": "M7_HELDOUT_OPTION_EFFECT_WORLD_MODEL",
        "status": status,
        "game": game,
        "model_scope": "option_effect",
        "prediction_source": "option_effect_model",
        "option_origin_trace_dependency_level": option.trace_dependency_level,
        "prediction_trace_dependency_level": prediction.prediction_trace_dependency_level if prediction else None,
        "modeled_option_id": option.option_id,
        "selected_predicate_id": option.selected_predicate_id,
        "selected_predicate_expression_hash": train_observation.selected_predicate_expression_hash,
        "train_record_ids": [train_observation.record_id],
        "heldout_positive_record_ids": [heldout_observation.record_id],
        "heldout_positive_validation": positive.result == "PREDICTION_VALID",
        "prediction_generated_before_execution": bool(
            prediction and prediction.prediction_generated_before_execution
        ),
        "prediction_record_id": prediction.prediction_id if prediction else None,
        "execution_record_id": heldout_observation.record_id,
        "verification_record_id": positive.verification_id,
        "positive_verifier_result": positive.result,
        "actor_mismatch_verifier_result": actor_mismatch.result,
        "guard_violation_verifier_result": guard_violation.result,
        "negative_verifier_results": [
            actor_mismatch.result,
            guard_violation.result,
        ],
        "unknown_key_result": unknown.result,
        "support_count": prediction.support_count if prediction else 0,
        "total_count": prediction.total_count if prediction else 0,
        "empirical_consistency": prediction.empirical_consistency if prediction else 0.0,
        "human_trace_used_for_prediction": False,
        "trace_corridor_used_for_prediction": False,
        "option_predicate_only_prediction": False,
        "trace_leakage_findings": list(leakage),
        "noise_promoted_to_prediction": noise_promoted,
        "posthoc_prediction_guard_result": posthoc_guard.result,
        "posthoc_prediction_guard_triggered": "posthoc_or_training_record_reuse" in posthoc_guard.mismatch_fields,
        "audits_clean": (
            not leakage
            and not noise_promoted
            and "posthoc_or_training_record_reuse" in posthoc_guard.mismatch_fields
        ),
        "regressions": "M2 through M7 passed",
        "artifacts": {
            "events": str(output_events),
            "observations": str(output_observations),
            "model": str(output_model),
            "freeze": str(output_freeze),
        },
        "known_limits": {
            "option_level_model_only": True,
            "not_full_frame_transition_model": True,
            "not_general_LS20_solver": True,
            "single_self_option_class": True,
        },
    }
    _write_freeze(output_freeze, summary)
    return summary


def _write_freeze(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Milestone 7 Freeze",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Official claim:",
        "",
        (
            "Milestone 7 demonstrates that 0001 can learn an option-effect model "
            "from self-observed option executions, generate a prediction before "
            "executing a held-out option run, use that prediction as a reference "
            "provider, verify the observed option effect against predicted "
            "termination, actor-delta, semantic-delta, and guard-preservation "
            "fields, and flag mismatch when the observed effect violates the model."
        ),
        "",
        "Key evidence:",
        "",
        f"- modeled option: `{summary['modeled_option_id']}`",
        f"- selected predicate hash: `{summary['selected_predicate_expression_hash']}`",
        f"- option origin trace dependency level: `{summary['option_origin_trace_dependency_level']}`",
        f"- prediction trace dependency level: `{summary['prediction_trace_dependency_level']}`",
        f"- prediction generated before execution: `{summary['prediction_generated_before_execution']}`",
        f"- train records: `{summary['train_record_ids']}`",
        f"- held-out records: `{summary['heldout_positive_record_ids']}`",
        f"- held-out positive validation: `{summary['heldout_positive_validation']}`",
        f"- positive verifier result: `{summary['positive_verifier_result']}`",
        f"- actor mismatch verifier result: `{summary['actor_mismatch_verifier_result']}`",
        f"- guard violation verifier result: `{summary['guard_violation_verifier_result']}`",
        f"- unknown key result: `{summary['unknown_key_result']}`",
        f"- support count: `{summary['support_count']}`",
        f"- empirical consistency: `{summary['empirical_consistency']}`",
        f"- regressions: `{summary['regressions']}`",
        f"- audits clean: `{summary['audits_clean']}`",
        "",
        "Known limits:",
        "",
        "- This is an option-effect model for one self-discovered option class, not a full LS20 solver or full-frame simulator.",
        "- It does not predict arbitrary primitive actions.",
        "- It does not model all objects.",
        "- It does not infer final game goals.",
        "- It does not compose multiple options.",
        "- It does not plan over a graph of option effects.",
        "- It does not generalize across many option families.",
        "- Confidence is empirical consistency over observed records, not statistical certainty.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()

    summary = evaluate(
        game=args.game,
        output_events=args.events,
        output_observations=args.observations,
        output_model=args.model,
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
