"""Option-effect world model for 0001 milestone 7."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal

from .events import Delta
from .milestone4 import OptionCandidate, selected_predicate_expression_hash
from .milestone6 import DiscoveryRecord, SelfOptionExecution


VerificationResult = Literal[
    "PREDICTION_VALID",
    "MODEL_PREDICTION_MISMATCH",
    "GUARD_VIOLATION",
    "UNKNOWN",
]
M7Status = Literal[
    "complete_with_known_limits",
    "partial_effect_record_only",
    "partial_unvalidated_prediction",
    "failed_posthoc_prediction",
    "failed_trace_leakage",
    "failed_noise_promotion",
    "failed_no_mismatch_detection",
]


@dataclass(frozen=True)
class OptionEffectObservation:
    record_id: str
    option_id: str
    selected_predicate_id: str
    selected_predicate_expression_hash: str
    start_signature_class_hash: str
    start_signature_class: dict[str, Any]
    exact_start_observation_hash: str
    action_plan: tuple[int, ...]
    observed_actor_delta: dict[str, Any]
    observed_semantic_delta: dict[str, Any]
    termination_satisfied: bool
    guards_preserved: bool
    stop_reason: str
    provenance: tuple[dict[str, Any], ...]

    def key(self) -> tuple[str, str, str]:
        return (
            self.option_id,
            self.selected_predicate_expression_hash,
            self.start_signature_class_hash,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "option_id": self.option_id,
            "selected_predicate_id": self.selected_predicate_id,
            "selected_predicate_expression_hash": self.selected_predicate_expression_hash,
            "start_signature_class_hash": self.start_signature_class_hash,
            "start_signature_class": self.start_signature_class,
            "exact_start_observation_hash": self.exact_start_observation_hash,
            "action_plan": list(self.action_plan),
            "observed_actor_delta": self.observed_actor_delta,
            "observed_semantic_delta": self.observed_semantic_delta,
            "termination_satisfied": self.termination_satisfied,
            "guards_preserved": self.guards_preserved,
            "stop_reason": self.stop_reason,
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class OptionEffectPrediction:
    prediction_id: str
    model_id: str
    option_id: str
    selected_predicate_id: str
    selected_predicate_expression_hash: str
    start_signature_class_hash: str
    predicted_termination_satisfied: bool
    predicted_actor_delta: dict[str, Any]
    predicted_semantic_delta: dict[str, Any]
    predicted_guards_preserved: bool
    predicted_stop_reason: str
    empirical_consistency: float
    support_count: int
    total_count: int
    prediction_source: str = "option_effect_model"
    prediction_trace_dependency_level: int = 5
    prediction_generated_before_execution: bool = True

    def to_record(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "model_id": self.model_id,
            "option_id": self.option_id,
            "selected_predicate_id": self.selected_predicate_id,
            "selected_predicate_expression_hash": self.selected_predicate_expression_hash,
            "start_signature_class_hash": self.start_signature_class_hash,
            "predicted_termination_satisfied": self.predicted_termination_satisfied,
            "predicted_actor_delta": self.predicted_actor_delta,
            "predicted_semantic_delta": self.predicted_semantic_delta,
            "predicted_guards_preserved": self.predicted_guards_preserved,
            "predicted_stop_reason": self.predicted_stop_reason,
            "empirical_consistency": self.empirical_consistency,
            "support_count": self.support_count,
            "total_count": self.total_count,
            "confidence_interpretation": (
                "empirical consistency on observed records, not statistical certainty"
            ),
            "prediction_source": self.prediction_source,
            "prediction_trace_dependency_level": self.prediction_trace_dependency_level,
            "prediction_generated_before_execution": self.prediction_generated_before_execution,
        }


@dataclass(frozen=True)
class OptionEffectVerification:
    verification_id: str
    prediction_id: str | None
    observation_record_id: str
    result: VerificationResult
    mismatch_fields: tuple[str, ...]
    prediction_generated_before_execution: bool
    heldout_positive: bool
    training_record_ids: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "verification_id": self.verification_id,
            "prediction_id": self.prediction_id,
            "observation_record_id": self.observation_record_id,
            "result": self.result,
            "mismatch_fields": list(self.mismatch_fields),
            "prediction_generated_before_execution": self.prediction_generated_before_execution,
            "heldout_positive": self.heldout_positive,
            "training_record_ids": list(self.training_record_ids),
        }


class OptionEffectModel:
    def __init__(
        self,
        *,
        model_id: str = "option_effect_model_000",
        observations: tuple[OptionEffectObservation, ...] = (),
    ) -> None:
        self.model_id = model_id
        self._observations = observations
        self._groups: dict[
            tuple[str, str, str],
            tuple[OptionEffectObservation, ...],
        ] = defaultdict(tuple)
        grouped: dict[
            tuple[str, str, str],
            list[OptionEffectObservation],
        ] = defaultdict(list)
        for observation in observations:
            grouped[observation.key()].append(observation)
        self._groups = {
            key: tuple(values)
            for key, values in grouped.items()
        }

    @classmethod
    def train(
        cls,
        observations: tuple[OptionEffectObservation, ...],
        *,
        model_id: str = "option_effect_model_000",
    ) -> "OptionEffectModel":
        return cls(model_id=model_id, observations=observations)

    @property
    def observations(self) -> tuple[OptionEffectObservation, ...]:
        return self._observations

    def predict(
        self,
        *,
        prediction_id: str,
        option: OptionCandidate,
        start_signature_class_hash: str,
    ) -> OptionEffectPrediction | None:
        key = (
            option.option_id,
            selected_predicate_expression_hash(option),
            start_signature_class_hash,
        )
        records = self._groups.get(key, ())
        if not records:
            return None
        prototype = records[0]
        matching = tuple(
            record for record in records
            if _same_effect(record, prototype)
        )
        return OptionEffectPrediction(
            prediction_id=prediction_id,
            model_id=self.model_id,
            option_id=option.option_id,
            selected_predicate_id=option.selected_predicate_id or "",
            selected_predicate_expression_hash=selected_predicate_expression_hash(option),
            start_signature_class_hash=start_signature_class_hash,
            predicted_termination_satisfied=prototype.termination_satisfied,
            predicted_actor_delta=prototype.observed_actor_delta,
            predicted_semantic_delta=prototype.observed_semantic_delta,
            predicted_guards_preserved=prototype.guards_preserved,
            predicted_stop_reason=prototype.stop_reason,
            empirical_consistency=len(matching) / len(records),
            support_count=len(matching),
            total_count=len(records),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_scope": "option_effect",
            "observation_count": len(self._observations),
            "keys": [
                {
                    "option_id": key[0],
                    "selected_predicate_expression_hash": key[1],
                    "start_signature_class_hash": key[2],
                    "record_count": len(records),
                }
                for key, records in sorted(self._groups.items())
            ],
            "prediction_source": "option_effect_model",
        }


class ModelPredictionVerifier:
    def verify(
        self,
        *,
        verification_id: str,
        prediction: OptionEffectPrediction | None,
        observation: OptionEffectObservation,
        training_record_ids: tuple[str, ...],
        heldout_positive: bool,
    ) -> OptionEffectVerification:
        if prediction is None:
            return OptionEffectVerification(
                verification_id=verification_id,
                prediction_id=None,
                observation_record_id=observation.record_id,
                result="UNKNOWN",
                mismatch_fields=("prediction",),
                prediction_generated_before_execution=True,
                heldout_positive=heldout_positive,
                training_record_ids=training_record_ids,
            )
        if observation.record_id in training_record_ids:
            return OptionEffectVerification(
                verification_id=verification_id,
                prediction_id=prediction.prediction_id,
                observation_record_id=observation.record_id,
                result="UNKNOWN",
                mismatch_fields=("posthoc_or_training_record_reuse",),
                prediction_generated_before_execution=prediction.prediction_generated_before_execution,
                heldout_positive=False,
                training_record_ids=training_record_ids,
            )

        mismatch_fields = []
        if prediction.predicted_termination_satisfied != observation.termination_satisfied:
            mismatch_fields.append("termination_satisfied")
        if prediction.predicted_actor_delta != observation.observed_actor_delta:
            mismatch_fields.append("actor_delta")
        if prediction.predicted_semantic_delta != observation.observed_semantic_delta:
            mismatch_fields.append("semantic_delta")
        if prediction.predicted_guards_preserved != observation.guards_preserved:
            mismatch_fields.append("guards_preserved")
        if prediction.predicted_stop_reason != observation.stop_reason:
            mismatch_fields.append("stop_reason")

        if not mismatch_fields:
            result: VerificationResult = "PREDICTION_VALID"
        elif "guards_preserved" in mismatch_fields or "semantic_delta" in mismatch_fields:
            result = "GUARD_VIOLATION"
        else:
            result = "MODEL_PREDICTION_MISMATCH"
        return OptionEffectVerification(
            verification_id=verification_id,
            prediction_id=prediction.prediction_id,
            observation_record_id=observation.record_id,
            result=result,
            mismatch_fields=tuple(mismatch_fields),
            prediction_generated_before_execution=prediction.prediction_generated_before_execution,
            heldout_positive=heldout_positive,
            training_record_ids=training_record_ids,
        )


class ModelBackedPredictionProvider:
    def __init__(self, model: OptionEffectModel) -> None:
        self.model = model
        self.source = "option_effect_model"

    def predict(
        self,
        *,
        prediction_id: str,
        option: OptionCandidate,
        start_signature_class_hash: str,
    ) -> OptionEffectPrediction | None:
        return self.model.predict(
            prediction_id=prediction_id,
            option=option,
            start_signature_class_hash=start_signature_class_hash,
        )


def start_signature_class_for_option(option: OptionCandidate) -> dict[str, Any]:
    return {
        "kind": "option_applicability_signature",
        "initiation_condition": True,
        "termination_initially_true": False,
        "semantic_guards": True,
        "controlled_entity_region": "valid_start_region",
        "option_id": option.option_id,
        "selected_predicate_id": option.selected_predicate_id,
        "selected_predicate_expression_hash": selected_predicate_expression_hash(option),
    }


def start_signature_class_hash(signature: dict[str, Any]) -> str:
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def observation_from_self_execution(
    *,
    record_id: str,
    option: OptionCandidate,
    discovery: DiscoveryRecord,
    execution: SelfOptionExecution,
    exact_start_observation_hash: str,
) -> OptionEffectObservation:
    signature = start_signature_class_for_option(option)
    semantic_delta = _semantic_delta_from_execution(execution)
    guards_preserved = semantic_delta["kind"] == "none"
    return OptionEffectObservation(
        record_id=record_id,
        option_id=option.option_id,
        selected_predicate_id=option.selected_predicate_id or "",
        selected_predicate_expression_hash=selected_predicate_expression_hash(option),
        start_signature_class_hash=start_signature_class_hash(signature),
        start_signature_class=signature,
        exact_start_observation_hash=exact_start_observation_hash,
        action_plan=execution.plan,
        observed_actor_delta={
            "kind": "moves_into_region",
            "target_predicate_id": option.selected_predicate_id,
            "delta": list(_actor_delta_from_execution(execution)),
        },
        observed_semantic_delta=semantic_delta,
        termination_satisfied=execution.termination_predicate_satisfied,
        guards_preserved=guards_preserved,
        stop_reason=(
            "TERMINATION_PREDICATE_SATISFIED"
            if execution.completed
            else execution.failure_reason or execution.status
        ),
        provenance=(
            {
                "kind": "self_observed_option_execution",
                "discovery_record_id": discovery.record_id,
                "execution_assay_id": execution.assay_id,
            },
        ),
    )


def m7_status(
    *,
    observation_count: int,
    positive: OptionEffectVerification,
    negatives: tuple[OptionEffectVerification, ...],
    unknown: OptionEffectVerification,
    trace_leakage: bool,
    noise_promoted: bool,
    posthoc_prediction: bool,
) -> M7Status:
    if trace_leakage:
        return "failed_trace_leakage"
    if noise_promoted:
        return "failed_noise_promotion"
    if posthoc_prediction:
        return "failed_posthoc_prediction"
    if observation_count == 0:
        return "partial_effect_record_only"
    if positive.result != "PREDICTION_VALID":
        return "partial_unvalidated_prediction"
    if all(result.result == "PREDICTION_VALID" for result in negatives):
        return "failed_no_mismatch_detection"
    if unknown.result != "UNKNOWN":
        return "failed_no_mismatch_detection"
    return "complete_with_known_limits"


def trace_leakage_findings(payload: dict[str, Any]) -> tuple[str, ...]:
    encoded = repr(payload)
    forbidden = (
        "human_trace",
        "public_trace_segment",
        "trace_corridor",
        "human_action",
        "raw_action",
        "action_at_index",
    )
    return tuple(marker for marker in forbidden if marker in encoded)


def _actor_delta_from_execution(execution: SelfOptionExecution) -> Delta:
    dx = 0
    dy = 0
    for record in execution.records:
        delta = record.get("observed_delta", (0, 0))
        dx += int(delta[0])
        dy += int(delta[1])
    return (dx, dy)


def _semantic_delta_from_execution(execution: SelfOptionExecution) -> dict[str, Any]:
    world_diffs = [
        item
        for record in execution.records
        for item in record.get("world_diff", ())
    ]
    if not world_diffs:
        return {"kind": "none", "events": []}
    return {"kind": "unexpected_noncontrolled_change", "events": world_diffs}


def _same_effect(
    left: OptionEffectObservation,
    right: OptionEffectObservation,
) -> bool:
    return (
        left.observed_actor_delta == right.observed_actor_delta
        and left.observed_semantic_delta == right.observed_semantic_delta
        and left.termination_satisfied == right.termination_satisfied
        and left.guards_preserved == right.guards_preserved
        and left.stop_reason == right.stop_reason
    )
