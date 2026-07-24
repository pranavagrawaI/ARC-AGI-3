"""Option-applicability chain prediction for 0001 milestone 8."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal

from .milestone4 import OptionCandidate
from .milestone7 import (
    ModelPredictionVerifier,
    OptionEffectObservation,
    OptionEffectPrediction,
    OptionEffectVerification,
)


ChainResult = Literal[
    "CHAIN_VERIFIED",
    "LINK_EFFECT_MISMATCH",
    "NEXT_OPTION_NOT_APPLICABLE",
    "UNKNOWN_CHAIN",
]
M8Status = Literal[
    "complete_with_known_limits",
    "partial_graph_contract_only",
    "failed_posthoc_link_prediction",
    "failed_effect_mismatch_gate",
    "failed_next_applicability_gate",
    "failed_same_option_state_collapse",
]


@dataclass(frozen=True)
class OptionApplicabilityStateClass:
    state_class_id: str
    controlled_entity_region: str
    semantic_guards: bool
    termination_predicates: dict[str, bool]
    applicable_options: tuple[str, ...]
    non_controlled_signature_class: str
    exact_observation_hashes: tuple[str, ...] = ()

    def signature(self) -> dict[str, Any]:
        return {
            "kind": "option_applicability_signature",
            "controlled_entity_region": self.controlled_entity_region,
            "semantic_guards": self.semantic_guards,
            "termination_predicates": self.termination_predicates,
            "applicable_options": list(self.applicable_options),
            "non_controlled_signature_class": self.non_controlled_signature_class,
        }

    @property
    def signature_hash(self) -> str:
        encoded = json.dumps(self.signature(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]

    def allows(self, option_id: str) -> bool:
        return option_id in self.applicable_options and self.semantic_guards

    def to_record(self) -> dict[str, Any]:
        return {
            "state_class_id": self.state_class_id,
            **self.signature(),
            "signature_hash": self.signature_hash,
            "exact_observation_hashes_debug_only": list(self.exact_observation_hashes),
        }


@dataclass(frozen=True)
class OptionApplicabilityEdge:
    edge_id: str
    source_state_class_id: str
    option_id: str
    predicted_next_state_class_id: str
    enabled_options: tuple[str, ...]
    option_prediction: OptionEffectPrediction

    def to_record(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_state_class_id": self.source_state_class_id,
            "option_id": self.option_id,
            "predicted_next_state_class_id": self.predicted_next_state_class_id,
            "enabled_options": list(self.enabled_options),
            "option_prediction": self.option_prediction.to_record(),
        }


@dataclass(frozen=True)
class OptionChainLinkPrediction:
    link_index: int
    edge_id: str
    source_state_class_id: str
    option_id: str
    predicted_next_state_class_id: str
    enabled_options: tuple[str, ...]
    next_option_id: str | None
    option_prediction: OptionEffectPrediction

    def to_record(self) -> dict[str, Any]:
        return {
            "link_index": self.link_index,
            "edge_id": self.edge_id,
            "source_state_class_id": self.source_state_class_id,
            "option_id": self.option_id,
            "predicted_next_state_class_id": self.predicted_next_state_class_id,
            "enabled_options": list(self.enabled_options),
            "next_option_id": self.next_option_id,
            "option_prediction": self.option_prediction.to_record(),
        }


@dataclass(frozen=True)
class OptionChainPrediction:
    prediction_id: str
    graph_id: str
    start_state_class_id: str
    links: tuple[OptionChainLinkPrediction, ...]
    prediction_source: str = "option_applicability_graph"
    prediction_generated_before_execution: bool = True

    def to_record(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "graph_id": self.graph_id,
            "start_state_class_id": self.start_state_class_id,
            "links": [link.to_record() for link in self.links],
            "prediction_source": self.prediction_source,
            "prediction_generated_before_execution": self.prediction_generated_before_execution,
        }


@dataclass(frozen=True)
class OptionChainLinkVerification:
    link_index: int
    option_id: str
    observed_effect_record_id: str
    effect_verification: OptionEffectVerification
    predicted_next_state_class_id: str
    observed_next_state_class_id: str | None
    next_option_id: str | None
    next_option_applicable: bool
    next_link_released: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "link_index": self.link_index,
            "option_id": self.option_id,
            "observed_effect_record_id": self.observed_effect_record_id,
            "effect_verification": self.effect_verification.to_record(),
            "predicted_next_state_class_id": self.predicted_next_state_class_id,
            "observed_next_state_class_id": self.observed_next_state_class_id,
            "next_option_id": self.next_option_id,
            "next_option_applicable": self.next_option_applicable,
            "next_link_released": self.next_link_released,
        }


@dataclass(frozen=True)
class OptionChainVerification:
    verification_id: str
    prediction_id: str | None
    result: ChainResult
    executed_link_count: int
    link_verifications: tuple[OptionChainLinkVerification, ...]
    stopped_at_link: int | None
    stop_reason: str | None

    def to_record(self) -> dict[str, Any]:
        return {
            "verification_id": self.verification_id,
            "prediction_id": self.prediction_id,
            "result": self.result,
            "executed_link_count": self.executed_link_count,
            "link_verifications": [
                verification.to_record()
                for verification in self.link_verifications
            ],
            "stopped_at_link": self.stopped_at_link,
            "stop_reason": self.stop_reason,
        }


class OptionApplicabilityGraph:
    def __init__(
        self,
        *,
        graph_id: str = "option_applicability_graph_000",
        state_classes: tuple[OptionApplicabilityStateClass, ...],
        edges: tuple[OptionApplicabilityEdge, ...],
    ) -> None:
        self.graph_id = graph_id
        self.state_classes = state_classes
        self.edges = edges
        self._states = {state.state_class_id: state for state in state_classes}
        self._edges = {
            (edge.source_state_class_id, edge.option_id): edge
            for edge in edges
        }

    def state(self, state_class_id: str) -> OptionApplicabilityStateClass | None:
        return self._states.get(state_class_id)

    def predict_chain(
        self,
        *,
        prediction_id: str,
        start_state_class_id: str,
        options: tuple[OptionCandidate, ...],
    ) -> OptionChainPrediction | None:
        current_state_id = start_state_class_id
        links: list[OptionChainLinkPrediction] = []
        for index, option in enumerate(options):
            state = self._states.get(current_state_id)
            if state is None or not state.allows(option.option_id):
                return None
            edge = self._edges.get((current_state_id, option.option_id))
            if edge is None:
                return None
            next_option_id = (
                options[index + 1].option_id
                if index + 1 < len(options)
                else None
            )
            links.append(OptionChainLinkPrediction(
                link_index=index,
                edge_id=edge.edge_id,
                source_state_class_id=current_state_id,
                option_id=option.option_id,
                predicted_next_state_class_id=edge.predicted_next_state_class_id,
                enabled_options=edge.enabled_options,
                next_option_id=next_option_id,
                option_prediction=edge.option_prediction,
            ))
            current_state_id = edge.predicted_next_state_class_id
        return OptionChainPrediction(
            prediction_id=prediction_id,
            graph_id=self.graph_id,
            start_state_class_id=start_state_class_id,
            links=tuple(links),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_kind": "option_applicability_state_class",
            "edge_kind": "option_effect_prediction",
            "state_classes": [state.to_record() for state in self.state_classes],
            "edges": [edge.to_record() for edge in self.edges],
        }


class OptionChainVerifier:
    def __init__(self, *, effect_verifier: ModelPredictionVerifier | None = None) -> None:
        self.effect_verifier = effect_verifier or ModelPredictionVerifier()

    def verify(
        self,
        *,
        verification_id: str,
        prediction: OptionChainPrediction | None,
        observed_effects: tuple[OptionEffectObservation, ...],
        observed_next_state_classes: tuple[OptionApplicabilityStateClass, ...],
        training_record_ids_by_link: tuple[tuple[str, ...], ...],
    ) -> OptionChainVerification:
        if prediction is None:
            return OptionChainVerification(
                verification_id=verification_id,
                prediction_id=None,
                result="UNKNOWN_CHAIN",
                executed_link_count=0,
                link_verifications=(),
                stopped_at_link=None,
                stop_reason="missing_chain_prediction",
            )

        link_verifications: list[OptionChainLinkVerification] = []
        for link in prediction.links:
            if link.link_index >= len(observed_effects):
                return _chain_result(
                    verification_id=verification_id,
                    prediction=prediction,
                    result="UNKNOWN_CHAIN",
                    executed_link_count=len(link_verifications),
                    link_verifications=tuple(link_verifications),
                    stopped_at_link=link.link_index,
                    stop_reason="missing_observed_effect",
                )
            effect = observed_effects[link.link_index]
            training_ids = (
                training_record_ids_by_link[link.link_index]
                if link.link_index < len(training_record_ids_by_link)
                else ()
            )
            effect_verification = self.effect_verifier.verify(
                verification_id=f"{verification_id}:effect:{link.link_index}",
                prediction=link.option_prediction,
                observation=effect,
                training_record_ids=training_ids,
                heldout_positive=True,
            )
            observed_state = (
                observed_next_state_classes[link.link_index]
                if link.link_index < len(observed_next_state_classes)
                else None
            )
            next_option_applicable = _next_option_applicable(
                link=link,
                observed_state=observed_state,
            )
            next_link_released = (
                effect_verification.result == "PREDICTION_VALID"
                and next_option_applicable
            )
            link_verification = OptionChainLinkVerification(
                link_index=link.link_index,
                option_id=link.option_id,
                observed_effect_record_id=effect.record_id,
                effect_verification=effect_verification,
                predicted_next_state_class_id=link.predicted_next_state_class_id,
                observed_next_state_class_id=(
                    observed_state.state_class_id
                    if observed_state is not None
                    else None
                ),
                next_option_id=link.next_option_id,
                next_option_applicable=next_option_applicable,
                next_link_released=next_link_released,
            )
            link_verifications.append(link_verification)

            if effect_verification.result != "PREDICTION_VALID":
                return _chain_result(
                    verification_id=verification_id,
                    prediction=prediction,
                    result="LINK_EFFECT_MISMATCH",
                    executed_link_count=len(link_verifications),
                    link_verifications=tuple(link_verifications),
                    stopped_at_link=link.link_index,
                    stop_reason=effect_verification.result,
                )
            if not next_option_applicable:
                return _chain_result(
                    verification_id=verification_id,
                    prediction=prediction,
                    result="NEXT_OPTION_NOT_APPLICABLE",
                    executed_link_count=len(link_verifications),
                    link_verifications=tuple(link_verifications),
                    stopped_at_link=link.link_index,
                    stop_reason="next_option_applicability_not_established",
                )

        return _chain_result(
            verification_id=verification_id,
            prediction=prediction,
            result="CHAIN_VERIFIED",
            executed_link_count=len(link_verifications),
            link_verifications=tuple(link_verifications),
            stopped_at_link=None,
            stop_reason=None,
        )


def same_option_chain_is_honest(
    *,
    chain_prediction: OptionChainPrediction,
    observed_state_classes: tuple[OptionApplicabilityStateClass, ...],
) -> bool:
    option_ids = [link.option_id for link in chain_prediction.links]
    if len(set(option_ids)) != 1:
        return True
    state_ids = [chain_prediction.start_state_class_id] + [
        state.state_class_id
        for state in observed_state_classes[:len(chain_prediction.links)]
    ]
    return (
        len(state_ids) == len(set(state_ids))
        and all(
            link.next_option_id is None
            or state.allows(link.next_option_id)
            for link, state in zip(chain_prediction.links, observed_state_classes)
        )
    )


def m8_status(
    *,
    chain_prediction: OptionChainPrediction | None,
    chain_verification: OptionChainVerification,
    effect_mismatch_gate_exercised: bool,
    next_applicability_gate_exercised: bool,
    same_option_family_chain: bool,
    distinct_state_classes: bool,
    posthoc_link_prediction: bool,
) -> M8Status:
    if posthoc_link_prediction:
        return "failed_posthoc_link_prediction"
    if chain_prediction is None:
        return "partial_graph_contract_only"
    if not effect_mismatch_gate_exercised:
        return "failed_effect_mismatch_gate"
    if not next_applicability_gate_exercised:
        return "failed_next_applicability_gate"
    if same_option_family_chain and not distinct_state_classes:
        return "failed_same_option_state_collapse"
    if chain_verification.result != "CHAIN_VERIFIED":
        return "partial_graph_contract_only"
    return "complete_with_known_limits"


def _next_option_applicable(
    *,
    link: OptionChainLinkPrediction,
    observed_state: OptionApplicabilityStateClass | None,
) -> bool:
    if observed_state is None:
        return False
    if observed_state.state_class_id != link.predicted_next_state_class_id:
        return False
    if link.next_option_id is None:
        return True
    return (
        link.next_option_id in link.enabled_options
        and observed_state.allows(link.next_option_id)
    )


def _chain_result(
    *,
    verification_id: str,
    prediction: OptionChainPrediction,
    result: ChainResult,
    executed_link_count: int,
    link_verifications: tuple[OptionChainLinkVerification, ...],
    stopped_at_link: int | None,
    stop_reason: str | None,
) -> OptionChainVerification:
    return OptionChainVerification(
        verification_id=verification_id,
        prediction_id=prediction.prediction_id,
        result=result,
        executed_link_count=executed_link_count,
        link_verifications=link_verifications,
        stopped_at_link=stopped_at_link,
        stop_reason=stop_reason,
    )
