"""Feature-level progress learning for 0001 milestone 10."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from .milestone9 import AttemptMemory, ChainCandidate


FEATURE_PRIORS: dict[str, float] = {
    "level_completion": 1000.0,
    "stable_semantic_delta": 50.0,
    "new_option_enabled": 25.0,
    "new_state_class": 15.0,
    "uncertainty_reduction": 10.0,
    "volatile_visual_delta": 0.0,
}

UTILITY_WEIGHTS: dict[str, float] = {
    "level_completed": 100.0,
    "new_option_enabled": 2.0,
    "new_state_class": 1.5,
    "prediction_validated": 1.5,
    "stable_semantic_delta": 4.0,
    "guard_violation": -2.0,
    "action_conflict_or_next_not_applicable": -2.0,
}

M10Status = Literal[
    "complete_with_known_limits",
    "partial_progress_records_only",
    "partial_feature_update_without_selection_effect",
    "partial_selection_effect_candidate_memory_only",
    "failed_no_feature_generalization",
    "failed_outcome_leakage_into_selection",
    "failed_noise_as_progress",
    "failed_candidate_memory_confused_with_progress_learning",
    "failed_trace_leakage",
    "failed_oracle_goal_signal",
]


@dataclass(frozen=True)
class ProgressFeatureHypothesis:
    feature_id: str
    kind: str
    prior_weight: float
    posterior_weight: float
    support_count: int
    positive_outcomes: int
    negative_outcomes: int
    mean_observed_utility: float
    confidence: float
    status: str
    evidence: tuple[dict[str, Any], ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "kind": self.kind,
            "prior_weight": self.prior_weight,
            "posterior_weight": self.posterior_weight,
            "support_count": self.support_count,
            "positive_outcomes": self.positive_outcomes,
            "negative_outcomes": self.negative_outcomes,
            "mean_observed_utility": self.mean_observed_utility,
            "confidence": self.confidence,
            "status": self.status,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class ProgressOutcomeRecord:
    outcome_id: str
    candidate_id: str
    predicted_features: tuple[str, ...]
    observed_features: dict[str, bool]
    observed_utility_terms: dict[str, float]
    observed_utility: float
    downstream_effects: dict[str, Any]
    stop_reason: str
    provenance: tuple[dict[str, Any], ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "outcome_id": self.outcome_id,
            "candidate_id": self.candidate_id,
            "predicted_features": list(self.predicted_features),
            "observed_features": self.observed_features,
            "observed_utility_terms": self.observed_utility_terms,
            "observed_utility": self.observed_utility,
            "downstream_effects": self.downstream_effects,
            "stop_reason": self.stop_reason,
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class ProgressWeightUpdate:
    update_id: str
    feature_id: str
    prior_weight: float
    posterior_weight: float
    learned_weight_delta: float
    evidence_outcome_ids: tuple[str, ...]
    update_reason: str
    support_count_before: int
    support_count_after: int

    def to_record(self) -> dict[str, Any]:
        return {
            "update_id": self.update_id,
            "feature_id": self.feature_id,
            "prior_weight": self.prior_weight,
            "posterior_weight": self.posterior_weight,
            "learned_weight_delta": self.learned_weight_delta,
            "evidence_outcome_ids": list(self.evidence_outcome_ids),
            "update_reason": self.update_reason,
            "support_count_before": self.support_count_before,
            "support_count_after": self.support_count_after,
        }


@dataclass(frozen=True)
class ProgressLearningLedger:
    outcomes: tuple[ProgressOutcomeRecord, ...]
    feature_updates: tuple[ProgressWeightUpdate, ...]
    candidate_memory_updates: tuple[dict[str, Any], ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "outcomes": [outcome.to_record() for outcome in self.outcomes],
            "feature_updates": [update.to_record() for update in self.feature_updates],
            "candidate_memory_updates": list(self.candidate_memory_updates),
            "candidate_memory_separate_from_feature_learning": True,
        }


class ProgressHypothesisSet:
    def __init__(
        self,
        hypotheses: tuple[ProgressFeatureHypothesis, ...] | None = None,
    ) -> None:
        self._hypotheses = {
            item.feature_id: item
            for item in (
                hypotheses
                if hypotheses is not None
                else tuple(
                    ProgressFeatureHypothesis(
                        feature_id=feature_id,
                        kind="public_progress_signal",
                        prior_weight=weight,
                        posterior_weight=weight,
                        support_count=0,
                        positive_outcomes=0,
                        negative_outcomes=0,
                        mean_observed_utility=0.0,
                        confidence=0.0,
                        status="NEUTRAL",
                        evidence=(),
                    )
                    for feature_id, weight in sorted(FEATURE_PRIORS.items())
                )
            )
        }

    @classmethod
    def from_m9_priors(cls) -> "ProgressHypothesisSet":
        return cls()

    def hypothesis(self, feature_id: str) -> ProgressFeatureHypothesis:
        return self._hypotheses[feature_id]

    def hypotheses(self) -> tuple[ProgressFeatureHypothesis, ...]:
        return tuple(self._hypotheses[key] for key in sorted(self._hypotheses))

    def update_from_outcomes(
        self,
        outcomes: tuple[ProgressOutcomeRecord, ...],
    ) -> tuple["ProgressHypothesisSet", tuple[ProgressWeightUpdate, ...]]:
        by_feature: dict[str, list[ProgressOutcomeRecord]] = {
            feature_id: [] for feature_id in self._hypotheses
        }
        for outcome in outcomes:
            for feature_id in outcome.predicted_features:
                if feature_id in by_feature:
                    by_feature[feature_id].append(outcome)

        next_hypotheses: list[ProgressFeatureHypothesis] = []
        updates: list[ProgressWeightUpdate] = []
        for feature_id in sorted(self._hypotheses):
            current = self._hypotheses[feature_id]
            evidence = by_feature[feature_id]
            if not evidence:
                next_hypotheses.append(current)
                continue

            utilities = [item.observed_utility for item in evidence]
            mean_utility = sum(utilities) / len(utilities)
            learned_delta = _clamp(2.0 * mean_utility, -10.0, 10.0)
            posterior = current.prior_weight + learned_delta
            positive = sum(1 for value in utilities if value > 0)
            negative = sum(1 for value in utilities if value <= 0)
            status = "NEUTRAL"
            if posterior > current.prior_weight and evidence:
                status = "ACTIVE"
            if posterior < 0 or (
                feature_id == "volatile_visual_delta" and mean_utility <= 0
            ):
                status = "SUPPRESSED"
            support_after = current.support_count + len(evidence)
            next_item = replace(
                current,
                posterior_weight=posterior,
                support_count=support_after,
                positive_outcomes=current.positive_outcomes + positive,
                negative_outcomes=current.negative_outcomes + negative,
                mean_observed_utility=mean_utility,
                confidence=round(support_after / (support_after + 1), 4),
                status=status,
                evidence=current.evidence + tuple(
                    {
                        "kind": "progress_outcome_evidence",
                        "outcome_id": item.outcome_id,
                        "candidate_id": item.candidate_id,
                        "observed_utility": item.observed_utility,
                    }
                    for item in evidence
                ),
            )
            next_hypotheses.append(next_item)
            updates.append(
                ProgressWeightUpdate(
                    update_id=f"update_{feature_id}",
                    feature_id=feature_id,
                    prior_weight=current.prior_weight,
                    posterior_weight=posterior,
                    learned_weight_delta=learned_delta,
                    evidence_outcome_ids=tuple(item.outcome_id for item in evidence),
                    update_reason="deterministic_mean_public_utility_update",
                    support_count_before=current.support_count,
                    support_count_after=support_after,
                )
            )
        return ProgressHypothesisSet(tuple(next_hypotheses)), tuple(updates)

    def to_record(self) -> dict[str, Any]:
        return {
            "source": "m9_engineered_weights_as_priors",
            "features": [item.to_record() for item in self.hypotheses()],
        }


class LearnedProgressModel:
    def __init__(self, hypotheses: ProgressHypothesisSet) -> None:
        self.hypotheses = hypotheses

    def score_candidate(
        self,
        *,
        candidate: ChainCandidate,
        attempt_memory: AttemptMemory,
    ) -> ChainCandidate:
        score_terms: dict[str, float] = {}
        for feature_id, present in sorted(candidate.predicted_progress_features.items()):
            if feature_id not in FEATURE_PRIORS:
                continue
            weight = self.hypotheses.hypothesis(feature_id).posterior_weight
            score_terms[feature_id] = weight if present else 0.0
        cost = float(candidate.cost_features.get("predicted_action_cost", 0))
        guard_risk = float(candidate.risk_features.get("guard_risk", 0))
        prior_failure_count = attempt_memory.prior_failure_count(candidate.candidate_id)
        score_terms.update({
            "empirical_consistency": 5.0 * float(candidate.cost_features.get("empirical_consistency", 1.0)),
            "action_cost": -1.0 * cost,
            "guard_risk": -30.0 * guard_risk,
            "prior_failed_attempt_count": float(prior_failure_count),
            "prior_failure_penalty": attempt_memory.penalty(candidate.candidate_id),
            "non_applicability": -50.0 if not candidate.applicable else 0.0,
            "noise_as_progress": min(
                0.0,
                score_terms.get("volatile_visual_delta", 0.0),
            ),
        })
        return replace(
            candidate,
            score_terms=score_terms,
            score=sum(score_terms.values()),
            rejection_reason=(
                candidate.rejection_reason
                if candidate.applicable
                else candidate.rejection_reason or "not_applicable_in_current_state"
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "model": "learned_progress_feature_score",
            "hypotheses": self.hypotheses.to_record(),
        }


def outcome_record_from_candidate(
    *,
    outcome_id: str,
    candidate: ChainCandidate,
    observed_features: dict[str, bool],
    stop_reason: str,
    provenance: tuple[dict[str, Any], ...],
) -> ProgressOutcomeRecord:
    utility_terms = {
        key: (weight if observed_features.get(key) else 0.0)
        for key, weight in UTILITY_WEIGHTS.items()
    }
    predicted = tuple(
        feature_id
        for feature_id, present in sorted(candidate.predicted_progress_features.items())
        if present and feature_id in FEATURE_PRIORS
    )
    return ProgressOutcomeRecord(
        outcome_id=outcome_id,
        candidate_id=candidate.candidate_id,
        predicted_features=predicted,
        observed_features=dict(observed_features),
        observed_utility_terms=utility_terms,
        observed_utility=sum(utility_terms.values()),
        downstream_effects={
            "stable_public_transition": bool(
                observed_features.get("new_option_enabled")
                or observed_features.get("new_state_class")
                or observed_features.get("stable_semantic_delta")
                or observed_features.get("level_completed")
            ),
            "prediction_validated": bool(observed_features.get("prediction_validated")),
        },
        stop_reason=stop_reason,
        provenance=provenance,
    )


def m10_status(
    *,
    hypothesis_count: int,
    outcome_count: int,
    feature_update_count: int,
    heldout_score_changed: bool,
    selection_changed_due_to_progress_learning: bool,
    candidate_memory_only_explanation: bool,
    no_feature_generalization: bool,
    outcome_leakage: bool,
    noise_as_progress: bool,
    candidate_memory_confused: bool,
    trace_leakage: bool,
    oracle_signal: bool,
) -> M10Status:
    if outcome_leakage:
        return "failed_outcome_leakage_into_selection"
    if noise_as_progress:
        return "failed_noise_as_progress"
    if candidate_memory_confused:
        return "failed_candidate_memory_confused_with_progress_learning"
    if trace_leakage:
        return "failed_trace_leakage"
    if oracle_signal:
        return "failed_oracle_goal_signal"
    if no_feature_generalization:
        return "failed_no_feature_generalization"
    if outcome_count == 0:
        return "partial_progress_records_only"
    if feature_update_count == 0:
        return "partial_progress_records_only"
    if not heldout_score_changed:
        return "partial_feature_update_without_selection_effect"
    if candidate_memory_only_explanation:
        return "partial_selection_effect_candidate_memory_only"
    if not selection_changed_due_to_progress_learning:
        return "partial_feature_update_without_selection_effect"
    if hypothesis_count < 2:
        return "partial_progress_records_only"
    return "complete_with_known_limits"


def trace_or_oracle_findings(payload: dict[str, Any]) -> tuple[str, ...]:
    encoded = repr(payload)
    forbidden = (
        "human_trace",
        "public_trace_segment",
        "trace_corridor",
        "human_action",
        "raw_action",
        "action_at_index",
        "private",
        "oracle",
    )
    return tuple(marker for marker in forbidden if marker in encoded)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
