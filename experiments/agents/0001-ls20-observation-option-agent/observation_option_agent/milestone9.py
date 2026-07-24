"""Progress-guided option graph selection for 0001 milestone 9."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from .milestone4 import OptionCandidate
from .milestone8 import (
    OptionApplicabilityGraph,
    OptionChainPrediction,
    OptionChainVerification,
)


M9Status = Literal[
    "complete_with_known_limits",
    "partial_single_candidate_only",
    "partial_selection_without_learning_update",
    "partial_learning_update_without_reselection",
    "failed_hardcoded_selection",
    "failed_no_prediction_before_selection",
    "failed_outcome_leakage_into_selection",
    "failed_blind_retry_after_failure",
    "failed_oracle_goal_signal",
    "failed_trace_leakage",
    "failed_noise_as_progress",
]


@dataclass(frozen=True)
class ChainCandidate:
    candidate_id: str
    candidate_type: str
    option_ids: tuple[str, ...]
    start_state_class_id: str
    predicted_terminal_state_class_id: str | None
    predicted_progress_features: dict[str, bool]
    cost_features: dict[str, int | float]
    risk_features: dict[str, int | float | bool]
    provenance: tuple[dict[str, Any], ...]
    applicable: bool = True
    rejection_reason: str | None = None
    score_terms: dict[str, float] | None = None
    score: float = 0.0
    selected: bool = False
    selection_reason: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_type": self.candidate_type,
            "option_ids": list(self.option_ids),
            "start_state_class_id": self.start_state_class_id,
            "predicted_terminal_state_class_id": self.predicted_terminal_state_class_id,
            "predicted_progress_features": self.predicted_progress_features,
            "cost_features": self.cost_features,
            "risk_features": self.risk_features,
            "memory_features": {
                key: value
                for key, value in (self.score_terms or {}).items()
                if key.startswith("prior_")
            },
            "provenance": list(self.provenance),
            "applicable_in_current_state": self.applicable,
            "rejection_reason": self.rejection_reason,
            "score_terms": self.score_terms or {},
            "score": self.score,
            "selected": self.selected,
            "selection_reason": self.selection_reason,
        }


@dataclass(frozen=True)
class OptionSelectionLedger:
    round_index: int
    state_class_id: str
    candidates: tuple[ChainCandidate, ...]
    selected_candidate_id: str | None
    selection_generated_before_execution: bool
    candidate_scores_generated_before_execution: bool
    selection_policy: str = "deterministic_progress_score"

    def to_record(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "state_class_id": self.state_class_id,
            "selection_policy": self.selection_policy,
            "selection_generated_before_execution": self.selection_generated_before_execution,
            "candidate_scores_generated_before_execution": self.candidate_scores_generated_before_execution,
            "selected_candidate_id": self.selected_candidate_id,
            "candidates": [candidate.to_record() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class OptionAttemptOutcome:
    attempt_id: str
    candidate_id: str
    predicted_score: float
    verification_result: str
    observed_progress_features: dict[str, bool]
    final_state_class_id: str | None
    stopped_reason: str | None
    model_updates: tuple[dict[str, Any], ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "candidate_id": self.candidate_id,
            "predicted_score": self.predicted_score,
            "verification_result": self.verification_result,
            "observed_progress_features": self.observed_progress_features,
            "final_state_class_id": self.final_state_class_id,
            "stopped_reason": self.stopped_reason,
            "model_updates": list(self.model_updates),
        }


class AttemptMemory:
    def __init__(self) -> None:
        self._failed_attempts: dict[str, list[dict[str, Any]]] = {}

    def prior_failure_count(self, candidate_id: str) -> int:
        return len(self._failed_attempts.get(candidate_id, ()))

    def penalty(self, candidate_id: str) -> float:
        return -40.0 * self.prior_failure_count(candidate_id)

    def record_failure(
        self,
        *,
        candidate_id: str,
        reason: str,
    ) -> dict[str, Any]:
        update = {
            "update_type": "FAILED_CANDIDATE_PENALTY",
            "candidate_id": candidate_id,
            "reason": reason,
            "penalty_added": -40,
            "future_retry_requires_new_evidence": True,
        }
        self._failed_attempts.setdefault(candidate_id, []).append(update)
        return update

    def to_record(self) -> dict[str, Any]:
        return {
            "failed_attempts": {
                candidate_id: list(records)
                for candidate_id, records in sorted(self._failed_attempts.items())
            }
        }


class ProgressModel:
    def score_candidate(
        self,
        *,
        candidate: ChainCandidate,
        attempt_memory: AttemptMemory,
    ) -> ChainCandidate:
        features = candidate.predicted_progress_features
        cost = float(candidate.cost_features.get("predicted_action_cost", 0))
        guard_risk = float(candidate.risk_features.get("guard_risk", 0))
        prior_failure_count = attempt_memory.prior_failure_count(candidate.candidate_id)
        score_terms = {
            "predicted_level_completion": 1000.0 if features.get("level_completion") else 0.0,
            "stable_semantic_delta": 50.0 if features.get("stable_semantic_delta") else 0.0,
            "new_option_enabled": 25.0 if features.get("new_option_enabled") else 0.0,
            "new_state_class": 15.0 if features.get("new_state_class") else 0.0,
            "uncertainty_reduction": 10.0 if features.get("uncertainty_reduction") else 0.0,
            "empirical_consistency": 5.0 * float(candidate.cost_features.get("empirical_consistency", 1.0)),
            "action_cost": -1.0 * cost,
            "guard_risk": -30.0 * guard_risk,
            "prior_failed_attempt_count": float(prior_failure_count),
            "prior_failure_penalty": attempt_memory.penalty(candidate.candidate_id),
            "non_applicability": -50.0 if not candidate.applicable else 0.0,
            "noise_as_progress": 0.0,
        }
        score = sum(score_terms.values())
        rejection_reason = candidate.rejection_reason
        if not candidate.applicable and rejection_reason is None:
            rejection_reason = "not_applicable_in_current_state"
        return replace(
            candidate,
            score_terms=score_terms,
            score=score,
            rejection_reason=rejection_reason,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "model": "deterministic_progress_score",
            "weights": {
                "predicted_level_completion": 1000,
                "stable_semantic_delta": 50,
                "new_option_enabled": 25,
                "new_state_class": 15,
                "uncertainty_reduction": 10,
                "empirical_consistency": 5,
                "predicted_action_cost": -1,
                "guard_risk": -30,
                "prior_failed_attempt": -40,
                "non_applicable": -50,
            },
            "volatile_visual_delta_ignored": True,
            "noise_as_progress_score": 0,
        }


class SelectionPolicy:
    def __init__(self, *, progress_model: ProgressModel | None = None) -> None:
        self.progress_model = progress_model or ProgressModel()

    def select(
        self,
        *,
        round_index: int,
        state_class_id: str,
        candidates: tuple[ChainCandidate, ...],
        attempt_memory: AttemptMemory,
    ) -> OptionSelectionLedger:
        scored = tuple(
            self.progress_model.score_candidate(
                candidate=candidate,
                attempt_memory=attempt_memory,
            )
            for candidate in candidates
        )
        viable = tuple(candidate for candidate in scored if candidate.applicable)
        pool = viable if viable else scored
        selected = max(
            pool,
            key=lambda candidate: (candidate.score, candidate.candidate_id),
        ) if pool else None
        decorated = tuple(
            replace(
                candidate,
                selected=selected is not None and candidate.candidate_id == selected.candidate_id,
                selection_reason=(
                    "highest score among applicable candidates"
                    if selected is not None and candidate.candidate_id == selected.candidate_id
                    else None
                ),
            )
            for candidate in scored
        )
        return OptionSelectionLedger(
            round_index=round_index,
            state_class_id=state_class_id,
            candidates=decorated,
            selected_candidate_id=selected.candidate_id if selected else None,
            selection_generated_before_execution=True,
            candidate_scores_generated_before_execution=True,
        )


def generate_chain_candidates(
    *,
    graph: OptionApplicabilityGraph,
    start_state_class_id: str,
    option_a: OptionCandidate,
    option_b: OptionCandidate,
) -> tuple[ChainCandidate, ...]:
    state = graph.state(start_state_class_id)
    if state is None:
        return ()
    option_a_prediction = graph.predict_chain(
        prediction_id="pred_candidate_option_000",
        start_state_class_id=start_state_class_id,
        options=(option_a,),
    )
    option_b_prediction = graph.predict_chain(
        prediction_id="pred_candidate_option_001",
        start_state_class_id=start_state_class_id,
        options=(option_b,),
    )
    chain_prediction = graph.predict_chain(
        prediction_id="pred_candidate_chain_000",
        start_state_class_id=start_state_class_id,
        options=(option_a, option_b),
    )
    return (
        _candidate_from_prediction(
            candidate_id="option_000",
            candidate_type="single_option",
            option_ids=(option_a.option_id,),
            start_state_class_id=start_state_class_id,
            prediction=option_a_prediction,
            applicable=option_a_prediction is not None,
            action_cost=4,
            new_option_enabled=True,
            uncertainty_reduction=False,
        ),
        _candidate_from_prediction(
            candidate_id="option_001",
            candidate_type="single_option",
            option_ids=(option_b.option_id,),
            start_state_class_id=start_state_class_id,
            prediction=option_b_prediction,
            applicable=option_b_prediction is not None,
            action_cost=2,
            new_option_enabled=False,
            uncertainty_reduction=False,
        ),
        _candidate_from_prediction(
            candidate_id="chain_000",
            candidate_type="option_chain",
            option_ids=(option_a.option_id, option_b.option_id),
            start_state_class_id=start_state_class_id,
            prediction=chain_prediction,
            applicable=chain_prediction is not None,
            action_cost=6,
            new_option_enabled=True,
            uncertainty_reduction=True,
        ),
    )


def outcome_from_chain_verification(
    *,
    attempt_id: str,
    candidate: ChainCandidate,
    verification: OptionChainVerification,
) -> OptionAttemptOutcome:
    succeeded = verification.result == "CHAIN_VERIFIED"
    observed = {
        "level_completion": False,
        "stable_semantic_delta": False,
        "new_option_enabled": any(
            link.next_link_released
            for link in verification.link_verifications
        ),
        "new_state_class": succeeded,
        "prediction_validated": succeeded,
    }
    updates = []
    if not succeeded:
        updates.append({
            "update_type": "FAILED_CANDIDATE_OBSERVED",
            "candidate_id": candidate.candidate_id,
            "reason": verification.result,
        })
    return OptionAttemptOutcome(
        attempt_id=attempt_id,
        candidate_id=candidate.candidate_id,
        predicted_score=candidate.score,
        verification_result=verification.result,
        observed_progress_features=observed,
        final_state_class_id=(
            verification.link_verifications[-1].observed_next_state_class_id
            if verification.link_verifications
            else None
        ),
        stopped_reason=verification.stop_reason,
        model_updates=tuple(updates),
    )


def m9_status(
    *,
    candidate_count: int,
    selected_by_score: bool,
    learning_update_count: int,
    second_selection_reflected_update: bool,
    negative_candidate_deprioritized: bool,
    hardcoded_selection: bool,
    prediction_before_selection: bool,
    outcome_leakage: bool,
    blind_retry: bool,
    oracle_signal: bool,
    trace_leakage: bool,
    noise_as_progress: bool,
) -> M9Status:
    if hardcoded_selection:
        return "failed_hardcoded_selection"
    if not prediction_before_selection:
        return "failed_no_prediction_before_selection"
    if outcome_leakage:
        return "failed_outcome_leakage_into_selection"
    if blind_retry:
        return "failed_blind_retry_after_failure"
    if oracle_signal:
        return "failed_oracle_goal_signal"
    if trace_leakage:
        return "failed_trace_leakage"
    if noise_as_progress:
        return "failed_noise_as_progress"
    if candidate_count < 2:
        return "partial_single_candidate_only"
    if not selected_by_score:
        return "failed_hardcoded_selection"
    if learning_update_count == 0:
        return "partial_selection_without_learning_update"
    if not second_selection_reflected_update:
        return "partial_learning_update_without_reselection"
    if not negative_candidate_deprioritized:
        return "failed_blind_retry_after_failure"
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
        "private",
        "oracle",
    )
    return tuple(marker for marker in forbidden if marker in encoded)


def _candidate_from_prediction(
    *,
    candidate_id: str,
    candidate_type: str,
    option_ids: tuple[str, ...],
    start_state_class_id: str,
    prediction: OptionChainPrediction | None,
    applicable: bool,
    action_cost: int,
    new_option_enabled: bool,
    uncertainty_reduction: bool,
) -> ChainCandidate:
    terminal = (
        prediction.links[-1].predicted_next_state_class_id
        if prediction and prediction.links
        else None
    )
    support = min(
        (
            link.option_prediction.support_count
            for link in prediction.links
        ),
        default=0,
    ) if prediction else 0
    return ChainCandidate(
        candidate_id=candidate_id,
        candidate_type=candidate_type,
        option_ids=option_ids,
        start_state_class_id=start_state_class_id,
        predicted_terminal_state_class_id=terminal,
        predicted_progress_features={
            "level_completion": False,
            "stable_semantic_delta": False,
            "new_option_enabled": new_option_enabled and applicable,
            "new_state_class": bool(terminal and terminal != start_state_class_id),
            "uncertainty_reduction": uncertainty_reduction and support <= 1 and applicable,
            "volatile_visual_delta": False,
        },
        cost_features={
            "predicted_action_cost": action_cost,
            "empirical_consistency": min(
                (
                    link.option_prediction.empirical_consistency
                    for link in prediction.links
                ),
                default=0.0,
            ) if prediction else 0.0,
        },
        risk_features={"guard_risk": 0},
        provenance=(
            {
                "kind": "option_graph_prediction",
                "prediction_id": prediction.prediction_id if prediction else None,
            },
        ),
        applicable=applicable,
        rejection_reason=None if applicable else "not_applicable_in_current_state",
    )
