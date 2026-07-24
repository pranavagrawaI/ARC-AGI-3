"""Public goal-hypothesis formation for 0001 milestone 12."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal
import json

from .milestone9 import AttemptMemory, ChainCandidate
from .milestone10 import LearnedProgressModel


GoalEvidenceKind = Literal[
    "TERMINAL_PUBLIC_EVIDENCE",
    "PROGRESS_FEATURE_EVIDENCE",
    "TEMPLATE_TRANSFER_EVIDENCE",
    "GRAPH_EXPANSION_EVIDENCE",
    "NOISE_CONTROL_EVIDENCE",
]
GoalHypothesisStatus = Literal["ACTIVE", "TESTABLE", "SUPPRESSED", "UNTESTABLE"]
GoalTestMode = Literal["PURSUE_TARGET", "TEST_HYPOTHESIS", "DISCONFIRMATION_PROBE"]
GoalTestResult = Literal[
    "PUBLIC_SUPPORT_OBSERVED",
    "PUBLIC_CONTRADICTION_OBSERVED",
    "REJECTED_UNTESTABLE",
    "NOISE_ONLY",
]
M12Status = Literal[
    "complete_with_known_limits",
    "partial_goal_evidence_only",
    "partial_hypotheses_without_test",
    "partial_test_without_update",
    "partial_update_without_selection_effect",
    "partial_goal_hypothesis_duplicates_progress_feature",
    "failed_no_competing_hypotheses",
    "failed_fixed_hypothesis_selection",
    "failed_goal_selection_explained_by_progress_only_baseline",
    "failed_no_heldout_goal_generalization",
    "failed_terminal_action_script_promoted",
    "failed_hypothesis_without_public_predicate",
    "failed_oracle_goal_signal",
    "failed_trace_leakage",
    "failed_noise_promoted_to_goal",
    "failed_candidate_memory_confused_with_goal_learning",
    "failed_unpublic_goal_evidence",
]


@dataclass(frozen=True)
class GoalEvidenceRecord:
    evidence_id: str
    kind: GoalEvidenceKind
    source_artifact: str
    public: bool
    supports: tuple[str, ...]
    contradicts: tuple[str, ...]
    payload: dict[str, Any]
    provenance: tuple[dict[str, Any], ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "source_artifact": self.source_artifact,
            "public": self.public,
            "supports": list(self.supports),
            "contradicts": list(self.contradicts),
            "payload": self.payload,
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class GoalHypothesis:
    goal_hypothesis_id: str
    family: str
    claim_type: str
    final_goal_claim: bool
    target_predicate: dict[str, Any]
    support_features: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    confidence: float
    confidence_terms: dict[str, float]
    support_count: int
    contradiction_count: int
    status: GoalHypothesisStatus
    testable: bool
    known_limits: tuple[str, ...]
    forbidden_interpretations: tuple[str, ...]
    score: float = 0.0
    selected: bool = False
    selection_reason: str | None = None

    def has_public_predicate(self) -> bool:
        return bool(self.target_predicate.get("kind")) and bool(
            self.target_predicate.get("expression")
        )

    def promotes_literal_action_script(self) -> bool:
        encoded = json.dumps(self.target_predicate, sort_keys=True)
        forbidden = (
            "literal_action_is_goal",
            "trigger_action_is_goal",
            "action_sequence_is_goal",
            "probe_id_is_goal",
        )
        return any(marker in encoded for marker in forbidden)

    def to_record(self) -> dict[str, Any]:
        return {
            "goal_hypothesis_id": self.goal_hypothesis_id,
            "family": self.family,
            "claim_type": self.claim_type,
            "final_goal_claim": self.final_goal_claim,
            "target_predicate": self.target_predicate,
            "support_features": list(self.support_features),
            "evidence_ids": list(self.evidence_ids),
            "confidence": self.confidence,
            "confidence_terms": self.confidence_terms,
            "support_count": self.support_count,
            "contradiction_count": self.contradiction_count,
            "confidence_interpretation": (
                "small-count deterministic public support, not statistical certainty"
            ),
            "status": self.status,
            "testable": self.testable,
            "known_limits": list(self.known_limits),
            "forbidden_interpretations": list(self.forbidden_interpretations),
            "score": self.score,
            "selected": self.selected,
            "selection_reason": self.selection_reason,
        }


@dataclass(frozen=True)
class GoalTestCandidate:
    candidate_id: str
    goal_hypothesis_id: str
    test_mode: GoalTestMode
    bounded_action_budget: int
    predicted_support: tuple[str, ...]
    predicted_contradiction: tuple[str, ...]
    public_actions: tuple[int, ...]
    generated_before_execution: bool
    executed: bool = False
    result: GoalTestResult | None = None
    observed_public_outcome: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "goal_hypothesis_id": self.goal_hypothesis_id,
            "test_mode": self.test_mode,
            "bounded_action_budget": self.bounded_action_budget,
            "predicted_support": list(self.predicted_support),
            "predicted_contradiction": list(self.predicted_contradiction),
            "public_actions": list(self.public_actions),
            "generated_before_execution": self.generated_before_execution,
            "executed": self.executed,
            "result": self.result,
            "observed_public_outcome": self.observed_public_outcome or {},
        }


@dataclass(frozen=True)
class GoalHypothesisUpdate:
    update_id: str
    goal_hypothesis_id: str
    prior_confidence: float
    posterior_confidence: float
    confidence_delta: float
    evidence_id: str
    update_reason: str
    support_count_before: int
    support_count_after: int
    contradiction_count_before: int
    contradiction_count_after: int

    def to_record(self) -> dict[str, Any]:
        return {
            "update_id": self.update_id,
            "goal_hypothesis_id": self.goal_hypothesis_id,
            "prior_confidence": self.prior_confidence,
            "posterior_confidence": self.posterior_confidence,
            "confidence_delta": self.confidence_delta,
            "evidence_id": self.evidence_id,
            "update_reason": self.update_reason,
            "support_count_before": self.support_count_before,
            "support_count_after": self.support_count_after,
            "contradiction_count_before": self.contradiction_count_before,
            "contradiction_count_after": self.contradiction_count_after,
        }


@dataclass(frozen=True)
class GoalSelectionLedger:
    progress_only_top_candidate: str | None
    goal_aware_top_candidate: str | None
    goal_hypothesis_changed_selection: bool
    heldout_candidate_id: str
    heldout_score_before_update: float
    heldout_score_after_update: float
    heldout_candidate_goal_score_changed: bool
    candidate_memory_only_explanation_ruled_out: bool
    hypotheses: tuple[GoalHypothesis, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "progress_only_top_candidate": self.progress_only_top_candidate,
            "goal_aware_top_candidate": self.goal_aware_top_candidate,
            "goal_hypothesis_changed_selection": self.goal_hypothesis_changed_selection,
            "heldout_candidate_id": self.heldout_candidate_id,
            "heldout_score_before_update": self.heldout_score_before_update,
            "heldout_score_after_update": self.heldout_score_after_update,
            "heldout_candidate_goal_score_changed": (
                self.heldout_candidate_goal_score_changed
            ),
            "candidate_memory_only_explanation_ruled_out": (
                self.candidate_memory_only_explanation_ruled_out
            ),
            "hypotheses": [hypothesis.to_record() for hypothesis in self.hypotheses],
        }


class GoalHypothesisGenerator:
    def generate(
        self,
        *,
        evidence: tuple[GoalEvidenceRecord, ...],
    ) -> tuple[GoalHypothesis, ...]:
        evidence_by_support: dict[str, list[GoalEvidenceRecord]] = {}
        for item in evidence:
            for support in item.supports:
                evidence_by_support.setdefault(support, []).append(item)
        terminal = tuple(evidence_by_support.get("level_transition_triggering", ()))
        decision_surface = tuple(evidence_by_support.get("decision_surface_expansion", ()))
        template = tuple(evidence_by_support.get("transferable_template_progress", ()))
        noise = tuple(evidence_by_support.get("volatile_visual_delta", ()))
        hypotheses = [
            GoalHypothesis(
                goal_hypothesis_id="goal_hyp_level_transition_000",
                family="level_transition_triggering",
                claim_type="instrumental_goal_hypothesis",
                final_goal_claim=False,
                target_predicate={
                    "kind": "public_counter_transition",
                    "expression": {"levels_completed_increases": True},
                },
                support_features=("level_completion",),
                evidence_ids=tuple(item.evidence_id for item in terminal),
                confidence=0.0,
                confidence_terms={
                    "terminal_support": 0.25 if terminal else 0.0,
                    "progress_support": 0.0,
                    "template_transfer_support": 0.0,
                    "testability": 0.08,
                    "specificity": 0.06,
                    "noise_penalty": 0.0,
                },
                support_count=len(terminal),
                contradiction_count=0,
                status="TESTABLE",
                testable=True,
                known_limits=(
                    "public level transition evidence, not final objective proof",
                ),
                forbidden_interpretations=(
                    "literal_action_is_goal",
                    "prior_coordinate_is_goal",
                    "hidden_objective_known",
                ),
            ),
            GoalHypothesis(
                goal_hypothesis_id="goal_hyp_decision_surface_000",
                family="decision_surface_expansion",
                claim_type="instrumental_goal_hypothesis",
                final_goal_claim=False,
                target_predicate={
                    "kind": "state_class_property",
                    "expression": {
                        "enables_previously_unavailable_option": True,
                        "increases_applicable_candidate_count": True,
                    },
                },
                support_features=("new_option_enabled", "new_state_class"),
                evidence_ids=tuple(item.evidence_id for item in decision_surface),
                confidence=0.0,
                confidence_terms={
                    "terminal_support": 0.0,
                    "progress_support": 0.15 if decision_surface else 0.0,
                    "template_transfer_support": 0.0,
                    "testability": 0.07,
                    "specificity": 0.05,
                    "noise_penalty": 0.0,
                },
                support_count=len(decision_surface),
                contradiction_count=0,
                status="TESTABLE",
                testable=True,
                known_limits=("instrumental progress target, not terminal objective",),
                forbidden_interpretations=("new_state_itself_is_final_objective",),
            ),
            GoalHypothesis(
                goal_hypothesis_id="goal_hyp_template_transfer_000",
                family="transferable_template_progress",
                claim_type="instrumental_goal_hypothesis",
                final_goal_claim=False,
                target_predicate={
                    "kind": "template_validation_property",
                    "expression": {
                        "validated_template_instance_promoted": True,
                        "promoted_instance_enters_candidate_surface": True,
                    },
                },
                support_features=("new_option_enabled", "uncertainty_reduction"),
                evidence_ids=tuple(item.evidence_id for item in template),
                confidence=0.0,
                confidence_terms={
                    "terminal_support": 0.0,
                    "progress_support": 0.05,
                    "template_transfer_support": 0.18 if template else 0.0,
                    "testability": 0.08,
                    "specificity": 0.07,
                    "noise_penalty": 0.0,
                },
                support_count=len(template),
                contradiction_count=0,
                status="TESTABLE",
                testable=True,
                known_limits=("spatial template family only",),
                forbidden_interpretations=("template_kind_is_final_objective",),
            ),
            GoalHypothesis(
                goal_hypothesis_id="goal_hyp_noise_000",
                family="volatile_visual_delta",
                claim_type="negative_control",
                final_goal_claim=False,
                target_predicate={
                    "kind": "raw_visual_change",
                    "expression": {
                        "volatile_visual_delta": True,
                        "stable_semantic_delta": False,
                    },
                },
                support_features=("volatile_visual_delta",),
                evidence_ids=tuple(item.evidence_id for item in noise),
                confidence=0.0,
                confidence_terms={
                    "terminal_support": 0.0,
                    "progress_support": 0.0,
                    "template_transfer_support": 0.0,
                    "testability": 0.0,
                    "specificity": 0.02,
                    "noise_penalty": -0.25 if noise else -0.15,
                },
                support_count=0,
                contradiction_count=len(noise),
                status="SUPPRESSED",
                testable=False,
                known_limits=("negative control; not a pursuable hypothesis",),
                forbidden_interpretations=("raw_pixel_novelty_is_goal",),
            ),
        ]
        return tuple(_with_confidence(item) for item in hypotheses)


class GoalHypothesisScorer:
    def score(
        self,
        hypotheses: tuple[GoalHypothesis, ...],
    ) -> tuple[GoalHypothesis, ...]:
        scored = tuple(
            replace(
                hypothesis,
                score=_hypothesis_score(hypothesis),
            )
            for hypothesis in hypotheses
        )
        selected = max(scored, key=lambda item: (item.score, item.goal_hypothesis_id))
        return tuple(
            replace(
                item,
                selected=item.goal_hypothesis_id == selected.goal_hypothesis_id,
                selection_reason=(
                    "highest public goal-hypothesis support score"
                    if item.goal_hypothesis_id == selected.goal_hypothesis_id
                    else None
                ),
            )
            for item in scored
        )


class GoalTestCandidateFactory:
    def create(
        self,
        *,
        hypothesis: GoalHypothesis,
        public_actions: tuple[int, ...] = (4,),
        budget: int = 12,
    ) -> GoalTestCandidate:
        if not hypothesis.testable or hypothesis.status == "SUPPRESSED":
            return GoalTestCandidate(
                candidate_id=f"goal_test_{hypothesis.goal_hypothesis_id}",
                goal_hypothesis_id=hypothesis.goal_hypothesis_id,
                test_mode="DISCONFIRMATION_PROBE",
                bounded_action_budget=budget,
                predicted_support=(),
                predicted_contradiction=("hypothesis_not_testable",),
                public_actions=(),
                generated_before_execution=True,
                executed=False,
                result="REJECTED_UNTESTABLE",
            )
        return GoalTestCandidate(
            candidate_id=f"goal_test_{hypothesis.goal_hypothesis_id}",
            goal_hypothesis_id=hypothesis.goal_hypothesis_id,
            test_mode="TEST_HYPOTHESIS",
            bounded_action_budget=budget,
            predicted_support=(
                "public outcome satisfies target predicate",
                "downstream candidate prioritization improves",
            ),
            predicted_contradiction=(
                "no stable state change",
                "no option enabled",
                "noise only",
                "semantic guard violation",
            ),
            public_actions=public_actions[:budget],
            generated_before_execution=True,
        )


class GoalHypothesisUpdater:
    def update(
        self,
        *,
        hypothesis: GoalHypothesis,
        test: GoalTestCandidate,
        evidence: GoalEvidenceRecord,
    ) -> tuple[GoalHypothesis, GoalHypothesisUpdate]:
        support = test.result == "PUBLIC_SUPPORT_OBSERVED"
        delta = 0.12 if support else -0.12
        posterior = _clamp(hypothesis.confidence + delta, 0.0, 1.0)
        next_terms = dict(hypothesis.confidence_terms)
        if support:
            next_terms["test_outcome_support"] = 0.12
        else:
            next_terms["test_outcome_support"] = -0.12
        updated = replace(
            hypothesis,
            confidence=round(posterior, 4),
            confidence_terms=next_terms,
            support_count=hypothesis.support_count + (1 if support else 0),
            contradiction_count=hypothesis.contradiction_count + (0 if support else 1),
            evidence_ids=hypothesis.evidence_ids + (evidence.evidence_id,),
            status="ACTIVE" if support else hypothesis.status,
        )
        return updated, GoalHypothesisUpdate(
            update_id=f"goal_update_{hypothesis.goal_hypothesis_id}",
            goal_hypothesis_id=hypothesis.goal_hypothesis_id,
            prior_confidence=hypothesis.confidence,
            posterior_confidence=updated.confidence,
            confidence_delta=round(updated.confidence - hypothesis.confidence, 4),
            evidence_id=evidence.evidence_id,
            update_reason="bounded_public_goal_test_outcome",
            support_count_before=hypothesis.support_count,
            support_count_after=updated.support_count,
            contradiction_count_before=hypothesis.contradiction_count,
            contradiction_count_after=updated.contradiction_count,
        )


class GoalAwareSelector:
    def __init__(self, *, progress_model: LearnedProgressModel) -> None:
        self.progress_model = progress_model

    def progress_only_score(
        self,
        *,
        candidate: ChainCandidate,
    ) -> ChainCandidate:
        return self.progress_model.score_candidate(
            candidate=candidate,
            attempt_memory=AttemptMemory(),
        )

    def goal_aware_score(
        self,
        *,
        candidate: ChainCandidate,
        hypotheses: tuple[GoalHypothesis, ...],
    ) -> ChainCandidate:
        base = self.progress_only_score(candidate=candidate)
        bonus = 0.0
        for hypothesis in hypotheses:
            if not hypothesis.testable or hypothesis.status == "SUPPRESSED":
                continue
            if _candidate_matches_hypothesis(candidate, hypothesis):
                bonus += hypothesis.confidence * 100.0
        terms = dict(base.score_terms or {})
        terms["goal_hypothesis_bonus"] = round(bonus, 4)
        return replace(base, score_terms=terms, score=sum(terms.values()))


def m12_status(
    *,
    evidence_count: int,
    hypothesis_count: int,
    all_hypotheses_have_predicates: bool,
    fixed_hypothesis_selection: bool,
    test_generated: bool,
    test_updated_hypothesis: bool,
    heldout_goal_generalization: bool,
    progress_only_explains_selection: bool,
    noise_promoted: bool,
    candidate_memory_confused: bool,
    terminal_action_script_promoted: bool,
    unpublic_evidence: bool,
    trace_leakage: bool,
    oracle_signal: bool,
) -> M12Status:
    if oracle_signal:
        return "failed_oracle_goal_signal"
    if trace_leakage:
        return "failed_trace_leakage"
    if unpublic_evidence:
        return "failed_unpublic_goal_evidence"
    if terminal_action_script_promoted:
        return "failed_terminal_action_script_promoted"
    if noise_promoted:
        return "failed_noise_promoted_to_goal"
    if candidate_memory_confused:
        return "failed_candidate_memory_confused_with_goal_learning"
    if not all_hypotheses_have_predicates:
        return "failed_hypothesis_without_public_predicate"
    if hypothesis_count < 2:
        return "failed_no_competing_hypotheses"
    if fixed_hypothesis_selection:
        return "failed_fixed_hypothesis_selection"
    if progress_only_explains_selection:
        return "failed_goal_selection_explained_by_progress_only_baseline"
    if not heldout_goal_generalization:
        return "failed_no_heldout_goal_generalization"
    if evidence_count < 3:
        return "partial_goal_evidence_only"
    if not test_generated:
        return "partial_hypotheses_without_test"
    if not test_updated_hypothesis:
        return "partial_test_without_update"
    return "complete_with_known_limits"


def forbidden_goal_hypothesis_findings(payload: dict[str, Any]) -> tuple[str, ...]:
    encoded = json.dumps(payload, sort_keys=True)
    forbidden = (
        "human_trace",
        "public_trace_segment",
        "trace_corridor",
        "human_action",
        "raw_action",
        "action_at_index",
        "\"hidden_objective\"",
        "\"hidden_objective_label\"",
        "\"private_state\"",
        "stale_coordinate_used",
    )
    return tuple(marker for marker in forbidden if marker in encoded)


def execute_public_style_goal_test(
    *,
    test: GoalTestCandidate,
    observed_support: bool,
    observed_outcome: dict[str, Any],
) -> GoalTestCandidate:
    return replace(
        test,
        executed=bool(test.public_actions),
        result=(
            "PUBLIC_SUPPORT_OBSERVED"
            if observed_support
            else "PUBLIC_CONTRADICTION_OBSERVED"
        ),
        observed_public_outcome=dict(observed_outcome),
    )


def _with_confidence(hypothesis: GoalHypothesis) -> GoalHypothesis:
    confidence = _clamp(sum(hypothesis.confidence_terms.values()), 0.0, 1.0)
    return replace(hypothesis, confidence=round(confidence, 4))


def _hypothesis_score(hypothesis: GoalHypothesis) -> float:
    if hypothesis.status == "SUPPRESSED" or not hypothesis.testable:
        return -100.0 + hypothesis.confidence
    specificity = float(hypothesis.confidence_terms.get("specificity", 0.0))
    return round((hypothesis.confidence * 100.0) + (specificity * 10.0), 4)


def _candidate_matches_hypothesis(
    candidate: ChainCandidate,
    hypothesis: GoalHypothesis,
) -> bool:
    features = candidate.predicted_progress_features
    if hypothesis.family == "level_transition_triggering":
        return bool(features.get("level_completion")) or any(
            "terminal" in option_id for option_id in candidate.option_ids
        )
    if hypothesis.family == "decision_surface_expansion":
        return bool(features.get("new_option_enabled") or features.get("new_state_class"))
    if hypothesis.family == "transferable_template_progress":
        return any("template" in option_id for option_id in candidate.option_ids)
    return False


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
