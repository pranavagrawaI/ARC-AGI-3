"""Goal-hypothesis to live-candidate binding for 0001 milestone 12.6."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Any, Literal

from .milestone9 import (
    AttemptMemory,
    ChainCandidate,
    OptionSelectionLedger,
    ProgressModel,
)
from .milestone10 import LearnedProgressModel, ProgressHypothesisSet
from .milestone12 import GoalHypothesis


GoalBindingMode = Literal[
    "BIND_EXISTING_LIVE_CANDIDATE",
    "SYNTHESIZE_GOAL_CONDITIONED_PROBE",
    "NO_BINDING_AVAILABLE",
    "REJECTED_NO_GOAL_RELEVANCE",
]
GoalLiveOutcome = Literal[
    "STRONG_SUPPORT",
    "WEAK_SUPPORT",
    "UNKNOWN",
    "WEAK_CONTRADICTION",
    "CONTRADICTION",
]
M126Status = Literal[
    "complete_with_known_limits",
    "partial_binding_only",
    "partial_goal_probe_synthesized_not_actuated",
    "partial_no_bindable_live_candidate",
    "failed_fixture_candidate_injected",
    "failed_no_goal_live_binding",
    "failed_bound_candidate_not_on_live_surface",
    "failed_selected_candidate_not_actuated",
    "failed_goal_outcome_not_updated",
    "failed_overbroad_goal_binding",
    "failed_irrelevant_candidate_bound_to_goal",
    "failed_trace_or_oracle_leakage",
]


@dataclass(frozen=True)
class GoalLiveBinding:
    binding_id: str
    goal_hypothesis_id: str
    candidate_id: str | None
    binding_mode: GoalBindingMode
    binding_allowed: bool
    binding_relevance_terms: dict[str, float]
    binding_reason: str
    candidate_on_live_surface: bool
    fixture_candidate_used_as_live_candidate: bool
    provenance: tuple[dict[str, Any], ...]

    @property
    def goal_relevance_score(self) -> float:
        if not self.binding_allowed:
            return 0.0
        weights = {
            "terminal_directness": 100.0,
            "decision_surface_expansion_support": 6.0,
            "template_transfer_support": 4.0,
            "fresh_level_context_relevance": 1.0,
            "validated_new_level_option_provenance": 1.0,
        }
        return round(
            sum(
                weights.get(key, 0.0) * float(value)
                for key, value in self.binding_relevance_terms.items()
            ),
            4,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "binding_id": self.binding_id,
            "goal_hypothesis_id": self.goal_hypothesis_id,
            "candidate_id": self.candidate_id,
            "binding_mode": self.binding_mode,
            "binding_allowed": self.binding_allowed,
            "binding_relevance_terms": self.binding_relevance_terms,
            "goal_relevance_score": self.goal_relevance_score,
            "binding_reason": self.binding_reason,
            "candidate_on_live_surface": self.candidate_on_live_surface,
            "fixture_candidate_used_as_live_candidate": (
                self.fixture_candidate_used_as_live_candidate
            ),
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class GoalLiveUpdate:
    update_id: str
    goal_hypothesis_id: str
    candidate_id: str
    outcome: GoalLiveOutcome
    confidence_delta: float
    outcome_terms: dict[str, Any]
    public_actions_executed: bool
    update_reason: str

    def to_record(self) -> dict[str, Any]:
        return {
            "update_id": self.update_id,
            "goal_hypothesis_id": self.goal_hypothesis_id,
            "candidate_id": self.candidate_id,
            "outcome": self.outcome,
            "confidence_delta": self.confidence_delta,
            "outcome_terms": self.outcome_terms,
            "public_actions_executed": self.public_actions_executed,
            "update_reason": self.update_reason,
        }


class GoalHypothesisBinder:
    def bind_existing(
        self,
        *,
        hypothesis: GoalHypothesis,
        live_candidates: tuple[ChainCandidate, ...],
        fixture_candidate_id: str,
    ) -> tuple[GoalLiveBinding, ...]:
        live_ids = {candidate.candidate_id for candidate in live_candidates}
        bindings: list[GoalLiveBinding] = []
        for candidate in live_candidates:
            terms = _relevance_terms(hypothesis=hypothesis, candidate=candidate)
            allowed = _has_actionable_relevance(terms)
            mode: GoalBindingMode = (
                "BIND_EXISTING_LIVE_CANDIDATE"
                if allowed
                else "REJECTED_NO_GOAL_RELEVANCE"
            )
            bindings.append(GoalLiveBinding(
                binding_id=f"goal_binding_{hypothesis.goal_hypothesis_id}_{candidate.candidate_id}",
                goal_hypothesis_id=hypothesis.goal_hypothesis_id,
                candidate_id=candidate.candidate_id,
                binding_mode=mode,
                binding_allowed=allowed,
                binding_relevance_terms=terms,
                binding_reason=_binding_reason(
                    allowed=allowed,
                    hypothesis=hypothesis,
                    candidate=candidate,
                ),
                candidate_on_live_surface=candidate.candidate_id in live_ids,
                fixture_candidate_used_as_live_candidate=(
                    candidate.candidate_id == fixture_candidate_id
                ),
                provenance=({
                    "kind": "goal_live_binding",
                    "candidate_provenance": [dict(item) for item in candidate.provenance],
                },),
            ))
        if not bindings:
            return (GoalLiveBinding(
                binding_id=f"goal_binding_{hypothesis.goal_hypothesis_id}_none",
                goal_hypothesis_id=hypothesis.goal_hypothesis_id,
                candidate_id=None,
                binding_mode="NO_BINDING_AVAILABLE",
                binding_allowed=False,
                binding_relevance_terms={},
                binding_reason="no live candidates available for goal binding",
                candidate_on_live_surface=False,
                fixture_candidate_used_as_live_candidate=False,
                provenance=({"kind": "goal_live_binding_abstention"},),
            ),)
        return tuple(bindings)

    def best_allowed(self, bindings: tuple[GoalLiveBinding, ...]) -> GoalLiveBinding | None:
        allowed = tuple(binding for binding in bindings if binding.binding_allowed)
        if not allowed:
            return None
        return max(
            allowed,
            key=lambda binding: (
                binding.goal_relevance_score,
                binding.candidate_id or "",
            ),
        )


class GoalLiveSelector:
    def __init__(self, *, progress_model: ProgressModel | None = None) -> None:
        self.progress_model = progress_model or LearnedProgressModel(
            ProgressHypothesisSet.from_m9_priors()
        )

    def select(
        self,
        *,
        round_index: int,
        state_class_id: str,
        candidates: tuple[ChainCandidate, ...],
        bindings: tuple[GoalLiveBinding, ...],
        attempt_memory: AttemptMemory,
    ) -> OptionSelectionLedger:
        bindings_by_candidate = {
            binding.candidate_id: binding
            for binding in bindings
            if binding.binding_allowed and binding.candidate_id is not None
        }
        scored = []
        for candidate in candidates:
            base = self.progress_model.score_candidate(
                candidate=candidate,
                attempt_memory=attempt_memory,
            )
            binding = bindings_by_candidate.get(candidate.candidate_id)
            terms = dict(base.score_terms or {})
            if binding is not None:
                terms["goal_relevance_score"] = binding.goal_relevance_score
            else:
                terms["goal_relevance_score"] = 0.0
            scored.append(replace(base, score_terms=terms, score=sum(terms.values())))
        viable = tuple(candidate for candidate in scored if candidate.applicable)
        pool = viable if viable else tuple(scored)
        selected = max(
            pool,
            key=lambda candidate: (candidate.score, candidate.candidate_id),
        ) if pool else None
        decorated = tuple(
            replace(
                candidate,
                selected=selected is not None and candidate.candidate_id == selected.candidate_id,
                selection_reason=(
                    "highest live progress plus goal relevance score"
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
            selection_policy="live_progress_plus_goal_relevance",
        )


class GoalLiveOutcomeUpdater:
    def classify(
        self,
        *,
        hypothesis: GoalHypothesis,
        binding: GoalLiveBinding,
        candidate: ChainCandidate,
        execution: dict[str, Any],
    ) -> GoalLiveUpdate:
        public_actions_executed = bool(execution.get("public_actions_executed"))
        levels_before = int(execution.get("levels_before", 0))
        levels_after = int(execution.get("levels_after", levels_before))
        guard_violation = bool(execution.get("guard_violation"))
        action_conflict = bool(execution.get("action_conflict"))
        new_state_class = bool(candidate.predicted_progress_features.get("new_state_class"))
        new_option_enabled = bool(
            candidate.predicted_progress_features.get("new_option_enabled")
        )
        if levels_after > levels_before:
            outcome: GoalLiveOutcome = "STRONG_SUPPORT"
            delta = 0.2
        elif guard_violation or action_conflict:
            outcome = "CONTRADICTION"
            delta = -0.15
        elif public_actions_executed and (new_state_class or new_option_enabled):
            outcome = "WEAK_SUPPORT"
            delta = 0.05
        elif public_actions_executed:
            outcome = "UNKNOWN"
            delta = 0.0
        else:
            outcome = "WEAK_CONTRADICTION"
            delta = -0.05
        return GoalLiveUpdate(
            update_id=f"goal_live_update_{hypothesis.goal_hypothesis_id}_{candidate.candidate_id}",
            goal_hypothesis_id=hypothesis.goal_hypothesis_id,
            candidate_id=candidate.candidate_id,
            outcome=outcome,
            confidence_delta=delta,
            outcome_terms={
                "levels_before": levels_before,
                "levels_after": levels_after,
                "new_state_class": new_state_class,
                "new_option_enabled": new_option_enabled,
                "guard_violation": guard_violation,
                "action_conflict": action_conflict,
                "goal_relevance_score": binding.goal_relevance_score,
            },
            public_actions_executed=public_actions_executed,
            update_reason="live_goal_bound_candidate_public_outcome",
        )


def m126_status(
    *,
    fixture_candidate_injected: bool,
    binding_count: int,
    honest_relevant_candidate_exists: bool,
    selected_binding_on_live_surface: bool,
    selected_candidate_actuated: bool,
    outcome_updated: bool,
    overbroad_binding: bool,
    irrelevant_candidate_bound: bool,
    trace_or_oracle_leakage: bool,
    probe_synthesized_not_actuated: bool = False,
) -> M126Status:
    if trace_or_oracle_leakage:
        return "failed_trace_or_oracle_leakage"
    if fixture_candidate_injected:
        return "failed_fixture_candidate_injected"
    if overbroad_binding:
        return "failed_overbroad_goal_binding"
    if irrelevant_candidate_bound:
        return "failed_irrelevant_candidate_bound_to_goal"
    if honest_relevant_candidate_exists and binding_count == 0:
        return "failed_no_goal_live_binding"
    if binding_count == 0:
        return "partial_no_bindable_live_candidate"
    if not selected_binding_on_live_surface:
        return "failed_bound_candidate_not_on_live_surface"
    if probe_synthesized_not_actuated:
        return "partial_goal_probe_synthesized_not_actuated"
    if not selected_candidate_actuated:
        return "failed_selected_candidate_not_actuated"
    if not outcome_updated:
        return "failed_goal_outcome_not_updated"
    return "complete_with_known_limits"


def forbidden_goal_live_binding_findings(payload: dict[str, Any]) -> tuple[str, ...]:
    encoded = json.dumps(payload, sort_keys=True)
    forbidden = (
        "human_trace",
        "public_trace_segment",
        "trace_corridor",
        "human_action",
        "raw_action",
        "\"hidden_objective\"",
        "\"hidden_objective_label\"",
        "\"private_state\"",
        "stale_coordinate_used",
    )
    return tuple(marker for marker in forbidden if marker in encoded)


def _relevance_terms(
    *,
    hypothesis: GoalHypothesis,
    candidate: ChainCandidate,
) -> dict[str, float]:
    terms = {
        "decision_surface_expansion_support": 0.0,
        "template_transfer_support": 0.0,
        "fresh_level_context_relevance": 0.0,
        "validated_new_level_option_provenance": 0.0,
        "terminal_directness": 0.0,
    }
    features = candidate.predicted_progress_features
    provenance = json.dumps(candidate.provenance, sort_keys=True)
    option_ids = " ".join(candidate.option_ids)
    if hypothesis.family == "level_transition_triggering":
        if features.get("level_completion"):
            terms["terminal_directness"] = 1.0
        if features.get("new_state_class") or features.get("new_option_enabled"):
            terms["decision_surface_expansion_support"] = 1.0
        if "template" in option_ids or "template" in provenance:
            terms["template_transfer_support"] = 1.0
            terms["validated_new_level_option_provenance"] = 1.0
        if candidate.start_state_class_id.startswith("L"):
            terms["fresh_level_context_relevance"] = 1.0
    elif hypothesis.family == "decision_surface_expansion":
        if features.get("new_state_class") or features.get("new_option_enabled"):
            terms["decision_surface_expansion_support"] = 1.0
        if candidate.start_state_class_id.startswith("L"):
            terms["fresh_level_context_relevance"] = 1.0
    elif hypothesis.family == "transferable_template_progress":
        if "template" in option_ids or "template" in provenance:
            terms["template_transfer_support"] = 1.0
            terms["validated_new_level_option_provenance"] = 1.0
        if candidate.start_state_class_id.startswith("L"):
            terms["fresh_level_context_relevance"] = 1.0
    return terms


def _has_actionable_relevance(terms: dict[str, float]) -> bool:
    actionable = (
        "terminal_directness",
        "decision_surface_expansion_support",
        "template_transfer_support",
        "validated_new_level_option_provenance",
    )
    return any(float(terms.get(key, 0.0)) > 0.0 for key in actionable)


def _binding_reason(
    *,
    allowed: bool,
    hypothesis: GoalHypothesis,
    candidate: ChainCandidate,
) -> str:
    if not allowed:
        return "candidate has no explicit relevance relation to active goal hypothesis"
    if hypothesis.family == "level_transition_triggering":
        return (
            "validated transferred template expands live decision surface under "
            "active level-transition hypothesis"
        )
    return f"candidate has explicit relevance to {hypothesis.family}"
