"""Cross-level option-template transfer for 0001 milestone 11.7."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Literal

from .milestone4 import (
    OptionCandidate,
    Position,
    PredicateCandidate,
    predicate_expression_hash,
)
from .milestone7 import OptionEffectPrediction
from .milestone8 import (
    OptionApplicabilityEdge,
    OptionApplicabilityGraph,
    OptionApplicabilityStateClass,
)
from .milestone11 import (
    FrontierProbeCandidate,
    FrontierProbeExecution,
    GraphExpansion,
)
from .milestone116 import LevelContext


TemplateValidationResult = Literal[
    "VALIDATED_TEMPLATE_INSTANCE",
    "REJECTED_STALE_INSTANCE",
    "TARGET_UNREACHABLE",
    "ACTION_CONFLICT",
    "SEMANTIC_GUARD_VIOLATION",
    "UNKNOWN",
]
M117Status = Literal[
    "complete_with_known_limits",
    "partial_template_extracted_only",
    "partial_template_instantiated_no_validation",
    "partial_template_validated_not_used",
    "partial_template_stored_but_no_behavioral_effect",
    "partial_template_scored_not_executed",
    "partial_no_valid_template_instance",
    "failed_stale_spatial_instance_reused",
    "failed_prior_level_coordinate_leakage",
    "failed_prior_level_edge_leakage",
    "failed_source_probe_script_reused",
    "failed_template_contains_level_local_fact",
    "failed_instance_binding_from_prior_level_fact",
    "failed_no_stale_negative_control",
    "failed_template_no_contribution",
    "failed_template_validation_not_public",
    "failed_trace_leakage",
    "failed_oracle_signal",
]


@dataclass(frozen=True)
class OptionTemplate:
    template_id: str
    kind: str
    predicate_family: str
    radius: int
    metric: str
    guard_requirements: tuple[str, ...]
    validation_requirements: tuple[str, ...]
    planner_family: str
    expected_effect_class: str
    source_option_id: str
    source_level_index: int
    source_option_expression_hash: str
    source_probe_action_sequence_present_as_evidence: bool
    source_probe_sequence_used_as_policy: bool = False

    @property
    def expression(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "predicate_family": self.predicate_family,
            "radius": self.radius,
            "metric": self.metric,
            "guard_requirements": list(self.guard_requirements),
            "validation_requirements": list(self.validation_requirements),
            "planner_family": self.planner_family,
            "expected_effect_class": self.expected_effect_class,
        }

    @property
    def template_expression_hash(self) -> str:
        return _stable_hash(self.expression)

    def contains_level_local_fact(self) -> bool:
        encoded = json.dumps(self.expression, sort_keys=True)
        forbidden = (
            "center",
            "target_position",
            "state_class_id",
            "edge_id",
            "observation_hash",
            "action_sequence",
        )
        return any(marker in encoded for marker in forbidden)

    def to_record(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "kind": self.kind,
            "predicate_family": self.predicate_family,
            "radius": self.radius,
            "metric": self.metric,
            "guard_requirements": list(self.guard_requirements),
            "validation_requirements": list(self.validation_requirements),
            "planner_family": self.planner_family,
            "expected_effect_class": self.expected_effect_class,
            "source_option_id": self.source_option_id,
            "source_level_index": self.source_level_index,
            "source_option_expression_hash": self.source_option_expression_hash,
            "template_expression_hash": self.template_expression_hash,
            "source_probe_action_sequence_present_as_evidence": (
                self.source_probe_action_sequence_present_as_evidence
            ),
            "source_probe_sequence_used_as_policy": self.source_probe_sequence_used_as_policy,
            "contains_level_local_fact": self.contains_level_local_fact(),
            "expression": self.expression,
        }


@dataclass(frozen=True)
class OptionInstance:
    instance_id: str
    template_id: str
    level_context_id: str
    level_index: int
    source_state_class_id: str
    target_expression: dict[str, Any]
    validation_action_sequence: tuple[int, ...]
    template_provenance: tuple[dict[str, Any], ...]
    instance_binding_provenance: tuple[dict[str, Any], ...]
    prior_level_coordinates_used: bool
    prior_level_edges_used: bool
    prior_level_state_ids_used: bool
    prior_level_observation_hashes_used: bool
    source_probe_sequence_used_as_policy: bool
    new_level_plan_generated_from_current_state: bool
    selected: bool = False
    selection_reason: str | None = None

    @property
    def instance_expression_hash(self) -> str:
        return _stable_hash(self.target_expression)

    @property
    def forbidden_prior_fields_present(self) -> bool:
        return any((
            self.prior_level_coordinates_used,
            self.prior_level_edges_used,
            self.prior_level_state_ids_used,
            self.prior_level_observation_hashes_used,
            self.source_probe_sequence_used_as_policy,
        ))

    def to_record(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "template_id": self.template_id,
            "level_context_id": self.level_context_id,
            "level_index": self.level_index,
            "source_state_class_id": self.source_state_class_id,
            "target_expression": self.target_expression,
            "instance_expression_hash": self.instance_expression_hash,
            "validation_action_sequence": list(self.validation_action_sequence),
            "template_provenance": list(self.template_provenance),
            "instance_binding_provenance": list(self.instance_binding_provenance),
            "prior_level_coordinates_used": self.prior_level_coordinates_used,
            "prior_level_edges_used": self.prior_level_edges_used,
            "prior_level_state_ids_used": self.prior_level_state_ids_used,
            "prior_level_observation_hashes_used": self.prior_level_observation_hashes_used,
            "source_probe_sequence_used_as_policy": self.source_probe_sequence_used_as_policy,
            "new_level_plan_generated_from_current_state": (
                self.new_level_plan_generated_from_current_state
            ),
            "forbidden_prior_fields_present": self.forbidden_prior_fields_present,
            "selected": self.selected,
            "selection_reason": self.selection_reason,
        }


@dataclass(frozen=True)
class TemplateValidation:
    validation_id: str
    instance_id: str
    result: TemplateValidationResult
    public_actions_used: bool
    validation_actions: tuple[int, ...]
    semantic_guards_preserved: bool
    start_position: Position | None
    final_position: Position | None
    promoted_option_id: str | None
    reason: str

    def to_record(self) -> dict[str, Any]:
        return {
            "validation_id": self.validation_id,
            "instance_id": self.instance_id,
            "result": self.result,
            "public_actions_used": self.public_actions_used,
            "validation_actions": list(self.validation_actions),
            "semantic_guards_preserved": self.semantic_guards_preserved,
            "start_position": list(self.start_position) if self.start_position else None,
            "final_position": list(self.final_position) if self.final_position else None,
            "promoted_option_id": self.promoted_option_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TemplateContribution:
    changed_candidate_generation: bool
    changed_candidate_score: bool
    changed_selection_surface: bool
    generic_frontier_baseline_candidate_count: int
    template_augmented_candidate_count: int

    @property
    def detected(self) -> bool:
        return any((
            self.changed_candidate_generation,
            self.changed_candidate_score,
            self.changed_selection_surface,
        ))

    def to_record(self) -> dict[str, Any]:
        return {
            "changed_candidate_generation": self.changed_candidate_generation,
            "changed_candidate_score": self.changed_candidate_score,
            "changed_selection_surface": self.changed_selection_surface,
            "generic_frontier_baseline_candidate_count": (
                self.generic_frontier_baseline_candidate_count
            ),
            "template_augmented_candidate_count": self.template_augmented_candidate_count,
            "template_contribution_detected": self.detected,
        }


class OptionTemplateExtractor:
    def extract(
        self,
        *,
        options: tuple[OptionCandidate, ...],
        source_level_index: int,
    ) -> tuple[OptionTemplate, ...]:
        templates: list[OptionTemplate] = []
        for index, option in enumerate(options):
            if option.source != "frontier_probe":
                continue
            expression = _selected_predicate_expression(option)
            if expression.get("type") != "region_membership":
                continue
            templates.append(OptionTemplate(
                template_id=f"template_frontier_region_navigation_{index:03d}",
                kind="frontier_region_navigation",
                predicate_family="region_membership",
                radius=int(expression.get("radius", 2)),
                metric=str(expression.get("metric", "manhattan")),
                guard_requirements=("semantic_guards_valid", "controlled_entity_tracked"),
                validation_requirements=("public_probe_success", "stable_actor_delta"),
                planner_family="bounded_action_delta_planner",
                expected_effect_class="stable_actor_region_transition",
                source_option_id=option.option_id,
                source_level_index=source_level_index,
                source_option_expression_hash=(
                    option.selected_predicate_expression_hash
                    or predicate_expression_hash(expression)
                ),
                source_probe_action_sequence_present_as_evidence=False,
            ))
        return tuple(templates)


class CrossLevelTemplateRegistry:
    def __init__(self, *, templates: tuple[OptionTemplate, ...] = ()) -> None:
        self.templates = templates

    def carry_into(self, *, level_context: LevelContext) -> dict[str, Any]:
        return {
            "level_context_id": level_context.context_id,
            "level_index": level_context.level_index,
            "template_count": len(self.templates),
            "templates": [template.to_record() for template in self.templates],
            "level_local_instances_invalidated": True,
        }

    def to_record(self) -> dict[str, Any]:
        return {"templates": [template.to_record() for template in self.templates]}


class LevelTemplateInstantiator:
    def instantiate(
        self,
        *,
        template: OptionTemplate,
        level_context: LevelContext,
        observation_hash: str,
        controlled_position: Position,
        controlled_position_source: str,
        available_actions: tuple[int, ...],
    ) -> OptionInstance:
        action = _fresh_validation_action(available_actions)
        target_expression = {
            "type": "region_membership",
            "center_source": "fresh_level_context_probe_result",
            "anchor_position": list(controlled_position),
            "radius": template.radius,
            "metric": template.metric,
            "excludes_initiation": True,
        }
        return OptionInstance(
            instance_id=f"level_{level_context.level_index}:instance_{template.template_id}",
            template_id=template.template_id,
            level_context_id=level_context.context_id,
            level_index=level_context.level_index,
            source_state_class_id=level_context.initial_state_class_id,
            target_expression=target_expression,
            validation_action_sequence=(action,),
            template_provenance=({
                "source": "prior_level_frontier_option_template",
                "source_option_id": template.source_option_id,
                "source_level_index": template.source_level_index,
                "allowed": True,
            },),
            instance_binding_provenance=({
                "source": "fresh_level_public_observation",
                "level_index": level_context.level_index,
                "observation_hash": observation_hash,
                "controlled_position": list(controlled_position),
                "controlled_position_source": controlled_position_source,
                "allowed": True,
            },),
            prior_level_coordinates_used=False,
            prior_level_edges_used=False,
            prior_level_state_ids_used=False,
            prior_level_observation_hashes_used=False,
            source_probe_sequence_used_as_policy=False,
            new_level_plan_generated_from_current_state=True,
            selected=True,
            selection_reason="fresh new-level binding from transferred template",
        )

    def stale_instance_from_option(
        self,
        *,
        option: OptionCandidate,
        template: OptionTemplate,
        level_context: LevelContext,
    ) -> OptionInstance:
        expression = _selected_predicate_expression(option)
        return OptionInstance(
            instance_id=f"stale_level_{template.source_level_index}_{option.option_id}_instance",
            template_id=template.template_id,
            level_context_id=level_context.context_id,
            level_index=level_context.level_index,
            source_state_class_id=level_context.initial_state_class_id,
            target_expression={
                "type": "region_membership",
                "center": expression.get("center")
                or option.termination_condition.get("target_position"),
                "center_source": "prior_level_coordinate",
                "radius": expression.get("radius", template.radius),
                "metric": expression.get("metric", template.metric),
            },
            validation_action_sequence=(),
            template_provenance=({
                "source": "prior_level_frontier_option_template",
                "source_option_id": option.option_id,
                "source_level_index": template.source_level_index,
                "allowed": True,
            },),
            instance_binding_provenance=({
                "source": "prior_level_spatial_option_instance",
                "level_index": template.source_level_index,
                "allowed": False,
            },),
            prior_level_coordinates_used=True,
            prior_level_edges_used=False,
            prior_level_state_ids_used=False,
            prior_level_observation_hashes_used=False,
            source_probe_sequence_used_as_policy=False,
            new_level_plan_generated_from_current_state=False,
        )


class TemplateInstanceEvaluator:
    def evaluate(
        self,
        *,
        instance: OptionInstance,
        execution: FrontierProbeExecution | None,
    ) -> TemplateValidation:
        if instance.forbidden_prior_fields_present:
            return TemplateValidation(
                validation_id=f"validate_{instance.instance_id}",
                instance_id=instance.instance_id,
                result="REJECTED_STALE_INSTANCE",
                public_actions_used=False,
                validation_actions=instance.validation_action_sequence,
                semantic_guards_preserved=False,
                start_position=None,
                final_position=None,
                promoted_option_id=None,
                reason="instance binding contains prior-level spatial or policy facts",
            )
        if execution is None:
            return TemplateValidation(
                validation_id=f"validate_{instance.instance_id}",
                instance_id=instance.instance_id,
                result="UNKNOWN",
                public_actions_used=False,
                validation_actions=instance.validation_action_sequence,
                semantic_guards_preserved=False,
                start_position=None,
                final_position=None,
                promoted_option_id=None,
                reason="validation probe was not executed",
            )
        if not execution.semantic_guards_preserved:
            result: TemplateValidationResult = "SEMANTIC_GUARD_VIOLATION"
            reason = "semantic guards failed during template validation"
        elif execution.actions_executed == 0:
            result = "ACTION_CONFLICT"
            reason = "validation executed no public actions"
        elif execution.start_position is None or execution.final_position is None:
            result = "UNKNOWN"
            reason = "controlled position unavailable during validation"
        elif execution.start_position == execution.final_position:
            result = "TARGET_UNREACHABLE"
            reason = "validation did not produce stable actor movement"
        else:
            result = "VALIDATED_TEMPLATE_INSTANCE"
            reason = "fresh level-local instance validated by public probe"
        return TemplateValidation(
            validation_id=f"validate_{instance.instance_id}",
            instance_id=instance.instance_id,
            result=result,
            public_actions_used=execution.actions_executed > 0,
            validation_actions=instance.validation_action_sequence,
            semantic_guards_preserved=execution.semantic_guards_preserved,
            start_position=execution.start_position,
            final_position=execution.final_position,
            promoted_option_id=None,
            reason=reason,
        )


def promote_validated_template_instance(
    *,
    expansion_index: int,
    template: OptionTemplate,
    instance: OptionInstance,
    validation: TemplateValidation,
    execution: FrontierProbeExecution,
    source_state: OptionApplicabilityStateClass,
) -> GraphExpansion | None:
    if validation.result != "VALIDATED_TEMPLATE_INSTANCE":
        return None
    if execution.start_position is None or execution.final_position is None:
        return None
    option_id = f"level_{instance.level_index}:self_option_template_{expansion_index:03d}"
    predicate_id = f"{option_id}:pred_region"
    expression = {
        "type": "region_membership",
        "center": list(execution.final_position),
        "center_source": "fresh_level_context_validation_probe",
        "radius": template.radius,
        "metric": template.metric,
        "excludes_initiation": True,
    }
    predicate = PredicateCandidate(
        predicate_id=predicate_id,
        kind=template.predicate_family,
        abstraction_level=2,
        mode="state",
        expression=expression,
        evidence=({
            "kind": "cross_level_template_validation",
            "template_id": template.template_id,
            "instance_id": instance.instance_id,
            "source": "fresh_level_public_probe",
        },),
        validates_original=True,
        terminal_selective=True,
        true_at_initiation=False,
        true_at_terminal=True,
        earlier_true_rate=0.0,
        score=20.0,
        selected=True,
    )
    option = OptionCandidate(
        option_id=option_id,
        segment_range=(-1, -1),
        initiation_condition={
            "type": "template_instance_source_state",
            "state_class_id": source_state.state_class_id,
            "actor_position": list(execution.start_position),
            "template_id": template.template_id,
        },
        termination_condition={
            "type": "controlled_entity_reaches_template_instance_region",
            "target_position": list(execution.final_position),
            "target_source": "fresh_level_public_probe",
            "radius": template.radius,
            "forbidden_form": "prior_level_coordinate_reuse",
        },
        expected_actor_delta={
            "from_position": list(execution.start_position),
            "to_position": list(execution.final_position),
            "delta": [
                execution.final_position[0] - execution.start_position[0],
                execution.final_position[1] - execution.start_position[1],
            ],
        },
        expected_semantic_delta={"type": "none"},
        allowed_recovery_envelope={"max_actions": 12, "allowed_world_diff": "none"},
        invalidating_conditions=(
            "semantic_mismatch",
            "target_unreachable",
            "action_conflict",
        ),
        evidence=(
            {
                "kind": "cross_level_option_template_instance",
                "template_id": template.template_id,
                "instance_id": instance.instance_id,
            },
            {
                "kind": "fresh_level_public_validation_probe",
                "probe_id": execution.probe_id,
            },
        ),
        source="cross_level_template_instance",
        trace_dependency_level=4,
        predicate_abstraction_level=2,
        termination_predicate_candidates=(predicate,),
        selected_predicate_id=predicate_id,
        selected_predicate_expression_hash=predicate_expression_hash(expression),
        guard_predicates=({"type": "semantic_signature_compatible"},),
        invalidation_predicates=({"type": "semantic_guard_violation"},),
    )
    state = OptionApplicabilityStateClass(
        state_class_id=f"L{instance.level_index}_S{expansion_index + 1}",
        controlled_entity_region=f"level_{instance.level_index}_template_region_{expansion_index:03d}",
        semantic_guards=True,
        termination_predicates={predicate_id: True},
        applicable_options=(),
        non_controlled_signature_class=f"level_{instance.level_index}_public_signature",
        exact_observation_hashes=(execution.final_observation_hash,),
    )
    prediction = OptionEffectPrediction(
        prediction_id=f"pred_template_{option_id}",
        model_id="cross_level_template_effect_model",
        option_id=option.option_id,
        selected_predicate_id=predicate_id,
        selected_predicate_expression_hash=option.selected_predicate_expression_hash or "",
        start_signature_class_hash=source_state.signature_hash,
        predicted_termination_satisfied=True,
        predicted_actor_delta=option.expected_actor_delta,
        predicted_semantic_delta=option.expected_semantic_delta,
        predicted_guards_preserved=True,
        predicted_stop_reason="TERMINATION_PREDICATE_SATISFIED",
        empirical_consistency=1.0,
        support_count=1,
        total_count=1,
    )
    edge = OptionApplicabilityEdge(
        edge_id=f"edge_template_{instance.level_index}_{expansion_index:03d}",
        source_state_class_id=source_state.state_class_id,
        option_id=option.option_id,
        predicted_next_state_class_id=state.state_class_id,
        enabled_options=(),
        option_prediction=prediction,
    )
    return GraphExpansion(
        probe_id=execution.probe_id,
        source_state_class_id=source_state.state_class_id,
        new_state_class_id=state.state_class_id,
        new_option_id=option.option_id,
        new_edge_id=edge.edge_id,
        promotion_reason="validated cross-level option template instance",
        option=option,
        state_class=state,
        edge=edge,
    )


def m117_status(
    *,
    transition_handled: bool,
    template_count: int,
    template_contains_level_local_fact: bool,
    instance_count: int,
    instance_binding_from_prior_level_fact: bool,
    stale_negative_control_ran: bool,
    stale_instance_rejected: bool,
    validation_probe_public: bool,
    validated_instance_count: int,
    promoted_instance_count: int,
    promoted_instance_scored: bool,
    template_contribution_detected: bool,
    stale_spatial_instance_reused: bool,
    prior_coordinate_leakage: bool,
    prior_edge_leakage: bool,
    source_probe_script_reused: bool,
    trace_leakage: bool,
    oracle_signal: bool,
) -> M117Status:
    if trace_leakage:
        return "failed_trace_leakage"
    if oracle_signal:
        return "failed_oracle_signal"
    if stale_spatial_instance_reused:
        return "failed_stale_spatial_instance_reused"
    if prior_coordinate_leakage:
        return "failed_prior_level_coordinate_leakage"
    if prior_edge_leakage:
        return "failed_prior_level_edge_leakage"
    if source_probe_script_reused:
        return "failed_source_probe_script_reused"
    if template_contains_level_local_fact:
        return "failed_template_contains_level_local_fact"
    if instance_binding_from_prior_level_fact:
        return "failed_instance_binding_from_prior_level_fact"
    if not stale_negative_control_ran:
        return "failed_no_stale_negative_control"
    if not stale_instance_rejected:
        return "failed_stale_spatial_instance_reused"
    if not transition_handled or template_count == 0:
        return "partial_template_extracted_only"
    if instance_count == 0:
        return "partial_template_extracted_only"
    if not validation_probe_public:
        return "failed_template_validation_not_public"
    if validated_instance_count == 0:
        return "partial_no_valid_template_instance"
    if promoted_instance_count == 0:
        return "partial_template_validated_not_used"
    if not promoted_instance_scored:
        return "partial_template_validated_not_used"
    if not template_contribution_detected:
        return "failed_template_no_contribution"
    return "complete_with_known_limits"


def cross_level_transfer_findings(payload: dict[str, Any]) -> tuple[str, ...]:
    encoded = json.dumps(payload, sort_keys=True)
    forbidden_markers = (
        "human_trace",
        "public_trace_segment",
        "trace_corridor",
        "human_action",
        "raw_action",
        "action_at_index",
        "private",
        "changer",
        "clock",
        "goal",
        "trap",
    )
    return tuple(marker for marker in forbidden_markers if marker in encoded)


def _selected_predicate_expression(option: OptionCandidate) -> dict[str, Any]:
    if option.termination_predicate_candidates:
        for candidate in option.termination_predicate_candidates:
            if candidate.selected:
                return dict(candidate.expression)
    return dict(option.termination_condition)


def _fresh_validation_action(available_actions: tuple[int, ...]) -> int:
    legal = tuple(action for action in available_actions if action in {1, 2, 3, 4})
    if not legal:
        return 1
    return max(legal)


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
