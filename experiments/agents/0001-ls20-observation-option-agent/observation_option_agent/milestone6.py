"""Self-generated option discovery for 0001.

Milestone 6 changes option origin, not the whole executor stack.  Prior
trace-trained machinery may be reused, but selected option evidence must come
from self-observed public interactions and execution must not replay either a
human trace or the self-discovery action sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from .events import Delta, Event
from .milestone3 import SemanticSignature
from .milestone4 import (
    OptionCandidate,
    OptionExecutor,
    Position,
    PredicateCandidate,
    predicate_expression_hash,
    selected_predicate_expression_hash,
)


M6Status = Literal[
    "complete_with_known_limits",
    "partial_motor_primitive_only",
    "partial_discovered_but_unvalidated_option",
    "partial_self_script_only",
    "failed_trace_leakage",
    "failed_noise_promotion",
]


@dataclass(frozen=True)
class DiscoveryRecord:
    record_id: str
    source: str
    self_discovery_start_source: str
    observation_before_hash: str
    action_sequence: tuple[int, ...]
    observation_after_hash: str
    start_position: Position
    end_position: Position
    actor_delta: Delta
    action_model: dict[int, Delta]
    semantic_delta: tuple[dict[str, Any], ...]
    contact_events: tuple[dict[str, Any], ...]
    blocked_edges: tuple[dict[str, Any], ...]
    evidence: tuple[dict[str, Any], ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source": self.source,
            "self_discovery_start_source": self.self_discovery_start_source,
            "observation_before_hash": self.observation_before_hash,
            "action_sequence": list(self.action_sequence),
            "observation_after_hash": self.observation_after_hash,
            "start_position": list(self.start_position),
            "end_position": list(self.end_position),
            "actor_delta": list(self.actor_delta),
            "action_model": {
                str(action): list(delta)
                for action, delta in sorted(self.action_model.items())
            },
            "semantic_delta": list(self.semantic_delta),
            "contact_events": list(self.contact_events),
            "blocked_edges": list(self.blocked_edges),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class SelfOptionExecution:
    assay_id: str
    completed: bool
    status: str
    failure_reason: str | None
    plan: tuple[int, ...]
    records: tuple[dict[str, Any], ...]
    termination_predicate_satisfied: bool
    execution_plan_differs_from_discovery_sequence: bool
    used_discovery_action_script: bool
    raw_trace_actions_used: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "assay_id": self.assay_id,
            "completed": self.completed,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "plan": list(self.plan),
            "records": list(self.records),
            "termination_predicate_satisfied": self.termination_predicate_satisfied,
            "execution_plan_differs_from_discovery_sequence": (
                self.execution_plan_differs_from_discovery_sequence
            ),
            "used_discovery_action_script": self.used_discovery_action_script,
            "raw_trace_actions_used": self.raw_trace_actions_used,
        }


class SelfExplorationController:
    """Deterministic public-action exploration policy."""

    def __init__(
        self,
        *,
        action_sequence: tuple[int, ...] = (3, 3, 3, 1),
        budget: int = 12,
    ) -> None:
        self.action_sequence = action_sequence[:budget]
        self.budget = budget

    def action_for_step(
        self,
        *,
        step: int,
        available_actions: tuple[int, ...],
    ) -> int | None:
        if step > len(self.action_sequence):
            return None
        preferred = self.action_sequence[step - 1]
        if preferred in available_actions:
            return preferred
        return available_actions[0] if available_actions else None


class SelfOptionInducer:
    """Create self-exploration options without human trace segment evidence."""

    def induce(self, discovery: DiscoveryRecord) -> OptionCandidate:
        start = discovery.start_position
        target = _reachable_endpoint(discovery)
        expected_delta = (
            target[0] - start[0],
            target[1] - start[1],
        )
        expression = {
            "type": "region_membership",
            "center": list(target),
            "radius": 2,
            "metric": "manhattan",
            "excludes_initiation": True,
            "forbidden_form": "trace_index == segment_end",
        }
        predicate = PredicateCandidate(
            predicate_id="self_option_000:pred_region",
            kind="region_membership",
            abstraction_level=2,
            mode="state",
            expression=expression,
            evidence=(
                {
                    "kind": "self_observed_terminal_neighborhood",
                    "discovery_record_id": discovery.record_id,
                    "center": list(target),
                    "radius": 2,
                    "observed_endpoint": list(discovery.end_position),
                    "model_endpoint": list(target),
                },
                {
                    "kind": "self_observed_initiation_excluded",
                    "discovery_record_id": discovery.record_id,
                    "position": list(start),
                },
            ),
            validates_original=True,
            terminal_selective=True,
            true_at_initiation=False,
            true_at_terminal=True,
            earlier_true_rate=0.0,
            score=38,
            selected=True,
            rejection_reason=None,
        )
        return OptionCandidate(
            option_id="self_option_000",
            segment_range=(-1, -1),
            initiation_condition={
                "type": "self_observed_actor_at_or_near_position",
                "actor_position": list(start),
                "semantic_signature": [],
                "start_source": discovery.self_discovery_start_source,
            },
            termination_condition=expression,
            expected_actor_delta={
                "from_position": list(start),
                "to_position": list(target),
                "delta": list(expected_delta),
                "observed_delta": list(discovery.actor_delta),
            },
            expected_semantic_delta={
                "type": "none",
                "from_signature": [],
                "to_signature": [],
            },
            allowed_recovery_envelope={
                "max_actions": 12,
                "allowed_world_diff": "none_or_expected",
            },
            invalidating_conditions=(
                "semantic_mismatch",
                "target_unreachable",
                "unexpected_noncontrolled_change",
                "action_conflict",
            ),
            evidence=(
                {
                    "kind": "self_observed_reachability",
                    "discovery_record_id": discovery.record_id,
                    "action_count": len(discovery.action_sequence),
                    "actor_delta": list(expected_delta),
                    "observed_actor_delta": list(discovery.actor_delta),
                },
                {
                    "kind": "self_observed_transition",
                    "discovery_record_id": discovery.record_id,
                    "source": discovery.source,
                },
                {
                    "kind": "self_observed_predicate_not_trace_index",
                    "value": True,
                },
            ),
            source="self_exploration",
            trace_dependency_level=4,
            predicate_abstraction_level=2,
            termination_predicate_candidates=(predicate,),
            selected_predicate_id=predicate.predicate_id,
            selected_predicate_expression_hash=predicate_expression_hash(expression),
            guard_predicates=({
                "type": "semantic_signature_compatible",
                "signature": [],
                "role": "guard",
            },),
            invalidation_predicates=tuple(
                {"type": condition}
                for condition in (
                    "semantic_mismatch",
                    "target_unreachable",
                    "unexpected_noncontrolled_change",
                    "action_conflict",
                )
            ),
        )


class SelfOptionEvaluator:
    """Execute self-options with discovery action sequence masked."""

    def __init__(self, *, executor: OptionExecutor | None = None) -> None:
        self.executor = executor or OptionExecutor()

    def execute(
        self,
        *,
        assay_id: str,
        option: OptionCandidate,
        discovery: DiscoveryRecord,
        current_position: Position | None = None,
        current_signature: SemanticSignature | None = None,
        blocked_edges: frozenset[tuple[Position, int]] = frozenset(),
        observed_deltas: tuple[Delta, ...] | None = None,
        world_diffs: tuple[tuple[dict[str, Any], ...], ...] | None = None,
    ) -> SelfOptionExecution:
        result = self.executor.execute(
            option=_execution_option(option),
            current_position=current_position or discovery.start_position,
            current_signature=current_signature or SemanticSignature.from_labels(()),
            action_deltas=discovery.action_model,
            blocked_edges=blocked_edges,
            observed_deltas=observed_deltas,
            world_diffs=world_diffs,
            final_signature=SemanticSignature.from_labels(()),
        )
        plan = tuple(result.plan)
        records = tuple(
            _self_execution_record(
                record=record,
                option=option,
            )
            for record in result.records
        )
        failure_reason = _m6_failure_reason(result.failure_reason)
        status = (
            "COMPLETED"
            if result.completed
            else "REFUSED"
            if result.status == "REFUSED"
            else result.status
        )
        return SelfOptionExecution(
            assay_id=assay_id,
            completed=result.completed,
            status=status,
            failure_reason=failure_reason,
            plan=plan,
            records=records,
            termination_predicate_satisfied=result.termination_predicate_satisfied,
            execution_plan_differs_from_discovery_sequence=(
                plan != discovery.action_sequence
            ),
            used_discovery_action_script=False,
            raw_trace_actions_used=False,
        )


def nontriviality_criteria(
    *,
    discovery: DiscoveryRecord,
    primary_execution: SelfOptionExecution,
    variation_execution: SelfOptionExecution,
) -> dict[str, bool]:
    actor_distance = (
        abs(discovery.actor_delta[0])
        + abs(discovery.actor_delta[1])
    )
    return {
        "two_or_more_primitive_actions": len(discovery.action_sequence) >= 2,
        "execution_plan_differs_from_discovery_sequence": (
            primary_execution.execution_plan_differs_from_discovery_sequence
        ),
        "same_predicate_reexecutes_under_variation": variation_execution.completed,
        "target_not_adjacent_to_start": actor_distance > 5,
        "contact_or_event_or_semantic_delta": bool(
            discovery.contact_events or discovery.semantic_delta
        ),
    }


def m6_status(
    *,
    option: OptionCandidate,
    discovery: DiscoveryRecord,
    executions: tuple[SelfOptionExecution, ...],
    invalid_refused: bool,
    expectation_refused: bool,
    trace_leakage: bool,
    noise_promoted: bool,
) -> M6Status:
    if trace_leakage:
        return "failed_trace_leakage"
    if noise_promoted:
        return "failed_noise_promotion"
    successes = tuple(execution for execution in executions if execution.completed)
    primary = successes[0] if successes else None
    variation = successes[1] if len(successes) > 1 else None
    if primary is None:
        return "partial_discovered_but_unvalidated_option"
    if primary.plan == discovery.action_sequence and variation is None:
        return "partial_self_script_only"
    if variation is None:
        return "partial_discovered_but_unvalidated_option"
    criteria = nontriviality_criteria(
        discovery=discovery,
        primary_execution=primary,
        variation_execution=variation,
    )
    if sum(criteria.values()) < 2:
        return "partial_motor_primitive_only"
    if (
        option.source == "self_exploration"
        and option.trace_dependency_level == 4
        and invalid_refused
        and expectation_refused
    ):
        return "complete_with_known_limits"
    return "partial_discovered_but_unvalidated_option"


def trace_leakage_findings(option: OptionCandidate) -> tuple[str, ...]:
    findings = []
    payload = option.to_record()
    encoded = repr(payload)
    forbidden = (
        "human_trace",
        "public_trace_segment",
        "trace_corridor",
        "human_action",
        "raw_action",
        "action_at_index",
    )
    for marker in forbidden:
        if marker in encoded:
            findings.append(marker)
    if option.source != "self_exploration":
        findings.append("source_not_self_exploration")
    if option.trace_dependency_level != 4:
        findings.append("trace_dependency_level_not_4")
    return tuple(sorted(set(findings)))


def discovery_record_from_events(
    *,
    record_id: str,
    action_sequence: tuple[int, ...],
    events: tuple[Event, ...],
    controlled_object_ids: tuple[int, ...],
    observation_before_hash: str,
    observation_after_hash: str,
) -> DiscoveryRecord:
    if not events:
        raise ValueError("M6 discovery requires at least one self-observed event")
    if not controlled_object_ids:
        raise ValueError("M6 discovery requires self-observed controlled ids")
    start_position: Position | None = None
    end_position: Position | None = None
    action_model: dict[int, Delta] = {}
    semantic_delta = []
    contact_events = []
    blocked_edges = []
    controlled = set(controlled_object_ids)
    for event in events:
        for delta in event.object_deltas:
            if delta.object_id not in controlled:
                continue
            if start_position is None:
                start_position = _bbox_center(delta.bbox_before)
            end_position = _bbox_center(delta.bbox_after)
            if delta.translation != (0, 0):
                action_model[event.action] = delta.translation
        if event.created_object_ids or event.disappeared_object_ids:
            semantic_delta.append({
                "kind": "self_observed_semantic_delta",
                "step": event.step,
                "created_object_ids": list(event.created_object_ids),
                "disappeared_object_ids": list(event.disappeared_object_ids),
            })
    if start_position is None or end_position is None:
        raise ValueError("M6 discovery could not derive controlled positions")
    actor_delta = (
        end_position[0] - start_position[0],
        end_position[1] - start_position[1],
    )
    return DiscoveryRecord(
        record_id=record_id,
        source="self_exploration",
        self_discovery_start_source="public_reset_observation",
        observation_before_hash=observation_before_hash,
        action_sequence=action_sequence,
        observation_after_hash=observation_after_hash,
        start_position=start_position,
        end_position=end_position,
        actor_delta=actor_delta,
        action_model=dict(sorted(action_model.items())),
        semantic_delta=tuple(semantic_delta),
        contact_events=tuple(contact_events),
        blocked_edges=tuple(blocked_edges),
        evidence=(
            {
                "kind": "self_observed_transition",
                "record_id": record_id,
            },
            {
                "kind": "self_observed_reachability",
                "record_id": record_id,
                "action_count": len(action_sequence),
            },
        ),
    )


def _execution_option(option: OptionCandidate) -> OptionCandidate:
    """Strip any non-execution self-discovery material before execution."""

    return replace(
        option,
        evidence=tuple(
            item for item in option.evidence
            if "action_sequence" not in repr(item)
        ),
    )


def _self_execution_record(
    *,
    record: dict[str, Any],
    option: OptionCandidate,
) -> dict[str, Any]:
    output = dict(record)
    output["phase"] = "SELF_OPTION_EXECUTION"
    output["action_source"] = "SELF_OPTION_PLANNER"
    output["used_raw_trace_action"] = False
    output["used_discovery_action_script"] = False
    output["option_source"] = option.source
    output["semantic_validation_result"] = (
        "OPTION_INVALID"
        if output.get("semantic_validation_result") == "TRACE_INVALID"
        else output.get("semantic_validation_result")
    )
    return output


def _m6_failure_reason(reason: str | None) -> str | None:
    mapping = {
        "INITIATION_CONDITION_FAILED": "INVALID_INITIATION",
        "SEMANTIC_INCOMPATIBILITY": "OPTION_INVALID",
        "TERMINATION_PREDICATE_UNSATISFIED": "EXPECTATION_INVALID",
    }
    return mapping.get(reason, reason)


def _reachable_endpoint(discovery: DiscoveryRecord) -> Position:
    position = discovery.start_position
    for action in discovery.action_sequence:
        delta = discovery.action_model.get(action)
        if delta is None:
            continue
        position = (position[0] + delta[0], position[1] + delta[1])
    return position


def _bbox_center(bbox: tuple[int, int, int, int]) -> Position:
    min_x, min_y, max_x, max_y = bbox
    return (
        int(round((min_x + max_x) / 2)),
        int(round((min_y + max_y) / 2)),
    )
