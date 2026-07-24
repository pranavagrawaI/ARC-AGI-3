"""Trace-to-option abstraction for 0001.

Milestone 4 separates option induction from option execution. Induction may
inspect public demonstration evidence. Execution receives only an induced
option predicate, never the raw action list for the segment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Literal

from .events import Delta
from .milestone2 import TraceReference
from .milestone3 import (
    LocalRecoveryPlanner,
    SemanticSignature,
    StructuredRecoveryAgent,
    TraceCorridor,
)


TraceDependencyLevel = Literal[0, 1, 2, 3, 4, 5]
PredicateMode = Literal["state", "transition"]
OptionExecutionStatus = Literal[
    "COMPLETED",
    "REFUSED",
    "ACTION_CONFLICT",
    "WORLD_DRIFT",
]
Position = tuple[int, int]


@dataclass(frozen=True)
class PredicateCandidate:
    predicate_id: str
    kind: str
    abstraction_level: int
    mode: PredicateMode
    expression: dict[str, Any]
    evidence: tuple[dict[str, Any], ...]
    validates_original: bool
    terminal_selective: bool
    true_at_initiation: bool
    true_at_terminal: bool
    earlier_true_rate: float
    score: float
    selected: bool = False
    rejection_reason: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "predicate_id": self.predicate_id,
            "kind": self.kind,
            "abstraction_level": self.abstraction_level,
            "mode": self.mode,
            "expression": self.expression,
            "expression_hash": predicate_expression_hash(self.expression),
            "evidence": list(self.evidence),
            "validates_original": self.validates_original,
            "terminal_selective": self.terminal_selective,
            "true_at_initiation": self.true_at_initiation,
            "true_at_terminal": self.true_at_terminal,
            "earlier_true_rate": self.earlier_true_rate,
            "score": self.score,
            "selected": self.selected,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class OptionCandidate:
    option_id: str
    segment_range: tuple[int, int]
    initiation_condition: dict[str, Any]
    termination_condition: dict[str, Any]
    expected_actor_delta: dict[str, Any]
    expected_semantic_delta: dict[str, Any]
    allowed_recovery_envelope: dict[str, Any]
    invalidating_conditions: tuple[str, ...]
    evidence: tuple[dict[str, Any], ...]
    source: str = "human_trace"
    trace_dependency_level: TraceDependencyLevel = 3
    predicate_abstraction_level: int = 1
    termination_predicate_candidates: tuple[PredicateCandidate, ...] = ()
    selected_predicate_id: str | None = None
    selected_predicate_expression_hash: str | None = None
    guard_predicates: tuple[dict[str, Any], ...] = ()
    invalidation_predicates: tuple[dict[str, Any], ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "segment_range": list(self.segment_range),
            "initiation_condition": self.initiation_condition,
            "termination_condition": self.termination_condition,
            "expected_actor_delta": self.expected_actor_delta,
            "expected_semantic_delta": self.expected_semantic_delta,
            "allowed_recovery_envelope": self.allowed_recovery_envelope,
            "invalidating_conditions": list(self.invalidating_conditions),
            "evidence": list(self.evidence),
            "source": self.source,
            "trace_dependency_level": self.trace_dependency_level,
            "predicate_abstraction_level": self.predicate_abstraction_level,
            "termination_predicate_candidates": [
                candidate.to_record()
                for candidate in self.termination_predicate_candidates
            ],
            "selected_predicate_id": self.selected_predicate_id,
            "selected_predicate_expression_hash": self.selected_predicate_expression_hash,
            "guard_predicates": list(self.guard_predicates),
            "invalidation_predicates": list(self.invalidation_predicates),
        }


class OptionSegmenter:
    """Induce state-based option candidates from a trace reference."""

    def induce(self, reference: TraceReference) -> tuple[OptionCandidate, ...]:
        boundaries = self.boundaries(reference)
        candidates = []
        for start, end in zip(boundaries, boundaries[1:]):
            if end <= start:
                continue
            start_state = reference.state(start)
            end_state = reference.state(end)
            if start_state.controlled_center is None or end_state.controlled_center is None:
                continue
            start_position = _round_position(start_state.controlled_center)
            end_position = _round_position(end_state.controlled_center)
            if start_position == end_position and start_state.nearby_signature == end_state.nearby_signature:
                continue
            semantic_delta = (
                "none"
                if start_state.nearby_signature == end_state.nearby_signature
                else "signature_change"
            )
            candidates.append(OptionCandidate(
                option_id=f"option_{start:03d}_{end:03d}",
                segment_range=(start, end),
                initiation_condition={
                    "type": "actor_at_or_near_position_with_semantic_signature",
                    "actor_position": list(start_position),
                    "semantic_signature": _labels(start_state.nearby_signature),
                },
                termination_condition={
                    "type": "actor_reaches_observable_position",
                    "target_position": list(end_position),
                    "tolerance": 0,
                    "forbidden_form": "trace_index == segment_end",
                },
                expected_actor_delta={
                    "from_position": list(start_position),
                    "to_position": list(end_position),
                    "delta": [
                        end_position[0] - start_position[0],
                        end_position[1] - start_position[1],
                    ],
                },
                expected_semantic_delta={
                    "type": semantic_delta,
                    "from_signature": _labels(start_state.nearby_signature),
                    "to_signature": _labels(end_state.nearby_signature),
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
                        "kind": "public_trace_segment",
                        "start": start,
                        "end": end,
                    },
                    {
                        "kind": "controlled_entity_delta",
                        "from": list(start_position),
                        "to": list(end_position),
                    },
                    {
                        "kind": "termination_predicate_not_trace_index",
                        "value": True,
                    },
                ),
                trace_dependency_level=3,
                predicate_abstraction_level=1,
            ))
        return tuple(candidates)

    def boundaries(self, reference: TraceReference) -> tuple[int, ...]:
        first = _first_controlled_step(reference)
        if first is None:
            return (0, reference.completion_step)

        boundaries = [first]
        previous_delta: Delta | None = None
        previous_signature = reference.state(first).nearby_signature
        for index in range(first + 1, reference.completion_step + 1):
            state = reference.state(index)
            if state.nearby_signature != previous_signature:
                boundaries.append(index)
                previous_signature = state.nearby_signature
                previous_delta = None
                continue
            action = reference.state(index - 1).action
            delta = reference.action_model.get(action) if action is not None else None
            if previous_delta is not None and delta is not None and delta != previous_delta:
                boundaries.append(index - 1)
            if delta is not None:
                previous_delta = delta
        if boundaries[-1] != reference.completion_step:
            boundaries.append(reference.completion_step)
        return _dedupe_sorted(boundaries)


class PredicateInducer:
    """Generate and select discriminative option termination predicates."""

    def enrich(
        self,
        *,
        option: OptionCandidate,
        reference: TraceReference,
        validates_original: bool = True,
    ) -> OptionCandidate:
        candidates = self.generate(
            option=option,
            reference=reference,
            validates_original=validates_original,
        )
        selected = self.select(candidates)
        if selected is None:
            selected = _candidate_by_kind(candidates, "exact_position")
        if selected is None:
            return option
        selected_candidates = tuple(
            replace(
                candidate,
                selected=candidate.predicate_id == selected.predicate_id,
                rejection_reason=(
                    None
                    if candidate.predicate_id == selected.predicate_id
                    else candidate.rejection_reason or "valid fallback but lower score"
                ),
            )
            for candidate in candidates
        )
        return replace(
            option,
            termination_condition=selected.expression,
            predicate_abstraction_level=selected.abstraction_level,
            termination_predicate_candidates=selected_candidates,
            selected_predicate_id=selected.predicate_id,
            selected_predicate_expression_hash=predicate_expression_hash(selected.expression),
            guard_predicates=self.guard_predicates(option),
            invalidation_predicates=tuple(
                {"type": condition}
                for condition in option.invalidating_conditions
            ),
        )

    def generate(
        self,
        *,
        option: OptionCandidate,
        reference: TraceReference,
        validates_original: bool,
    ) -> tuple[PredicateCandidate, ...]:
        start, end = option.segment_range
        start_position = _position_from(option.initiation_condition["actor_position"])
        target_position = _position_from(option.expected_actor_delta["to_position"])
        earlier_positions = [
            _round_position(reference.state(index).controlled_center)
            for index in range(start, end)
            if reference.state(index).controlled_center is not None
        ]
        candidates = [
            self._candidate(
                predicate_id=f"{option.option_id}:pred_exact",
                kind="exact_position",
                abstraction_level=1,
                mode="state",
                expression={
                    "type": "exact_position",
                    "target_position": list(target_position),
                    "tolerance": 0,
                    "forbidden_form": "trace_index == segment_end",
                },
                start_position=start_position,
                target_position=target_position,
                earlier_positions=earlier_positions,
                validates_original=validates_original,
                evidence=(
                    {"kind": "terminal_actor_position", "position": list(target_position)},
                ),
            ),
            self._region_candidate(
                option=option,
                start_position=start_position,
                target_position=target_position,
                earlier_positions=earlier_positions,
                validates_original=validates_original,
            ),
        ]
        candidates.append(self._semantic_guard_candidate(option, start_position, target_position))
        return tuple(candidates)

    def _region_candidate(
        self,
        *,
        option: OptionCandidate,
        start_position: Position,
        target_position: Position,
        earlier_positions: list[Position],
        validates_original: bool,
    ) -> PredicateCandidate:
        radius = max(1, min(2, _distance(start_position, target_position) - 1))
        return self._candidate(
            predicate_id=f"{option.option_id}:pred_region",
            kind="region_membership",
            abstraction_level=2,
            mode="state",
            expression={
                "type": "region_membership",
                "center": list(target_position),
                "radius": radius,
                "metric": "manhattan",
                "excludes_initiation": True,
                "forbidden_form": "trace_index == segment_end",
            },
            start_position=start_position,
            target_position=target_position,
            earlier_positions=earlier_positions,
            validates_original=validates_original,
            evidence=(
                {"kind": "terminal_neighborhood", "center": list(target_position), "radius": radius},
                {"kind": "initiation_excluded", "position": list(start_position)},
            ),
        )

    def select(self, candidates: tuple[PredicateCandidate, ...]) -> PredicateCandidate | None:
        survivors = [
            candidate
            for candidate in candidates
            if candidate.rejection_reason is None
        ]
        if not survivors:
            return None
        return max(
            survivors,
            key=lambda candidate: (
                candidate.score,
                candidate.abstraction_level,
                candidate.predicate_id,
            ),
        )

    def guard_predicates(self, option: OptionCandidate) -> tuple[dict[str, Any], ...]:
        return ({
            "type": "semantic_signature_compatible",
            "signature": option.expected_semantic_delta.get("from_signature", ()),
            "role": "guard",
        },)

    def _candidate(
        self,
        *,
        predicate_id: str,
        kind: str,
        abstraction_level: int,
        mode: PredicateMode,
        expression: dict[str, Any],
        start_position: Position,
        target_position: Position,
        earlier_positions: list[Position],
        validates_original: bool,
        evidence: tuple[dict[str, Any], ...],
    ) -> PredicateCandidate:
        true_at_initiation = evaluate_position_expression(expression, start_position)
        true_at_terminal = evaluate_position_expression(expression, target_position)
        earlier_true = sum(evaluate_position_expression(expression, position) for position in earlier_positions)
        earlier_rate = earlier_true / len(earlier_positions) if earlier_positions else 0.0
        terminal_selective = (
            not true_at_initiation
            and true_at_terminal
            and earlier_rate < 0.5
        )
        rejection = _gate_rejection(
            expression=expression,
            evidence=evidence,
            validates_original=validates_original,
            true_at_initiation=true_at_initiation,
            true_at_terminal=true_at_terminal,
            terminal_selective=terminal_selective,
        )
        score = (
            10 * abstraction_level
            + 5 * len(evidence)
            + (5 if terminal_selective else 0)
            + (3 if validates_original else 0)
            - (8 if earlier_rate >= 0.5 else 0)
            - (6 if kind == "exact_position" else 0)
            - (10 if _is_index_based(expression) else 0)
        )
        return PredicateCandidate(
            predicate_id=predicate_id,
            kind=kind,
            abstraction_level=abstraction_level,
            mode=mode,
            expression=expression,
            evidence=evidence,
            validates_original=validates_original,
            terminal_selective=terminal_selective,
            true_at_initiation=true_at_initiation,
            true_at_terminal=true_at_terminal,
            earlier_true_rate=round(earlier_rate, 4),
            score=score,
            rejection_reason=rejection,
        )

    def _semantic_guard_candidate(
        self,
        option: OptionCandidate,
        start_position: Position,
        target_position: Position,
    ) -> PredicateCandidate:
        expression = {
            "type": "semantic_signature_compatible",
            "signature": option.expected_semantic_delta.get("to_signature", ()),
            "role": "guard_not_termination",
        }
        return PredicateCandidate(
            predicate_id=f"{option.option_id}:pred_semantic_guard",
            kind="semantic_signature_compatibility",
            abstraction_level=4,
            mode="state",
            expression=expression,
            evidence=(
                {
                    "kind": "semantic_signature_context",
                    "signature": option.expected_semantic_delta.get("to_signature", ()),
                },
            ),
            validates_original=True,
            terminal_selective=False,
            true_at_initiation=True,
            true_at_terminal=True,
            earlier_true_rate=1.0,
            score=0,
            rejection_reason="classified_as_guard_not_termination",
        )


class HumanTraceReferenceProvider:
    def __init__(self, reference: TraceReference) -> None:
        self.reference = reference

    def candidate_states(self, *, start_index: int, length: int = 5) -> TraceCorridor:
        return TraceCorridor.from_reference(
            self.reference,
            start_index=start_index,
            length=length,
        )


class OptionPredicateReferenceProvider:
    def corridor_for(self, option: OptionCandidate) -> TraceCorridor:
        target = termination_target(option)
        labels = tuple(
            (label,)
            for label in option.expected_semantic_delta.get("to_signature", ())
        )
        return TraceCorridor.synthetic([
            (
                option.segment_range[1],
                target,
                labels,
            )
        ])


class WorldModelPredictionProvider:
    """Placeholder interface for the later trace-free provider."""

    def candidate_states(self, *_args, **_kwargs) -> TraceCorridor:  # pragma: no cover - contract only
        raise NotImplementedError("world-model prediction provider is a later milestone")


@dataclass(frozen=True)
class OptionExecutionResult:
    status: OptionExecutionStatus
    completed: bool
    termination_predicate_satisfied: bool
    trace_dependency_level: TraceDependencyLevel
    raw_action_list_masked: bool
    plan: tuple[int, ...]
    records: tuple[dict[str, Any], ...]
    failure_reason: str | None
    semantic_validation_result: str


class OptionExecutor:
    """Execute an option predicate without segment action access."""

    def __init__(
        self,
        *,
        planner: LocalRecoveryPlanner | None = None,
        recovery_agent: StructuredRecoveryAgent | None = None,
        provider: OptionPredicateReferenceProvider | None = None,
    ) -> None:
        self.planner = planner or LocalRecoveryPlanner(budget=12)
        self.recovery_agent = recovery_agent or StructuredRecoveryAgent()
        self.provider = provider or OptionPredicateReferenceProvider()

    def execute(
        self,
        *,
        option: OptionCandidate,
        current_position: Position,
        current_signature: SemanticSignature,
        action_deltas: dict[int, Delta],
        valid_positions: frozenset[Position] | None = None,
        blocked_edges: frozenset[tuple[Position, int]] = frozenset(),
        observed_deltas: tuple[Delta, ...] | None = None,
        world_diffs: tuple[tuple[dict[str, Any], ...], ...] | None = None,
        final_signature: SemanticSignature | None = None,
    ) -> OptionExecutionResult:
        if not self._initiation_satisfied(option, current_position, current_signature):
            return OptionExecutionResult(
                status="REFUSED",
                completed=False,
                termination_predicate_satisfied=False,
                trace_dependency_level=option.trace_dependency_level,
                raw_action_list_masked=True,
                plan=(),
                records=(),
                failure_reason="INITIATION_CONDITION_FAILED",
                semantic_validation_result="TRACE_INVALID",
            )

        corridor = self.provider.corridor_for(option)
        plan = self.planner.plan(
            start=current_position,
            corridor=corridor,
            action_deltas=action_deltas,
            blocked_edges=blocked_edges,
            valid_positions=valid_positions,
        )
        if not plan.recovered:
            return OptionExecutionResult(
                status="REFUSED",
                completed=False,
                termination_predicate_satisfied=False,
                trace_dependency_level=option.trace_dependency_level,
                raw_action_list_masked=True,
                plan=(),
                records=(),
                failure_reason=plan.failure_reason,
                semantic_validation_result="TRACE_INVALID",
            )

        expected = tuple(action_deltas[action] for action in plan.plan)
        observed = observed_deltas if observed_deltas is not None else expected
        diffs = world_diffs if world_diffs is not None else tuple(() for _ in plan.plan)
        execution = self.recovery_agent.execute_recovery_plan(
            plan=plan.plan,
            start=current_position,
            action_deltas=action_deltas,
            observed_deltas=observed,
            world_diffs=diffs,
        )
        records = self._option_records(
            option=option,
            plan=plan.plan,
            start=current_position,
            action_deltas=action_deltas,
            observed_deltas=observed,
            world_diffs=diffs,
        )
        if execution.status != "RECOVERED":
            executed_records = records[:len(execution.records)]
            return OptionExecutionResult(
                status="WORLD_DRIFT" if execution.status == "WORLD_DRIFT" else "ACTION_CONFLICT",
                completed=False,
                termination_predicate_satisfied=False,
                trace_dependency_level=option.trace_dependency_level,
                raw_action_list_masked=True,
                plan=plan.plan,
                records=executed_records,
                failure_reason=execution.failure_reason,
                semantic_validation_result="TRACE_INVALID",
            )

        final_position = _apply(current_position, observed)
        validation_signature = final_signature or current_signature
        satisfied = self.termination_satisfied(
            option=option,
            current_position=final_position,
            current_signature=validation_signature,
        )
        return OptionExecutionResult(
            status="COMPLETED" if satisfied else "REFUSED",
            completed=satisfied,
            termination_predicate_satisfied=satisfied,
            trace_dependency_level=option.trace_dependency_level,
            raw_action_list_masked=True,
            plan=plan.plan,
            records=records,
            failure_reason=None if satisfied else "TERMINATION_PREDICATE_UNSATISFIED",
            semantic_validation_result="VALID" if satisfied else "TRACE_INVALID",
        )

    def termination_satisfied(
        self,
        *,
        option: OptionCandidate,
        current_position: Position,
        current_signature: SemanticSignature,
    ) -> bool:
        expression = selected_termination_expression(option)
        target_signature = SemanticSignature.from_labels(
            option.expected_semantic_delta.get("to_signature", ())
        )
        return (
            evaluate_position_expression(expression, current_position)
            and current_signature.compatible_with(target_signature)
        )

    def _initiation_satisfied(
        self,
        option: OptionCandidate,
        current_position: Position,
        current_signature: SemanticSignature,
    ) -> bool:
        expected_position = tuple(option.initiation_condition["actor_position"])
        expected_signature = SemanticSignature.from_labels(
            option.initiation_condition.get("semantic_signature", ())
        )
        return (
            current_position == (int(expected_position[0]), int(expected_position[1]))
            and current_signature.compatible_with(expected_signature)
        )

    def _option_records(
        self,
        *,
        option: OptionCandidate,
        plan: tuple[int, ...],
        start: Position,
        action_deltas: dict[int, Delta],
        observed_deltas: tuple[Delta, ...],
        world_diffs: tuple[tuple[dict[str, Any], ...], ...],
    ) -> tuple[dict[str, Any], ...]:
        records = []
        position = start
        expression = selected_termination_expression(option)
        target_position = termination_target(option)
        for step, action in enumerate(plan, start=1):
            before = _distance(position, target_position)
            observed = observed_deltas[step - 1]
            after_position = (position[0] + observed[0], position[1] + observed[1])
            after = _distance(after_position, target_position)
            world_diff = list(world_diffs[step - 1])
            records.append({
                "step": step,
                "phase": "OPTION_EXECUTION",
                "action": action,
                "action_source": "OPTION_PLANNER",
                "target_predicate": expression["type"],
                "selected_predicate_id": option.selected_predicate_id,
                "selected_predicate_expression_hash": selected_predicate_expression_hash(option),
                "distance_to_termination_before": before,
                "distance_to_termination_after": after,
                "predicate_progress": {
                    "kind": _progress_kind(expression),
                    "before": before,
                    "after": after,
                },
                "predicted_delta": list(action_deltas[action]),
                "observed_delta": list(observed),
                "world_diff": world_diff,
                "semantic_validation_result": "TRACE_INVALID" if world_diff else "VALID",
                "used_raw_trace_action": False,
            })
            position = after_position
        return tuple(records)


def replay_actions_to_position(
    *,
    start: Position,
    actions: tuple[int, ...],
    action_deltas: dict[int, Delta],
) -> Position:
    position = start
    for action in actions:
        delta = action_deltas.get(action)
        if delta is None:
            continue
        position = (position[0] + delta[0], position[1] + delta[1])
    return position


def selected_termination_predicate(option: OptionCandidate) -> PredicateCandidate | None:
    if option.selected_predicate_id is None:
        return None
    for candidate in option.termination_predicate_candidates:
        if candidate.predicate_id == option.selected_predicate_id:
            return candidate
    return None


def selected_termination_expression(option: OptionCandidate) -> dict[str, Any]:
    selected = selected_termination_predicate(option)
    if selected is not None:
        return selected.expression
    return option.termination_condition


def selected_predicate_expression_hash(option: OptionCandidate) -> str:
    if option.selected_predicate_expression_hash is not None:
        return option.selected_predicate_expression_hash
    return predicate_expression_hash(selected_termination_expression(option))


def predicate_expression_hash(expression: dict[str, Any]) -> str:
    payload = json.dumps(expression, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def termination_target(option: OptionCandidate) -> Position:
    expression = selected_termination_expression(option)
    if "target_position" in expression:
        return _position_from(expression["target_position"])
    if "center" in expression:
        return _position_from(expression["center"])
    return _position_from(option.expected_actor_delta["to_position"])


def evaluate_position_expression(expression: dict[str, Any], position: Position) -> bool:
    kind = expression.get("type")
    if kind in {"actor_reaches_observable_position", "exact_position"}:
        target = _position_from(expression["target_position"])
        tolerance = int(expression.get("tolerance", 0))
        return _distance(position, target) <= tolerance
    if kind == "region_membership":
        center = _position_from(expression["center"])
        radius = int(expression.get("radius", 0))
        return _distance(position, center) <= radius
    return False


def _first_controlled_step(reference: TraceReference) -> int | None:
    for state in reference.states:
        if state.controlled_center is not None:
            return state.step
    return None


def _round_position(center: tuple[float, float]) -> Position:
    return (int(round(center[0])), int(round(center[1])))


def _position_from(value: Any) -> Position:
    return (int(value[0]), int(value[1]))


def _labels(signature: tuple[tuple[int, int, str], ...]) -> list[str]:
    return [
        f"value={value}|area={area}|shape={shape_id}"
        for value, area, shape_id in signature
    ]


def _dedupe_sorted(values: list[int]) -> tuple[int, ...]:
    return tuple(sorted(set(values)))


def _apply(start: Position, deltas: tuple[Delta, ...]) -> Position:
    position = start
    for delta in deltas:
        position = (position[0] + delta[0], position[1] + delta[1])
    return position


def _distance(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _candidate_by_kind(
    candidates: tuple[PredicateCandidate, ...],
    kind: str,
) -> PredicateCandidate | None:
    for candidate in candidates:
        if candidate.kind == kind:
            return candidate
    return None


def _gate_rejection(
    *,
    expression: dict[str, Any],
    evidence: tuple[dict[str, Any], ...],
    validates_original: bool,
    true_at_initiation: bool,
    true_at_terminal: bool,
    terminal_selective: bool,
) -> str | None:
    if not evidence:
        return "missing_evidence_provenance"
    if _is_index_based(expression):
        return "trace_index_based"
    if _is_raw_action_based(expression):
        return "raw_action_based"
    if true_at_initiation:
        return "true_at_initiation"
    if not true_at_terminal:
        return "not_true_at_terminal"
    if not validates_original:
        return "does_not_validate_m4_masked_execution"
    if not terminal_selective:
        return "not_terminal_selective_or_overbroad"
    return None


def _is_index_based(expression: dict[str, Any]) -> bool:
    if expression.get("type") == "trace_index":
        return True
    forbidden_metadata = {"forbidden_form"}
    return any(
        key not in forbidden_metadata
        and ("trace_index" in key or "segment_end" in key)
        for key in expression
    )


def _is_raw_action_based(expression: dict[str, Any]) -> bool:
    encoded = json.dumps(expression, sort_keys=True)
    return any(
        key in encoded
        for key in ("raw_action", "human_action", "next_action", "expected_action", "action_at_index")
    )


def _progress_kind(expression: dict[str, Any]) -> str:
    kind = expression.get("type")
    if kind == "region_membership":
        return "distance_to_region"
    if kind in {"actor_reaches_observable_position", "exact_position"}:
        return "distance_to_position"
    if kind == "semantic_delta_event":
        return "boolean_event"
    return "unknown"
