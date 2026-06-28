"""Structured recovery with trace-resume circuit breakers for 0001."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Literal

from experiments.harness import Observation

from .events import Delta
from .milestone2 import TraceReference
from .observation import ObservationParser


DriftType = Literal[
    "ACTION_CONFLICT",
    "WORLD_DRIFT",
    "ON_TRACE",
    "LOCAL_DRIFT",
    "TRACE_INVALID",
    "UNKNOWN",
]
ResumeValidation = Literal["TRACE_VALID", "TRACE_INVALID", "UNKNOWN"]
Position = tuple[int, int]
BlockedEdge = tuple[Position, int]


@dataclass(frozen=True)
class SemanticSignature:
    """Neutral non-controlled state summary.

    The v1 signature is a multiset of labels/descriptors. The compatibility
    method deliberately supports volatility filtering because the important
    comparison is corridor compatibility, not raw frame equality.
    """

    labels: tuple[str, ...]

    @classmethod
    def from_labels(cls, labels) -> "SemanticSignature":
        flat: list[str] = []
        for label in labels:
            if isinstance(label, (tuple, list)):
                flat.extend(str(item) for item in label)
            else:
                flat.append(str(label))
        return cls(tuple(sorted(flat)))

    @classmethod
    def from_observation(
        cls,
        observation: Observation,
        *,
        controlled_values: frozenset[int] = frozenset({1}),
        volatile_values: frozenset[int] = frozenset(),
    ) -> "SemanticSignature":
        scene = ObservationParser().parse(observation, frame_index=0)
        labels = []
        for detection in scene.detections:
            if detection.value in controlled_values or detection.value in volatile_values:
                continue
            center = (
                int(round(detection.centroid[0])),
                int(round(detection.centroid[1])),
            )
            labels.append(
                f"value={detection.value}|area={detection.area}|shape={detection.shape_id}|center={center}"
            )
        return cls.from_labels(labels)

    def compatible_with(
        self,
        other: "SemanticSignature",
        *,
        volatile_prefixes: tuple[str, ...] = (),
    ) -> bool:
        return (
            Counter(self._stable_labels(volatile_prefixes))
            == Counter(other._stable_labels(volatile_prefixes))
        )

    def differences_from(
        self,
        other: "SemanticSignature",
        *,
        volatile_prefixes: tuple[str, ...] = (),
    ) -> list[dict[str, str]]:
        left = Counter(self._stable_labels(volatile_prefixes))
        right = Counter(other._stable_labels(volatile_prefixes))
        changes = []
        for label in sorted((left - right).elements()):
            changes.append({"type": "unexpected_present", "label": label})
        for label in sorted((right - left).elements()):
            changes.append({"type": "expected_missing", "label": label})
        return changes

    def _stable_labels(self, volatile_prefixes: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            label
            for label in self.labels
            if not any(label.startswith(prefix) for prefix in volatile_prefixes)
        )


@dataclass(frozen=True)
class CorridorState:
    trace_index: int
    controlled_position: Position
    semantic_signature: SemanticSignature


@dataclass(frozen=True)
class TraceCorridor:
    states: tuple[CorridorState, ...]
    start_index: int
    end_index: int

    @classmethod
    def from_reference(
        cls,
        reference: TraceReference,
        *,
        start_index: int,
        length: int = 5,
    ) -> "TraceCorridor":
        start = max(0, start_index)
        end = min(reference.completion_step, start + length)
        states = []
        for index in range(start, end + 1):
            state = reference.state(index)
            if state.controlled_center is None:
                continue
            labels = [
                f"value={value}|area={area}|shape={shape_id}"
                for value, area, shape_id in state.nearby_signature
            ]
            states.append(CorridorState(
                trace_index=index,
                controlled_position=_round_position(state.controlled_center),
                semantic_signature=SemanticSignature.from_labels(labels),
            ))
        return cls(tuple(states), start, end)

    @classmethod
    def synthetic(
        cls,
        states: list[tuple[int, Position, tuple[tuple[str, ...], ...]]],
    ) -> "TraceCorridor":
        corridor_states = tuple(
            CorridorState(
                trace_index=index,
                controlled_position=position,
                semantic_signature=SemanticSignature.from_labels(labels),
            )
            for index, position, labels in states
        )
        return cls(
            corridor_states,
            states[0][0] if states else 0,
            states[-1][0] if states else 0,
        )

    def evaluate(
        self,
        *,
        current_signature: SemanticSignature,
        current_position: Position | None = None,
        volatile_prefixes: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        records = []
        for state in self.states:
            compatible = current_signature.compatible_with(
                state.semantic_signature,
                volatile_prefixes=volatile_prefixes,
            )
            offset = (
                None
                if current_position is None
                else (
                    current_position[0] - state.controlled_position[0],
                    current_position[1] - state.controlled_position[1],
                )
            )
            records.append({
                "trace_index": state.trace_index,
                "semantic_compatible": compatible,
                "actor_offset": list(offset) if offset is not None else None,
                "reason": None if compatible else "semantic_signature_mismatch",
                "semantic_features_changed": current_signature.differences_from(
                    state.semantic_signature,
                    volatile_prefixes=volatile_prefixes,
                ),
            })
        return records

    def state_by_index(self, trace_index: int) -> CorridorState | None:
        for state in self.states:
            if state.trace_index == trace_index:
                return state
        return None


@dataclass(frozen=True)
class DriftResult:
    drift_type: DriftType
    chosen_corridor_index: int | None
    corridor_candidates: tuple[dict[str, Any], ...]
    semantic_features_changed: tuple[dict[str, str], ...]
    resume_validation: ResumeValidation = "UNKNOWN"
    failure_reason: str | None = None


class DriftClassifier:
    def __init__(self, *, volatile_prefixes: tuple[str, ...] = ()) -> None:
        self.volatile_prefixes = volatile_prefixes

    def classify(
        self,
        *,
        current_center: tuple[float, float] | None,
        current_signature: SemanticSignature,
        corridor: TraceCorridor,
        action_result: dict[str, Any] | None,
    ) -> DriftResult:
        if action_result is not None:
            predicted = action_result.get("predicted_delta")
            observed = action_result.get("observed_delta")
            if predicted is not None and observed is not None and tuple(predicted) != tuple(observed):
                return DriftResult(
                    drift_type="ACTION_CONFLICT",
                    chosen_corridor_index=None,
                    corridor_candidates=tuple(),
                    semantic_features_changed=tuple(),
                    failure_reason="ACTION_DELTA_MISMATCH",
                )
            if action_result.get("world_diff"):
                return DriftResult(
                    drift_type="WORLD_DRIFT",
                    chosen_corridor_index=None,
                    corridor_candidates=tuple(),
                    semantic_features_changed=tuple(action_result["world_diff"]),
                    failure_reason="SEMANTIC_INCOMPATIBILITY",
                )

        if current_center is None:
            return DriftResult(
                drift_type="UNKNOWN",
                chosen_corridor_index=None,
                corridor_candidates=tuple(corridor.evaluate(
                    current_signature=current_signature,
                    current_position=None,
                    volatile_prefixes=self.volatile_prefixes,
                )),
                semantic_features_changed=tuple(),
                failure_reason="CONTROLLED_IDENTITY_UNCERTAIN",
            )

        current_position = _round_position(current_center)
        candidates = corridor.evaluate(
            current_signature=current_signature,
            current_position=current_position,
            volatile_prefixes=self.volatile_prefixes,
        )
        compatible_states = [
            (record, corridor.state_by_index(record["trace_index"]))
            for record in candidates
            if record["semantic_compatible"]
        ]
        if not compatible_states:
            changed = tuple(candidates[0]["semantic_features_changed"]) if candidates else tuple()
            return DriftResult(
                drift_type="WORLD_DRIFT",
                chosen_corridor_index=None,
                corridor_candidates=tuple(candidates),
                semantic_features_changed=changed,
                failure_reason="SEMANTIC_INCOMPATIBILITY",
            )

        compatible_states = [
            (record, state)
            for record, state in compatible_states
            if state is not None
        ]
        record, state = min(
            compatible_states,
            key=lambda item: (
                _distance(current_position, item[1].controlled_position),
                item[1].trace_index,
            ),
        )
        if current_position == state.controlled_position:
            drift_type: DriftType = "ON_TRACE"
        else:
            drift_type = "LOCAL_DRIFT"
        return DriftResult(
            drift_type=drift_type,
            chosen_corridor_index=state.trace_index,
            corridor_candidates=tuple(candidates),
            semantic_features_changed=tuple(),
            resume_validation="TRACE_VALID" if drift_type == "ON_TRACE" else "UNKNOWN",
        )


@dataclass(frozen=True)
class RecoveryPlanResult:
    recovered: bool
    plan: tuple[int, ...]
    resume_index: int | None
    corridor_candidates: tuple[dict[str, Any], ...]
    failure_reason: str | None
    blocked_movement_memory: tuple[dict[str, Any], ...]
    repeated_blocked_action_count: int
    non_greedy: bool


class LocalRecoveryPlanner:
    def __init__(self, *, budget: int = 12) -> None:
        self.budget = budget

    def plan(
        self,
        *,
        start: Position,
        corridor: TraceCorridor,
        action_deltas: dict[int, Delta],
        blocked_edges: set[BlockedEdge] | frozenset[BlockedEdge] = frozenset(),
        valid_positions: frozenset[Position] | None = None,
    ) -> RecoveryPlanResult:
        records = []
        candidates: list[tuple[int, int, tuple[int, ...], CorridorState]] = []
        for state in corridor.states:
            route = self._bfs(
                start=start,
                target=state.controlled_position,
                action_deltas=action_deltas,
                blocked_edges=blocked_edges,
                valid_positions=valid_positions,
            )
            reachable = route is not None
            records.append({
                "trace_index": state.trace_index,
                "semantic_compatible": True,
                "reachable": reachable,
                "plan_length": len(route) if route is not None else None,
                "reason": None if reachable else "TARGET_UNREACHABLE",
            })
            if route is not None:
                candidates.append((
                    len(route),
                    abs(state.trace_index - corridor.start_index),
                    route,
                    state,
                ))

        blocked_memory = tuple(
            {
                "position": list(position),
                "action": action,
            }
            for position, action in sorted(blocked_edges)
        )
        if not candidates:
            return RecoveryPlanResult(
                recovered=False,
                plan=(),
                resume_index=None,
                corridor_candidates=tuple(records),
                failure_reason="TARGET_UNREACHABLE",
                blocked_movement_memory=blocked_memory,
                repeated_blocked_action_count=0,
                non_greedy=False,
            )

        _length, _jump, route, state = min(candidates)
        greedy = _greedy_first_action(
            start=start,
            target=state.controlled_position,
            action_deltas=action_deltas,
            blocked_edges=blocked_edges,
            valid_positions=valid_positions,
        )
        return RecoveryPlanResult(
            recovered=True,
            plan=route,
            resume_index=state.trace_index,
            corridor_candidates=tuple(
                record | {"chosen": record["trace_index"] == state.trace_index}
                for record in records
            ),
            failure_reason=None,
            blocked_movement_memory=blocked_memory,
            repeated_blocked_action_count=0,
            non_greedy=bool(route and route[0] != greedy),
        )

    def _bfs(
        self,
        *,
        start: Position,
        target: Position,
        action_deltas: dict[int, Delta],
        blocked_edges: set[BlockedEdge] | frozenset[BlockedEdge],
        valid_positions: frozenset[Position] | None,
    ) -> tuple[int, ...] | None:
        queue = deque([(start, ())])
        seen = {start}
        while queue:
            position, path = queue.popleft()
            if position == target:
                return path
            if len(path) >= self.budget:
                continue
            for action in sorted(action_deltas):
                if (position, action) in blocked_edges:
                    continue
                delta = action_deltas[action]
                next_position = (position[0] + delta[0], position[1] + delta[1])
                if valid_positions is not None and next_position not in valid_positions:
                    continue
                if next_position in seen:
                    continue
                seen.add(next_position)
                queue.append((next_position, path + (action,)))
        return None


@dataclass(frozen=True)
class RecoveryExecutionResult:
    status: DriftType | Literal["RECOVERED"]
    records: tuple[dict[str, Any], ...]
    blocked_edges: frozenset[BlockedEdge]
    failure_reason: str | None = None


class StructuredRecoveryAgent:
    def __init__(self, *, classifier: DriftClassifier | None = None) -> None:
        self.classifier = classifier or DriftClassifier()

    def validate_resume(
        self,
        *,
        current_center: tuple[float, float] | None,
        current_signature: SemanticSignature,
        corridor: TraceCorridor,
        resume_index: int,
    ) -> ResumeValidation:
        if current_center is None:
            return "UNKNOWN"
        state = corridor.state_by_index(resume_index)
        if state is None:
            return "TRACE_INVALID"
        if _round_position(current_center) != state.controlled_position:
            return "TRACE_INVALID"
        if not current_signature.compatible_with(
            state.semantic_signature,
            volatile_prefixes=self.classifier.volatile_prefixes,
        ):
            return "TRACE_INVALID"
        return "TRACE_VALID"

    def execute_recovery_plan(
        self,
        *,
        plan: tuple[int, ...],
        start: Position,
        action_deltas: dict[int, Delta],
        observed_deltas: tuple[Delta, ...],
        world_diffs: tuple[tuple[dict[str, Any], ...], ...],
    ) -> RecoveryExecutionResult:
        position = start
        blocked: set[BlockedEdge] = set()
        records = []
        for index, action in enumerate(plan, start=1):
            predicted = action_deltas[action]
            observed = observed_deltas[index - 1]
            world_diff = list(world_diffs[index - 1])
            blocked_added = False
            if tuple(predicted) != tuple(observed):
                blocked.add((position, action))
                blocked_added = True
                classification = "ACTION_CONFLICT"
                records.append(_execution_record(
                    index,
                    action,
                    predicted,
                    observed,
                    world_diff,
                    classification,
                    blocked_added,
                ))
                return RecoveryExecutionResult(
                    status="ACTION_CONFLICT",
                    records=tuple(records),
                    blocked_edges=frozenset(blocked),
                    failure_reason="ACTION_DELTA_MISMATCH",
                )
            if world_diff:
                classification = "WORLD_DRIFT"
                records.append(_execution_record(
                    index,
                    action,
                    predicted,
                    observed,
                    world_diff,
                    classification,
                    blocked_added,
                ))
                return RecoveryExecutionResult(
                    status="WORLD_DRIFT",
                    records=tuple(records),
                    blocked_edges=frozenset(blocked),
                    failure_reason="SEMANTIC_INCOMPATIBILITY",
                )
            classification = "LOCAL_DRIFT"
            records.append(_execution_record(
                index,
                action,
                predicted,
                observed,
                world_diff,
                classification,
                blocked_added,
            ))
            position = (position[0] + observed[0], position[1] + observed[1])
        return RecoveryExecutionResult(
            status="RECOVERED",
            records=tuple(records),
            blocked_edges=frozenset(blocked),
        )


def _execution_record(
    step: int,
    action: int,
    predicted: Delta,
    observed: Delta,
    world_diff: list[dict[str, Any]],
    classification: str,
    blocked_added: bool,
) -> dict[str, Any]:
    return {
        "step": step,
        "phase": "RECOVERY",
        "action": action,
        "action_source": "LOCAL_RECOVERY_BFS",
        "predicted_delta": list(predicted),
        "observed_delta": list(observed),
        "world_diff": world_diff,
        "classification_after_action": classification,
        "blocked_edge_added": blocked_added,
        "trace_resumed_after_step": False,
    }


def _round_position(center: tuple[float, float]) -> Position:
    return (int(round(center[0])), int(round(center[1])))


def _distance(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _greedy_first_action(
    *,
    start: Position,
    target: Position,
    action_deltas: dict[int, Delta],
    blocked_edges: set[BlockedEdge] | frozenset[BlockedEdge],
    valid_positions: frozenset[Position] | None,
) -> int | None:
    current = _distance(start, target)
    best: tuple[int, int] | None = None
    for action in sorted(action_deltas):
        if (start, action) in blocked_edges:
            continue
        delta = action_deltas[action]
        next_position = (start[0] + delta[0], start[1] + delta[1])
        if valid_positions is not None and next_position not in valid_positions:
            continue
        distance = _distance(next_position, target)
        if distance >= current:
            continue
        candidate = (distance, action)
        if best is None or candidate < best:
            best = candidate
    return best[1] if best else None
