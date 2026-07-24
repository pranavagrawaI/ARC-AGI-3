"""Actor-centric local-geometry repair for 0001 M12.R2a."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal

import numpy as np

from experiments.harness import Observation

from .milestone12r0 import (
    PublicFactorTransition,
    factor_transition_outcome_symbol,
    forbidden_factor_discovery_findings,
    observation_hash,
)
from .milestone12r1 import (
    FactorContextEvidence,
    FactorCounterexampleAssay,
    FactorCounterexampleAudit,
)
from .observation import Detection, ObservationParser


M12R2AStatus = Literal[
    "complete_with_known_limits",
    "partial_local_geometry_insufficient",
    "partial_refinement_fragmented",
    "partial_no_alias_resolution",
    "failed_coordinate_control_preferred",
    "failed_named_object_ontology",
    "failed_oracle_or_private_state_used",
]


@dataclass(frozen=True)
class ActorLocalGeometryFactors:
    actor_position: tuple[int, int] | None
    actor_bbox: tuple[int, int, int, int] | None
    front_relation: str
    front_signature: tuple[tuple[str, int], ...]
    boundary_distance_bucket: str
    adjacent_non_background_dirs: tuple[int, ...]
    stencil_radius_1: tuple[tuple[str, int], ...]
    stencil_radius_2: tuple[tuple[str, int], ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "actor_position": list(self.actor_position) if self.actor_position else None,
            "actor_bbox": list(self.actor_bbox) if self.actor_bbox else None,
            "front_relation": self.front_relation,
            "front_signature": [
                {"kind": kind, "count": count}
                for kind, count in self.front_signature
            ],
            "boundary_distance_bucket": self.boundary_distance_bucket,
            "adjacent_non_background_dirs": list(self.adjacent_non_background_dirs),
            "stencil_radius_1": [
                {"kind": kind, "count": count}
                for kind, count in self.stencil_radius_1
            ],
            "stencil_radius_2": [
                {"kind": kind, "count": count}
                for kind, count in self.stencil_radius_2
            ],
        }


@dataclass(frozen=True)
class ActorLocalGeometryRecord:
    transition_id: str
    action: int
    base_factor_state_key: str
    local_geometry_key: str
    coordinate_control_key: str
    raw_state_key: str
    outcome: str
    pre_action_factors: ActorLocalGeometryFactors

    def to_record(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "action": self.action,
            "base_factor_state_key": self.base_factor_state_key,
            "local_geometry_key": self.local_geometry_key,
            "coordinate_control_key": self.coordinate_control_key,
            "raw_state_key": self.raw_state_key,
            "outcome": self.outcome,
            "pre_action_factors": self.pre_action_factors.to_record(),
        }


@dataclass(frozen=True)
class RepairContextEvidence:
    context_id: str
    key: str
    action: int
    support: int
    outcomes: tuple[tuple[str, int], ...]
    transition_outcomes: tuple[tuple[str, str], ...]
    status: Literal["stable", "contradicted", "unsupported"]

    def to_record(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "key": self.key,
            "action": self.action,
            "support": self.support,
            "outcomes": [
                {"outcome": outcome, "count": count}
                for outcome, count in self.outcomes
            ],
            "transition_outcomes": [
                {"transition_id": transition_id, "outcome": outcome}
                for transition_id, outcome in self.transition_outcomes
            ],
            "status": self.status,
        }


@dataclass(frozen=True)
class RepairKeyAudit:
    key_name: str
    state_count: int
    context_count: int
    reused_context_count: int
    stable_context_count: int
    contradicted_context_count: int
    unsupported_context_count: int
    prediction_accuracy: float
    contexts: tuple[RepairContextEvidence, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "key_name": self.key_name,
            "state_count": self.state_count,
            "context_count": self.context_count,
            "reused_context_count": self.reused_context_count,
            "stable_context_count": self.stable_context_count,
            "contradicted_context_count": self.contradicted_context_count,
            "unsupported_context_count": self.unsupported_context_count,
            "prediction_accuracy": self.prediction_accuracy,
            "contexts": [context.to_record() for context in self.contexts],
        }


@dataclass(frozen=True)
class AliasResolutionEvidence:
    r1_counterexample: str
    action: int
    r1_outcomes: tuple[str, ...]
    r2_split: tuple[dict[str, Any], ...]
    resolution: str
    diagnosis: str

    def to_record(self) -> dict[str, Any]:
        return {
            "r1_counterexample": self.r1_counterexample,
            "action": self.action,
            "r1_outcomes": list(self.r1_outcomes),
            "r2_split": list(self.r2_split),
            "resolution": self.resolution,
            "diagnosis": self.diagnosis,
        }


@dataclass(frozen=True)
class ActorLocalGeometryRepairAudit:
    transition_count: int
    raw_state_count: int
    base: FactorCounterexampleAudit
    local_geometry: RepairKeyAudit
    coordinate_control: RepairKeyAudit
    alias_resolutions: tuple[AliasResolutionEvidence, ...]
    resolved_alias_count: int
    terminal_alias_split_count: int

    def to_record(self) -> dict[str, Any]:
        return {
            "transition_count": self.transition_count,
            "raw_state_count": self.raw_state_count,
            "base": self.base.to_record(),
            "local_geometry": self.local_geometry.to_record(),
            "coordinate_control": self.coordinate_control.to_record(),
            "alias_resolutions": [
                resolution.to_record() for resolution in self.alias_resolutions
            ],
            "resolved_alias_count": self.resolved_alias_count,
            "terminal_alias_split_count": self.terminal_alias_split_count,
        }


class ActorLocalGeometryRefiner:
    def __init__(
        self,
        *,
        parser: ObservationParser | None = None,
        coordinate_bucket_size: int = 5,
    ) -> None:
        self.parser = parser or ObservationParser()
        self.coordinate_bucket_size = coordinate_bucket_size

    def refine(
        self,
        *,
        transition: PublicFactorTransition,
        before: Observation,
        actor_position: tuple[int, int] | None = None,
    ) -> ActorLocalGeometryRecord:
        grid = np.asarray(before.pixels[-1])
        scene = self.parser.parse(before, frame_index=0)
        background = int(scene.inferred_background)
        position = actor_position or _position_from_transition(transition)
        detection = _actor_detection(scene.detections, position)
        factors = _local_geometry_factors(
            grid=grid,
            background=background,
            action=transition.action,
            actor_position=position,
            detection=detection,
        )
        local_payload = {
            "base": transition.factor_state_key,
            "action_front_geometry": {
                "front_relation": factors.front_relation,
                "front_signature": factors.front_signature,
                "boundary_distance_bucket": factors.boundary_distance_bucket,
                "adjacent_non_background_dirs": factors.adjacent_non_background_dirs,
            },
        }
        coordinate_payload = {
            "base": transition.factor_state_key,
            "actor_position_bucket": _coordinate_bucket(
                position,
                bucket_size=self.coordinate_bucket_size,
            ),
        }
        return ActorLocalGeometryRecord(
            transition_id=transition.transition_id,
            action=transition.action,
            base_factor_state_key=transition.factor_state_key,
            local_geometry_key=_stable_digest(local_payload),
            coordinate_control_key=_stable_digest(coordinate_payload),
            raw_state_key=observation_hash(before),
            outcome=factor_transition_outcome_symbol(transition),
            pre_action_factors=factors,
        )


class ActorLocalGeometryRepairAssay:
    def __init__(self, *, minimum_support: int = 2) -> None:
        self.minimum_support = minimum_support

    def evaluate(
        self,
        records: tuple[ActorLocalGeometryRecord, ...],
        *,
        base_transitions: tuple[PublicFactorTransition, ...] | None = None,
    ) -> ActorLocalGeometryRepairAudit:
        if base_transitions is not None:
            base = FactorCounterexampleAssay(
                minimum_support=self.minimum_support,
            ).evaluate(base_transitions)
        else:
            base = _base_audit_from_records(records, minimum_support=self.minimum_support)
        local = _repair_key_audit(
            records,
            key_name="local_geometry",
            key_getter=lambda record: record.local_geometry_key,
            minimum_support=self.minimum_support,
        )
        coordinate = _repair_key_audit(
            records,
            key_name="coordinate_control",
            key_getter=lambda record: record.coordinate_control_key,
            minimum_support=self.minimum_support,
        )
        alias_resolutions = _alias_resolutions(
            records=records,
            base=base,
            minimum_support=self.minimum_support,
        )
        return ActorLocalGeometryRepairAudit(
            transition_count=len(records),
            raw_state_count=len({record.raw_state_key for record in records}),
            base=base,
            local_geometry=local,
            coordinate_control=coordinate,
            alias_resolutions=tuple(alias_resolutions),
            resolved_alias_count=sum(
                1 for item in alias_resolutions
                if item.resolution == "resolved_by_local_geometry"
            ),
            terminal_alias_split_count=sum(
                1 for item in alias_resolutions
                if any(outcome.startswith("status_delta") for outcome in item.r1_outcomes)
                and len(item.r2_split) > 1
            ),
        )


def m12r2a_status(
    *,
    raw_state_count: int,
    local_state_count: int,
    local_reused_context_count: int,
    local_stable_context_count: int,
    local_contradicted_context_count: int,
    local_prediction_accuracy: float,
    resolved_alias_count: int,
    coordinate_reused_context_count: int,
    coordinate_contradicted_context_count: int,
    named_object_ontology: bool,
    oracle_or_private_state_used: bool,
) -> M12R2AStatus:
    if oracle_or_private_state_used:
        return "failed_oracle_or_private_state_used"
    if named_object_ontology:
        return "failed_named_object_ontology"
    if (
        local_state_count >= raw_state_count
        or (
            raw_state_count > 0
            and local_state_count / raw_state_count > 0.8
            and local_reused_context_count < 3
        )
    ):
        return "partial_refinement_fragmented"
    if resolved_alias_count == 0:
        return "partial_no_alias_resolution"
    if (
        coordinate_contradicted_context_count < local_contradicted_context_count
        and coordinate_reused_context_count >= local_reused_context_count
    ):
        return "failed_coordinate_control_preferred"
    if (
        local_reused_context_count < 3
        or local_stable_context_count < 2
        or local_contradicted_context_count > 2
        or local_prediction_accuracy <= 0.6
    ):
        return "partial_local_geometry_insufficient"
    return "complete_with_known_limits"


def forbidden_local_geometry_findings(payload: dict[str, Any]) -> tuple[str, ...]:
    return forbidden_factor_discovery_findings(payload)


def _base_audit_from_records(
    records: tuple[ActorLocalGeometryRecord, ...],
    *,
    minimum_support: int,
) -> FactorCounterexampleAudit:
    grouped: dict[tuple[str, int], list[tuple[str, str]]] = {}
    for record in records:
        grouped.setdefault(
            (record.base_factor_state_key, record.action),
            [],
        ).append((record.transition_id, record.outcome))

    contexts = []
    counterexample_count = 0
    for index, ((factor_state_key, action), transition_outcomes) in enumerate(
        sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))
    ):
        outcomes = _outcome_counts(transition_outcomes)
        majority_count = max(outcomes.values()) if outcomes else 0
        if len(transition_outcomes) < minimum_support:
            status: Literal["stable", "contradicted", "unsupported"] = "unsupported"
        elif len(outcomes) == 1:
            status = "stable"
        else:
            status = "contradicted"
            counterexample_count += len(transition_outcomes) - majority_count
        contexts.append(FactorContextEvidence(
            context_id=f"base_context_{index:03d}",
            factor_state_key=factor_state_key,
            action=action,
            support=len(transition_outcomes),
            outcomes=tuple(sorted(
                outcomes.items(),
                key=lambda item: (-item[1], item[0]),
            )),
            transition_outcomes=tuple(transition_outcomes),
            majority_outcome=max(outcomes.items(), key=lambda item: (item[1], item[0]))[0],
            status=status,
            promoted=status == "stable",
            counterexample_count=(
                len(transition_outcomes) - majority_count
                if status == "contradicted"
                else 0
            ),
        ))

    return FactorCounterexampleAudit(
        transition_count=len(records),
        context_count=len(contexts),
        reused_context_count=sum(
            1 for context in contexts if context.support >= minimum_support
        ),
        stable_context_count=sum(
            1 for context in contexts if context.status == "stable"
        ),
        contradicted_context_count=sum(
            1 for context in contexts if context.status == "contradicted"
        ),
        unsupported_context_count=sum(
            1 for context in contexts if context.status == "unsupported"
        ),
        promoted_context_count=sum(1 for context in contexts if context.promoted),
        blocked_promotion_count=sum(
            1 for context in contexts if context.status == "contradicted"
        ),
        counterexample_count=counterexample_count,
        no_op_contradiction_count=sum(
            1
            for context in contexts
            if context.status == "contradicted"
            and any(outcome == "no_op" for outcome, _ in context.outcomes)
        ),
        contexts=tuple(contexts),
    )


def _repair_key_audit(
    records: tuple[ActorLocalGeometryRecord, ...],
    *,
    key_name: str,
    key_getter,
    minimum_support: int,
) -> RepairKeyAudit:
    grouped: dict[tuple[str, int], list[tuple[str, str]]] = {}
    counts: dict[tuple[str, int], dict[str, int]] = {}
    correct = 0
    predictions = 0
    for record in records:
        key = key_getter(record)
        context = (key, record.action)
        prediction = _predict(counts.get(context))
        if prediction is not None:
            predictions += 1
            correct += 1 if prediction == record.outcome else 0
        counts.setdefault(context, {})
        counts[context][record.outcome] = counts[context].get(record.outcome, 0) + 1
        grouped.setdefault(context, []).append((record.transition_id, record.outcome))

    contexts = []
    for index, ((key, action), transition_outcomes) in enumerate(
        sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))
    ):
        outcome_counts = _outcome_counts(transition_outcomes)
        if len(transition_outcomes) < minimum_support:
            status: Literal["stable", "contradicted", "unsupported"] = "unsupported"
        elif len(outcome_counts) == 1:
            status = "stable"
        else:
            status = "contradicted"
        contexts.append(RepairContextEvidence(
            context_id=f"{key_name}_context_{index:03d}",
            key=key,
            action=action,
            support=len(transition_outcomes),
            outcomes=tuple(sorted(
                outcome_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )),
            transition_outcomes=tuple(transition_outcomes),
            status=status,
        ))

    return RepairKeyAudit(
        key_name=key_name,
        state_count=len({key_getter(record) for record in records}),
        context_count=len(contexts),
        reused_context_count=sum(
            1 for context in contexts if context.support >= minimum_support
        ),
        stable_context_count=sum(
            1 for context in contexts if context.status == "stable"
        ),
        contradicted_context_count=sum(
            1 for context in contexts if context.status == "contradicted"
        ),
        unsupported_context_count=sum(
            1 for context in contexts if context.status == "unsupported"
        ),
        prediction_accuracy=round(correct / predictions, 4) if predictions else 0.0,
        contexts=tuple(contexts),
    )


def _alias_resolutions(
    *,
    records: tuple[ActorLocalGeometryRecord, ...],
    base: FactorCounterexampleAudit,
    minimum_support: int,
) -> list[AliasResolutionEvidence]:
    records_by_base: dict[tuple[str, int], list[ActorLocalGeometryRecord]] = {}
    for record in records:
        records_by_base.setdefault(
            (record.base_factor_state_key, record.action),
            [],
        ).append(record)

    resolutions = []
    for context in base.contexts:
        if context.status != "contradicted":
            continue
        base_records = records_by_base.get((context.factor_state_key, context.action), [])
        split = []
        for refined_key, refined_records in _group_records_by_local_key(base_records).items():
            outcomes = _outcome_counts([
                (record.transition_id, record.outcome)
                for record in refined_records
            ])
            split.append({
                "refined_key": refined_key,
                "support": len(refined_records),
                "pre_action_factors": refined_records[0].pre_action_factors.to_record(),
                "outcomes": [
                    {"outcome": outcome, "count": count}
                    for outcome, count in sorted(
                        outcomes.items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                ],
            })
        split_stable = (
            len(split) > 1
            and all(len(item["outcomes"]) == 1 for item in split)
            and all(item["support"] >= minimum_support for item in split)
        )
        if split_stable:
            resolution = "resolved_by_local_geometry"
            diagnosis = "pre_action_local_geometry_splits_all_observed_outcomes"
        elif len(split) > 1:
            resolution = "split_but_unresolved"
            diagnosis = "local_geometry_splits_context_but_some_split_remains_unsupported_or_contradicted"
        else:
            resolution = "unresolved"
            diagnosis = "local_geometry_did_not_split_this_alias"
        resolutions.append(AliasResolutionEvidence(
            r1_counterexample=f"{context.factor_state_key} + action {context.action}",
            action=context.action,
            r1_outcomes=tuple(
                outcome
                for transition_id, outcome in context.transition_outcomes
                for _ in (transition_id,)
            ),
            r2_split=tuple(sorted(split, key=lambda item: item["refined_key"])),
            resolution=resolution,
            diagnosis=diagnosis,
        ))
    return resolutions


def _group_records_by_local_key(
    records: list[ActorLocalGeometryRecord],
) -> dict[str, list[ActorLocalGeometryRecord]]:
    grouped: dict[str, list[ActorLocalGeometryRecord]] = {}
    for record in records:
        grouped.setdefault(record.local_geometry_key, []).append(record)
    return grouped


def _position_from_transition(
    transition: PublicFactorTransition,
) -> tuple[int, int] | None:
    if transition.actor_factor is None:
        return None
    return (
        int(round(transition.actor_factor.before_center[0])),
        int(round(transition.actor_factor.before_center[1])),
    )


def _actor_detection(
    detections: tuple[Detection, ...],
    position: tuple[int, int] | None,
) -> Detection | None:
    if position is None or not detections:
        return None
    for detection in detections:
        if position in detection.pixels:
            return detection
    return min(
        detections,
        key=lambda detection: (
            abs(detection.centroid[0] - position[0])
            + abs(detection.centroid[1] - position[1]),
            -detection.area,
        ),
    )


def _local_geometry_factors(
    *,
    grid: np.ndarray,
    background: int,
    action: int,
    actor_position: tuple[int, int] | None,
    detection: Detection | None,
) -> ActorLocalGeometryFactors:
    if actor_position is None or detection is None:
        return ActorLocalGeometryFactors(
            actor_position=actor_position,
            actor_bbox=None,
            front_relation="unknown_actor",
            front_signature=(),
            boundary_distance_bucket="unknown",
            adjacent_non_background_dirs=(),
            stencil_radius_1=(),
            stencil_radius_2=(),
        )
    delta = _action_delta(action)
    bbox = (
        detection.bbox.top,
        detection.bbox.left,
        detection.bbox.bottom,
        detection.bbox.right,
    )
    front_cells = _front_cells(grid.shape, bbox, delta)
    front_signature = _cell_kind_counts(
        grid=grid,
        background=background,
        cells=front_cells,
        actor_value=detection.value,
    )
    return ActorLocalGeometryFactors(
        actor_position=actor_position,
        actor_bbox=bbox,
        front_relation=_front_relation(front_signature),
        front_signature=front_signature,
        boundary_distance_bucket=_boundary_distance_bucket(grid.shape, bbox, delta),
        adjacent_non_background_dirs=_adjacent_non_background_dirs(
            grid=grid,
            background=background,
            bbox=bbox,
        ),
        stencil_radius_1=_stencil_signature(
            grid=grid,
            background=background,
            actor_position=actor_position,
            actor_value=detection.value,
            radius=1,
        ),
        stencil_radius_2=_stencil_signature(
            grid=grid,
            background=background,
            actor_position=actor_position,
            actor_value=detection.value,
            radius=2,
        ),
    )


def _action_delta(action: int) -> tuple[int, int]:
    return {
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }.get(action, (0, 0))


def _front_cells(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    delta: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    top, left, bottom, right = bbox
    if delta == (-1, 0):
        return tuple((top - 1, col) for col in range(left, right + 1))
    if delta == (1, 0):
        return tuple((bottom + 1, col) for col in range(left, right + 1))
    if delta == (0, -1):
        return tuple((row, left - 1) for row in range(top, bottom + 1))
    if delta == (0, 1):
        return tuple((row, right + 1) for row in range(top, bottom + 1))
    return ()


def _cell_kind_counts(
    *,
    grid: np.ndarray,
    background: int,
    cells: tuple[tuple[int, int], ...],
    actor_value: int,
) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    height, width = grid.shape
    for row, col in cells:
        if not (0 <= row < height and 0 <= col < width):
            kind = "out_of_bounds"
        else:
            value = int(grid[row, col])
            if value == background:
                kind = "background"
            elif value == actor_value:
                kind = "same_value_non_background"
            else:
                kind = "other_non_background"
        counts[kind] = counts.get(kind, 0) + 1
    return tuple(sorted(counts.items()))


def _front_relation(signature: tuple[tuple[str, int], ...]) -> str:
    kinds = {kind for kind, _ in signature}
    if not kinds:
        return "unknown_frontier"
    if kinds == {"background"}:
        return "open"
    if kinds == {"out_of_bounds"}:
        return "boundary"
    if "background" in kinds:
        return "mixed"
    return "blocked_candidate"


def _boundary_distance_bucket(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    delta: tuple[int, int],
) -> str:
    height, width = shape
    top, left, bottom, right = bbox
    if delta == (-1, 0):
        distance = top
    elif delta == (1, 0):
        distance = height - bottom - 1
    elif delta == (0, -1):
        distance = left
    elif delta == (0, 1):
        distance = width - right - 1
    else:
        return "unknown"
    if distance <= 0:
        return "at_boundary"
    if distance == 1:
        return "near_boundary"
    if distance <= 3:
        return "mid_boundary_distance"
    return "far_boundary"


def _adjacent_non_background_dirs(
    *,
    grid: np.ndarray,
    background: int,
    bbox: tuple[int, int, int, int],
) -> tuple[int, ...]:
    dirs = []
    for action, delta in ((1, (-1, 0)), (2, (1, 0)), (3, (0, -1)), (4, (0, 1))):
        signature = _cell_kind_counts(
            grid=grid,
            background=background,
            cells=_front_cells(grid.shape, bbox, delta),
            actor_value=background,
        )
        if any(kind not in {"background", "out_of_bounds"} for kind, _ in signature):
            dirs.append(action)
    return tuple(dirs)


def _stencil_signature(
    *,
    grid: np.ndarray,
    background: int,
    actor_position: tuple[int, int],
    actor_value: int,
    radius: int,
) -> tuple[tuple[str, int], ...]:
    row, col = actor_position
    cells = tuple(
        (next_row, next_col)
        for next_row in range(row - radius, row + radius + 1)
        for next_col in range(col - radius, col + radius + 1)
    )
    return _cell_kind_counts(
        grid=grid,
        background=background,
        cells=cells,
        actor_value=actor_value,
    )


def _coordinate_bucket(
    position: tuple[int, int] | None,
    *,
    bucket_size: int,
) -> tuple[int, int] | None:
    if position is None:
        return None
    size = max(1, bucket_size)
    return (position[0] // size, position[1] // size)


def _stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _outcome_counts(records: list[tuple[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, outcome in records:
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def _predict(counts: dict[str, int] | None) -> str | None:
    if not counts:
        return None
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]
