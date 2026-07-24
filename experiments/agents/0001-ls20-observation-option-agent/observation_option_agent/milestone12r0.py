"""Public factor discovery for 0001 M12.R0."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections import Counter
from typing import Any, Literal

from experiments.harness import Observation

from .observation import Detection, ObservationParser
from .tracking import ObjectTracker


Position = tuple[int, int]
M12R0Status = Literal[
    "complete_with_known_limits",
    "partial_factor_extraction_only",
    "partial_no_prediction_gain",
    "failed_no_actor_factor",
    "failed_no_stable_factor_split",
    "failed_no_changed_factor_split",
    "failed_named_object_ontology",
    "failed_oracle_or_private_state_used",
]


@dataclass(frozen=True)
class PublicFactor:
    factor_id: str
    value: int
    area: int
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    shape_id: str
    persistence: str

    def to_record(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "value": self.value,
            "area": self.area,
            "bbox": list(self.bbox),
            "center": list(self.center),
            "shape_id": self.shape_id,
            "persistence": self.persistence,
        }


@dataclass(frozen=True)
class ChangedFactor:
    factor_id: str
    value_before: int
    value_after: int
    area_before: int
    area_after: int
    bbox_before: tuple[int, int, int, int]
    bbox_after: tuple[int, int, int, int]
    change_kind: str

    def to_record(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "value_before": self.value_before,
            "value_after": self.value_after,
            "area_before": self.area_before,
            "area_after": self.area_after,
            "bbox_before": list(self.bbox_before),
            "bbox_after": list(self.bbox_after),
            "change_kind": self.change_kind,
        }


@dataclass(frozen=True)
class ActorFactor:
    factor_id: str
    value: int
    before_center: tuple[float, float]
    after_center: tuple[float, float]
    delta: tuple[int, int]
    area: int
    source: str

    def to_record(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "value": self.value,
            "before_center": list(self.before_center),
            "after_center": list(self.after_center),
            "delta": list(self.delta),
            "area": self.area,
            "source": self.source,
        }


@dataclass(frozen=True)
class PublicFactorTransition:
    transition_id: str
    action: int
    actor_factor: ActorFactor | None
    stable_factors: tuple[PublicFactor, ...]
    changed_factors: tuple[ChangedFactor, ...]
    moved_factors: tuple[ActorFactor, ...]
    status_delta: int
    no_op: bool
    factor_state_key: str
    raw_state_key: str

    def to_record(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "action": self.action,
            "actor_factor": (
                self.actor_factor.to_record()
                if self.actor_factor is not None
                else None
            ),
            "stable_factors": [factor.to_record() for factor in self.stable_factors],
            "changed_factors": [factor.to_record() for factor in self.changed_factors],
            "moved_factors": [factor.to_record() for factor in self.moved_factors],
            "status_delta": self.status_delta,
            "no_op": self.no_op,
            "factor_state_key": self.factor_state_key,
            "raw_state_key": self.raw_state_key,
        }


@dataclass(frozen=True)
class FactorTransitionAudit:
    transition_count: int
    raw_state_count: int
    factor_state_count: int
    raw_context_reuse_count: int
    factor_context_reuse_count: int
    raw_prediction_accuracy: float
    factor_prediction_accuracy: float
    compression_gain: float
    prediction_gain: float
    records: tuple[dict[str, Any], ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "transition_count": self.transition_count,
            "raw_state_count": self.raw_state_count,
            "factor_state_count": self.factor_state_count,
            "raw_context_reuse_count": self.raw_context_reuse_count,
            "factor_context_reuse_count": self.factor_context_reuse_count,
            "raw_prediction_accuracy": self.raw_prediction_accuracy,
            "factor_prediction_accuracy": self.factor_prediction_accuracy,
            "compression_gain": self.compression_gain,
            "prediction_gain": self.prediction_gain,
            "records": list(self.records),
        }


class PublicFactorExtractor:
    def __init__(
        self,
        *,
        parser: ObservationParser | None = None,
        tracker: ObjectTracker | None = None,
    ) -> None:
        self.parser = parser or ObservationParser()
        self.tracker = tracker or ObjectTracker()
        self._started = False

    def extract(
        self,
        *,
        transition_id: str,
        before: Observation,
        action: int,
        after: Observation,
        controlled_position: Position | None = None,
    ) -> PublicFactorTransition:
        before_scene = self.parser.parse(before, frame_index=0)
        after_scene = self.parser.parse(after, frame_index=1)
        if not self._started:
            self.tracker.update(before_scene, force_new=True)
            self._started = True
        tracks = self.tracker.update(after_scene)
        moved = tuple(
            _actor_from_match(match, source="tracked_motion")
            for match in tracks.matches
            if _delta(match.before, match.after) != (0, 0)
        )
        actor = _select_actor(moved=moved, controlled_position=controlled_position)
        stable = tuple(
            _public_factor(match.after, factor_id=f"stable_{match.object_id}")
            for match in tracks.matches
            if _delta(match.before, match.after) == (0, 0)
            and match.before.value == match.after.value
            and match.before.shape_id == match.after.shape_id
        )
        changed = tuple(
            _changed_factor(match, factor_id=f"changed_{match.object_id}")
            for match in tracks.matches
            if _delta(match.before, match.after) == (0, 0)
            and (
                match.before.value != match.after.value
                or match.before.shape_id != match.after.shape_id
                or match.before.area != match.after.area
            )
        )
        status_delta = int(after.levels_completed) - int(before.levels_completed)
        no_op = actor is None and status_delta == 0
        return PublicFactorTransition(
            transition_id=transition_id,
            action=action,
            actor_factor=actor,
            stable_factors=tuple(sorted(stable, key=lambda item: (item.value, item.area))),
            changed_factors=tuple(sorted(
                changed,
                key=lambda item: (item.value_before, item.value_after, item.area_before),
            )),
            moved_factors=moved,
            status_delta=status_delta,
            no_op=no_op,
            factor_state_key=_pre_action_factor_state_key(
                before_scene.detections,
                status=int(before.levels_completed),
            ),
            raw_state_key=observation_hash(before),
        )


class FactorTransitionAssay:
    def evaluate(
        self,
        transitions: tuple[PublicFactorTransition, ...],
    ) -> FactorTransitionAudit:
        raw_counts: dict[tuple[str, int], dict[str, int]] = {}
        factor_counts: dict[tuple[str, int], dict[str, int]] = {}
        records = []
        raw_correct = 0
        factor_correct = 0
        raw_predictions = 0
        factor_predictions = 0
        raw_reuse = 0
        factor_reuse = 0
        for transition in transitions:
            outcome = _outcome_symbol(transition)
            raw_context = (transition.raw_state_key, transition.action)
            factor_context = (transition.factor_state_key, transition.action)
            raw_prediction = _predict(raw_counts.get(raw_context))
            factor_prediction = _predict(factor_counts.get(factor_context))
            if raw_prediction is not None:
                raw_predictions += 1
                raw_reuse += 1
                raw_correct += 1 if raw_prediction == outcome else 0
            if factor_prediction is not None:
                factor_predictions += 1
                factor_reuse += 1
                factor_correct += 1 if factor_prediction == outcome else 0
            records.append({
                "transition_id": transition.transition_id,
                "action": transition.action,
                "outcome": outcome,
                "raw_context_seen": raw_prediction is not None,
                "factor_context_seen": factor_prediction is not None,
                "raw_prediction": raw_prediction,
                "factor_prediction": factor_prediction,
            })
            _increment(raw_counts, raw_context, outcome)
            _increment(factor_counts, factor_context, outcome)
        raw_state_count = len({transition.raw_state_key for transition in transitions})
        factor_state_count = len({transition.factor_state_key for transition in transitions})
        raw_accuracy = round(raw_correct / raw_predictions, 4) if raw_predictions else 0.0
        factor_accuracy = (
            round(factor_correct / factor_predictions, 4) if factor_predictions else 0.0
        )
        compression_gain = (
            round((raw_state_count - factor_state_count) / raw_state_count, 4)
            if raw_state_count
            else 0.0
        )
        return FactorTransitionAudit(
            transition_count=len(transitions),
            raw_state_count=raw_state_count,
            factor_state_count=factor_state_count,
            raw_context_reuse_count=raw_reuse,
            factor_context_reuse_count=factor_reuse,
            raw_prediction_accuracy=raw_accuracy,
            factor_prediction_accuracy=factor_accuracy,
            compression_gain=compression_gain,
            prediction_gain=round(factor_accuracy - raw_accuracy, 4),
            records=tuple(records),
        )


def factor_hypotheses_from_transitions(
    transitions: tuple[PublicFactorTransition, ...],
) -> tuple[dict[str, Any], ...]:
    actor_support = sum(1 for transition in transitions if transition.actor_factor is not None)
    no_op_support = sum(1 for transition in transitions if transition.no_op)
    stable_support = sum(1 for transition in transitions if transition.stable_factors)
    changed_support = sum(1 for transition in transitions if transition.changed_factors)
    hypotheses = [
        {
            "hypothesis_id": "factor_actor_motion_000",
            "kind": "actor_motion_factor",
            "support": actor_support,
            "predictive_use": "action_effect_outcome",
            "status": "ACTIVE" if actor_support else "ABSENT",
        },
        {
            "hypothesis_id": "factor_no_op_000",
            "kind": "no_op_action_effect_factor",
            "support": no_op_support,
            "predictive_use": "blocked_or_ineffective_action_regularities",
            "status": "ACTIVE" if no_op_support else "ABSENT",
        },
        {
            "hypothesis_id": "factor_stability_000",
            "kind": "stable_component_factor",
            "support": stable_support,
            "predictive_use": "pre_action_factor_state_compression",
            "status": "ACTIVE" if stable_support else "ABSENT",
        },
        {
            "hypothesis_id": "factor_change_000",
            "kind": "changed_component_factor",
            "support": changed_support,
            "predictive_use": "non_actor_state_change_detection",
            "status": "ACTIVE" if changed_support else "ABSENT",
        },
    ]
    return tuple(hypotheses)


def m12r0_status(
    *,
    transition_count: int,
    actor_factor_count: int,
    stable_factor_transition_count: int,
    changed_factor_transition_count: int,
    compression_gain: float,
    factor_context_reuse_count: int,
    raw_context_reuse_count: int,
    prediction_gain: float,
    named_object_ontology: bool,
    oracle_or_private_state_used: bool,
) -> M12R0Status:
    if oracle_or_private_state_used:
        return "failed_oracle_or_private_state_used"
    if named_object_ontology:
        return "failed_named_object_ontology"
    if transition_count == 0 or actor_factor_count == 0:
        return "failed_no_actor_factor"
    if stable_factor_transition_count == 0:
        return "failed_no_stable_factor_split"
    if changed_factor_transition_count == 0:
        return "failed_no_changed_factor_split"
    if (
        compression_gain <= 0.0
        or factor_context_reuse_count <= raw_context_reuse_count
        or prediction_gain <= 0.0
    ):
        return "partial_no_prediction_gain"
    return "complete_with_known_limits"


def forbidden_factor_discovery_findings(payload: dict[str, Any]) -> tuple[str, ...]:
    encoded = repr(payload)
    forbidden = (
        "private",
        "oracle",
        "wall",
        "changer",
        "door",
        "trap",
        "clock",
        "goal",
        "human_trace",
        "role",
    )
    return tuple(marker for marker in forbidden if marker in encoded)


def observation_hash(observation: Observation) -> str:
    digest = hashlib.sha256()
    digest.update(observation.pixels.tobytes())
    digest.update(json.dumps({
        "levels_completed": observation.levels_completed,
        "state": observation.state,
        "full_reset": observation.full_reset,
    }, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def factor_transition_outcome_symbol(transition: PublicFactorTransition) -> str:
    return _outcome_symbol(transition)


def _actor_from_match(match, *, source: str) -> ActorFactor:
    delta = _delta(match.before, match.after)
    return ActorFactor(
        factor_id=f"actor_candidate_{match.object_id}",
        value=match.after.value,
        before_center=match.before.centroid,
        after_center=match.after.centroid,
        delta=delta,
        area=match.after.area,
        source=source,
    )


def _select_actor(
    *,
    moved: tuple[ActorFactor, ...],
    controlled_position: Position | None,
) -> ActorFactor | None:
    if not moved:
        return None
    if controlled_position is not None:
        return min(
            moved,
            key=lambda item: (
                abs(item.after_center[0] - controlled_position[0])
                + abs(item.after_center[1] - controlled_position[1]),
                -item.area,
            ),
        )
    return max(moved, key=lambda item: (item.area, item.factor_id))


def _public_factor(detection: Detection, *, factor_id: str) -> PublicFactor:
    return PublicFactor(
        factor_id=factor_id,
        value=detection.value,
        area=detection.area,
        bbox=_bbox_tuple(detection),
        center=detection.centroid,
        shape_id=detection.shape_id,
        persistence="stable",
    )


def _changed_factor(match, *, factor_id: str) -> ChangedFactor:
    return ChangedFactor(
        factor_id=factor_id,
        value_before=match.before.value,
        value_after=match.after.value,
        area_before=match.before.area,
        area_after=match.after.area,
        bbox_before=_bbox_tuple(match.before),
        bbox_after=_bbox_tuple(match.after),
        change_kind=(
            "value_change"
            if match.before.value != match.after.value
            else "shape_or_area_change"
        ),
    )


def _bbox_tuple(detection: Detection) -> tuple[int, int, int, int]:
    return (
        detection.bbox.top,
        detection.bbox.left,
        detection.bbox.bottom,
        detection.bbox.right,
    )


def _delta(before: Detection, after: Detection) -> tuple[int, int]:
    return (
        int(round(after.centroid[0] - before.centroid[0])),
        int(round(after.centroid[1] - before.centroid[1])),
    )


def _pre_action_factor_state_key(
    detections: tuple[Detection, ...],
    *,
    status: int,
) -> str:
    histogram = Counter(
        (detection.value, _area_bucket(detection.area))
        for detection in detections
    )
    payload = {
        "status": status,
        "component_histogram": sorted(
            (value, bucket, count)
            for (value, bucket), count in histogram.items()
        ),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _area_bucket(area: int) -> str:
    if area <= 1:
        return "single"
    if area <= 4:
        return "small"
    if area <= 16:
        return "medium"
    return "large"


def _outcome_symbol(transition: PublicFactorTransition) -> str:
    if transition.status_delta:
        return f"status_delta:{transition.status_delta}"
    if transition.actor_factor is not None:
        return f"actor_delta:{transition.actor_factor.delta[0]},{transition.actor_factor.delta[1]}"
    if transition.no_op:
        return "no_op"
    if transition.changed_factors:
        return "non_actor_change"
    return "unknown"


def _predict(counts: dict[str, int] | None) -> str | None:
    if not counts:
        return None
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _increment(
    counts_by_context: dict[tuple[str, int], dict[str, int]],
    context: tuple[str, int],
    outcome: str,
) -> None:
    counts_by_context.setdefault(context, {})
    counts_by_context[context][outcome] = counts_by_context[context].get(outcome, 0) + 1
