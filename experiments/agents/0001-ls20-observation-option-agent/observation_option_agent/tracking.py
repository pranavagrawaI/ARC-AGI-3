"""Temporal identity over neutral visual detections."""

from __future__ import annotations

from dataclasses import dataclass

from .observation import BoundingBox, Detection, VisualScene


@dataclass(frozen=True)
class TrackedDetection:
    object_id: int
    detection: Detection


@dataclass(frozen=True)
class ObjectMatch:
    object_id: int
    before: Detection
    after: Detection
    missing_frames: int = 0


@dataclass(frozen=True)
class TrackUpdate:
    current: tuple[TrackedDetection, ...]
    matches: tuple[ObjectMatch, ...]
    created: tuple[TrackedDetection, ...]
    disappeared: tuple[TrackedDetection, ...]
    hidden: tuple[TrackedDetection, ...]
    reappeared: tuple[TrackedDetection, ...]


@dataclass(frozen=True)
class _TrackState:
    detection: Detection
    missing_frames: int


def _centroid_distance(left: Detection, right: Detection) -> float:
    return abs(left.centroid[0] - right.centroid[0]) + abs(left.centroid[1] - right.centroid[1])


def _bbox_intersection(left: BoundingBox, right: BoundingBox) -> int:
    height = max(0, min(left.bottom, right.bottom) - max(left.top, right.top) + 1)
    width = max(0, min(left.right, right.right) - max(left.left, right.left) + 1)
    return height * width


def _candidate_cost(before: Detection, after: Detection) -> tuple[int, float] | None:
    distance = _centroid_distance(before, after)
    if before.value == after.value and before.shape == after.shape:
        return (0, distance)

    shared_pixels = len(before.pixels & after.pixels)
    if shared_pixels:
        union = len(before.pixels | after.pixels)
        area_ratio = abs(before.area - after.area) / max(before.area, after.area)
        return (1, (1.0 - shared_pixels / union) + area_ratio + distance / 1000.0)

    bbox_overlap = _bbox_intersection(before.bbox, after.bbox)
    if bbox_overlap:
        bbox_area = max(
            before.bbox.height * before.bbox.width,
            after.bbox.height * after.bbox.width,
        )
        return (2, (1.0 - bbox_overlap / bbox_area) + distance / 1000.0)
    return None


class ObjectTracker:
    def __init__(self, *, max_missing_frames: int = 1) -> None:
        if max_missing_frames < 0:
            raise ValueError("max_missing_frames must be non-negative")
        self.max_missing_frames = max_missing_frames
        self._next_id = 1
        self._active: dict[int, _TrackState] = {}

    def update(self, scene: VisualScene, *, force_new: bool = False) -> TrackUpdate:
        if force_new:
            disappeared = tuple(
                TrackedDetection(object_id, state.detection)
                for object_id, state in sorted(self._active.items())
            )
            self._active = {}
            created = tuple(self._create(detection) for detection in scene.detections)
            self._active = {
                item.object_id: _TrackState(item.detection, 0)
                for item in created
            }
            return TrackUpdate(created, (), created, disappeared, (), ())

        candidates: list[tuple[int, float, int, int]] = []
        for object_id, state in self._active.items():
            for index, after in enumerate(scene.detections):
                cost = _candidate_cost(state.detection, after)
                if cost is not None:
                    candidates.append((cost[0], cost[1], object_id, index))
        candidates.sort()

        claimed_ids: set[int] = set()
        claimed_detections: set[int] = set()
        matches: list[ObjectMatch] = []
        reappeared: list[TrackedDetection] = []
        next_active: dict[int, _TrackState] = {}
        for _tier, _cost, object_id, index in candidates:
            if object_id in claimed_ids or index in claimed_detections:
                continue
            prior = self._active[object_id]
            before = prior.detection
            after = scene.detections[index]
            claimed_ids.add(object_id)
            claimed_detections.add(index)
            next_active[object_id] = _TrackState(after, 0)
            matches.append(ObjectMatch(
                object_id,
                before,
                after,
                prior.missing_frames,
            ))
            if prior.missing_frames:
                reappeared.append(TrackedDetection(object_id, after))

        disappeared = []
        hidden = []
        for object_id, state in sorted(self._active.items()):
            if object_id in claimed_ids:
                continue
            missing_frames = state.missing_frames + 1
            tracked = TrackedDetection(object_id, state.detection)
            if missing_frames > self.max_missing_frames:
                disappeared.append(tracked)
                continue
            next_active[object_id] = _TrackState(state.detection, missing_frames)
            hidden.append(tracked)
        created_items: list[TrackedDetection] = []
        for index, detection in enumerate(scene.detections):
            if index in claimed_detections:
                continue
            tracked = self._create(detection)
            next_active[tracked.object_id] = _TrackState(detection, 0)
            created_items.append(tracked)

        self._active = next_active
        current = tuple(
            TrackedDetection(object_id, state.detection)
            for object_id, state in sorted(next_active.items())
            if state.missing_frames == 0
        )
        return TrackUpdate(
            current=current,
            matches=tuple(sorted(matches, key=lambda match: match.object_id)),
            created=tuple(created_items),
            disappeared=tuple(disappeared),
            hidden=tuple(hidden),
            reappeared=tuple(reappeared),
        )

    def _create(self, detection: Detection) -> TrackedDetection:
        tracked = TrackedDetection(self._next_id, detection)
        self._next_id += 1
        return tracked
