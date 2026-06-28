"""Observation-derived transition events and action-coupling evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from experiments.harness import Observation

from .tracking import TrackUpdate


Delta = tuple[int, int]


@dataclass(frozen=True)
class ObjectDelta:
    object_id: int
    translation: Delta
    value_before: int
    value_after: int
    shape_changed: bool
    area_before: int
    area_after: int
    bbox_before: tuple[int, int, int, int]
    bbox_after: tuple[int, int, int, int]
    shape_transform: str = "same"

    @property
    def moved(self) -> bool:
        return self.translation != (0, 0)

    @property
    def appearance_changed(self) -> bool:
        return self.value_changed or self.shape_changed or self.area_delta != 0

    @property
    def value_changed(self) -> bool:
        return self.value_before != self.value_after

    @property
    def area_delta(self) -> int:
        return self.area_after - self.area_before


@dataclass(frozen=True)
class MotionGroup:
    translation: Delta
    object_ids: tuple[int, ...]


@dataclass(frozen=True)
class Event:
    step: int
    action: int
    levels_before: int
    levels_after: int
    state_after: str
    full_reset: bool
    changed_pixels: int
    scene_change_fraction: float
    object_deltas: tuple[ObjectDelta, ...]
    created_object_ids: tuple[int, ...]
    disappeared_object_ids: tuple[int, ...]
    motion_groups: tuple[MotionGroup, ...]
    reappeared_object_ids: tuple[int, ...] = ()

    @property
    def level_delta(self) -> int:
        return self.levels_after - self.levels_before


class EventDetector:
    def detect(
        self,
        *,
        step: int,
        before: Observation,
        action: int,
        after: Observation,
        tracks: TrackUpdate,
    ) -> Event:
        before_grid = np.asarray(before.pixels[-1])
        after_grid = np.asarray(after.pixels[-1])
        changed_pixels = int(np.count_nonzero(before_grid != after_grid))
        deltas: list[ObjectDelta] = []
        motion_members: dict[Delta, list[int]] = defaultdict(list)
        for match in tracks.matches:
            if match.missing_frames:
                continue
            translation = (
                int(round(match.after.centroid[0] - match.before.centroid[0])),
                int(round(match.after.centroid[1] - match.before.centroid[1])),
            )
            delta = ObjectDelta(
                object_id=match.object_id,
                translation=translation,
                value_before=match.before.value,
                value_after=match.after.value,
                shape_changed=match.before.shape != match.after.shape,
                area_before=match.before.area,
                area_after=match.after.area,
                bbox_before=(
                    match.before.bbox.top,
                    match.before.bbox.left,
                    match.before.bbox.bottom,
                    match.before.bbox.right,
                ),
                bbox_after=(
                    match.after.bbox.top,
                    match.after.bbox.left,
                    match.after.bbox.bottom,
                    match.after.bbox.right,
                ),
                shape_transform=_shape_transform(match.before.shape, match.after.shape),
            )
            if delta.moved or delta.appearance_changed:
                deltas.append(delta)
            if delta.moved:
                motion_members[translation].append(match.object_id)
        motion_groups = tuple(
            MotionGroup(translation, tuple(sorted(object_ids)))
            for translation, object_ids in sorted(motion_members.items())
        )
        return Event(
            step=step,
            action=action,
            levels_before=before.levels_completed,
            levels_after=after.levels_completed,
            state_after=after.state,
            full_reset=after.full_reset,
            changed_pixels=changed_pixels,
            scene_change_fraction=changed_pixels / before_grid.size,
            object_deltas=tuple(deltas),
            created_object_ids=tuple(item.object_id for item in tracks.created),
            disappeared_object_ids=tuple(item.object_id for item in tracks.disappeared),
            motion_groups=motion_groups,
            reappeared_object_ids=tuple(
                item.object_id for item in tracks.reappeared
            ),
        )


def _normalize_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    if not shape:
        return ()
    top = min(row for row, _col in shape)
    left = min(col for _row, col in shape)
    return tuple(sorted((row - top, col - left) for row, col in shape))


def _rotate_clockwise(
    shape: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    if not shape:
        return ()
    height = max(row for row, _col in shape) + 1
    return _normalize_shape(tuple((col, height - 1 - row) for row, col in shape))


def _reflect_horizontal(
    shape: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    if not shape:
        return ()
    width = max(col for _row, col in shape) + 1
    return _normalize_shape(tuple((row, width - 1 - col) for row, col in shape))


def _shape_transform(
    before: tuple[tuple[int, int], ...],
    after: tuple[tuple[int, int], ...],
) -> str:
    if before == after:
        return "same"
    rotated = before
    for turns in (1, 2, 3):
        rotated = _rotate_clockwise(rotated)
        if rotated == after:
            return f"rotation:{turns}"
    if _reflect_horizontal(before) == after:
        return "reflection"
    return "unclassified_form_change"
class ActionCouplingMemory:
    """Accumulate which neutral identities move consistently after each action."""

    def __init__(self) -> None:
        self._evidence: dict[int, dict[int, Counter[Delta]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        self._group_evidence: dict[tuple[int, ...], dict[int, Counter[Delta]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        self._levels: dict[int, int] = {}
        self._group_levels: dict[tuple[int, ...], int] = {}

    def observe(self, event: Event) -> None:
        for delta in event.object_deltas:
            if not delta.moved:
                continue
            self._levels.setdefault(delta.object_id, event.levels_before)
            self._evidence[delta.object_id][event.action][delta.translation] += 1

        for group in event.motion_groups:
            if len(group.object_ids) < 2:
                continue
            self._group_levels.setdefault(group.object_ids, event.levels_before)
            self._group_evidence[group.object_ids][event.action][group.translation] += 1

    def summaries(self, *, minimum_moves: int = 2) -> list[dict[str, object]]:
        summaries = []
        for object_id, by_action in self._evidence.items():
            summary = self._summarize(by_action, minimum_moves=minimum_moves)
            if summary is None:
                continue
            summaries.append(summary | {
                "object_id": object_id,
                "first_level": self._levels[object_id],
            })
        summaries.sort(
            key=lambda item: (item["action_dependence"], item["consistency"], item["moves"]),
            reverse=True,
        )
        return summaries

    def co_motion_summaries(self, *, minimum_moves: int = 2) -> list[dict[str, object]]:
        summaries = []
        for object_ids, by_action in self._group_evidence.items():
            summary = self._summarize(by_action, minimum_moves=minimum_moves)
            if summary is None:
                continue
            summaries.append(summary | {
                "object_ids": list(object_ids),
                "first_level": self._group_levels[object_ids],
            })
        summaries.sort(
            key=lambda item: (item["action_dependence"], item["consistency"], item["moves"]),
            reverse=True,
        )
        return summaries

    @staticmethod
    def _summarize(
        by_action: dict[int, Counter[Delta]],
        *,
        minimum_moves: int,
    ) -> dict[str, object] | None:
        total = sum(sum(counter.values()) for counter in by_action.values())
        if total < minimum_moves or len(by_action) < 2:
            return None
        action_models = {}
        correct = 0
        dominant_deltas = set()
        for action, counter in sorted(by_action.items()):
            delta, count = counter.most_common(1)[0]
            dominant_deltas.add(delta)
            observations = sum(counter.values())
            correct += count
            action_models[str(action)] = {
                "delta": list(delta),
                "support": count,
                "observations": observations,
            }
        if len(dominant_deltas) < 2:
            return None
        return {
            "moves": total,
            "consistency": round(correct / total, 4),
            "action_dependence": round(len(dominant_deltas) / len(by_action), 4),
            "actions": action_models,
        }
