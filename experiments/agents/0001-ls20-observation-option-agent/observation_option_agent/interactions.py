"""Geometric interaction evidence around an inferred controlled entity."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .control import ControlEstimate
from .events import Delta, Event
from .tracking import TrackUpdate


@dataclass(frozen=True)
class ContactCandidate:
    object_id: int
    value: int
    shape_id: str
    overlap_pixels: int
    object_area: int
    object_coverage: float


@dataclass(frozen=True)
class InteractionEvent:
    step: int
    action: int
    controlled_object_ids: tuple[int, ...]
    expected_translation: Delta
    observed_translation: Delta
    blocked: bool
    contact_candidates: tuple[ContactCandidate, ...]
    changed_other_object_ids: tuple[int, ...]
    created_other_object_ids: tuple[int, ...]
    disappeared_other_object_ids: tuple[int, ...]


class InteractionDetector:
    """Test which neutral detections occupy an intended destination."""

    def detect(
        self,
        *,
        event: Event,
        tracks: TrackUpdate,
        control: ControlEstimate | None,
        expected_translation: Delta | None,
    ) -> InteractionEvent | None:
        if control is None or expected_translation is None:
            return None

        controlled_ids = set(control.object_ids)
        before_by_id = {match.object_id: match.before for match in tracks.matches}
        before_by_id.update(
            (item.object_id, item.detection) for item in tracks.disappeared
        )
        before_by_id.update(
            (item.object_id, item.detection) for item in tracks.hidden
        )
        controlled_before = [
            before_by_id[object_id]
            for object_id in control.object_ids
            if object_id in before_by_id
        ]
        if not controlled_before:
            return None

        dr, dc = expected_translation
        intended_pixels = {
            (row + dr, col + dc)
            for detection in controlled_before
            for row, col in detection.pixels
        }
        contacts = []
        for object_id, detection in before_by_id.items():
            if object_id in controlled_ids:
                continue
            overlap = len(intended_pixels & detection.pixels)
            coverage = overlap / detection.area
            if overlap and coverage >= 0.5:
                contacts.append(ContactCandidate(
                    object_id=object_id,
                    value=detection.value,
                    shape_id=detection.shape_id,
                    overlap_pixels=overlap,
                    object_area=detection.area,
                    object_coverage=round(coverage, 4),
                ))
        contacts.sort(
            key=lambda item: (-item.object_coverage, -item.overlap_pixels, item.object_id)
        )

        translations = Counter(
            delta.translation
            for delta in event.object_deltas
            if delta.object_id in controlled_ids and delta.moved
        )
        observed = translations.most_common(1)[0][0] if translations else (0, 0)
        changed_other = tuple(sorted(
            delta.object_id
            for delta in event.object_deltas
            if delta.object_id not in controlled_ids
            and (delta.moved or delta.appearance_changed)
        ))
        return InteractionEvent(
            step=event.step,
            action=event.action,
            controlled_object_ids=control.object_ids,
            expected_translation=expected_translation,
            observed_translation=observed,
            blocked=observed != expected_translation,
            contact_candidates=tuple(contacts),
            changed_other_object_ids=changed_other,
            created_other_object_ids=tuple(sorted(
                object_id for object_id in event.created_object_ids
                if object_id not in controlled_ids
            )),
            disappeared_other_object_ids=tuple(sorted(
                object_id for object_id in event.disappeared_object_ids
                if object_id not in controlled_ids
            )),
        )
