"""Live routing and contact probing for neutral relation proposals."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from experiments.harness import Decision, Observation, Transition

from .control import ControlEstimate, ControlledEntityBelief
from .events import EventDetector
from .interactions import InteractionDetector, InteractionEvent
from .observation import Detection, ObservationParser
from .probing import InterventionResult, ProbeProposal
from .tracking import ObjectTracker, TrackUpdate
from .transition_learning import ContactSignature, contact_effect_symbols_from_record


PendingProbe = Literal["positive", "negative"]


@dataclass(frozen=True)
class LiveProbeSummary:
    actions: int
    bootstrap_actions: int
    positive_attempts: int
    positive_successes: int
    negative_controls: int
    negative_control_violations: int
    completed: bool
    notes: tuple[str, ...]


def _signature_from_detections(detections: list[Detection]) -> ContactSignature:
    return tuple(sorted(
        (detection.value, detection.area, detection.shape_id)
        for detection in detections
    ))


def _contact_signature_for_action(
    *,
    tracks: TrackUpdate,
    control: ControlEstimate,
    action: int,
) -> ContactSignature:
    expected_delta = control.action_model.get(action)
    if expected_delta is None:
        return ()
    by_id = {item.object_id: item.detection for item in tracks.current}
    controlled = [
        by_id[object_id]
        for object_id in control.object_ids
        if object_id in by_id
    ]
    if not controlled:
        return ()
    controlled_ids = set(control.object_ids)
    dr, dc = expected_delta
    intended_pixels = {
        (row + dr, col + dc)
        for detection in controlled
        for row, col in detection.pixels
    }
    contacts = []
    for object_id, detection in by_id.items():
        if object_id in controlled_ids:
            continue
        overlap = len(intended_pixels & detection.pixels)
        if overlap and overlap / detection.area >= 0.5:
            contacts.append(detection)
    return _signature_from_detections(contacts)


def choose_contact_action(
    *,
    proposal: ProbeProposal,
    tracks: TrackUpdate,
    control: ControlEstimate,
    available_actions: tuple[int, ...],
) -> int | None:
    for action in available_actions:
        if _contact_signature_for_action(
            tracks=tracks,
            control=control,
            action=action,
        ) == proposal.contact_signature:
            return action
    return None


def choose_nonmatching_contact_action(
    *,
    proposal: ProbeProposal,
    tracks: TrackUpdate,
    control: ControlEstimate,
    available_actions: tuple[int, ...],
) -> int | None:
    for action in available_actions:
        signature = _contact_signature_for_action(
            tracks=tracks,
            control=control,
            action=action,
        )
        if signature and signature != proposal.contact_signature:
            return action
    return None


def choose_route_action(
    *,
    proposal: ProbeProposal,
    tracks: TrackUpdate,
    control: ControlEstimate,
    available_actions: tuple[int, ...],
) -> int | None:
    by_id = {item.object_id: item.detection for item in tracks.current}
    controlled = [
        by_id[object_id]
        for object_id in control.object_ids
        if object_id in by_id
    ]
    if not controlled:
        return None
    controlled_center = _centroid(controlled)
    matched = _matching_detections(proposal.contact_signature, tuple(by_id.values()))
    if not matched:
        return None
    target_center = _centroid(matched)
    best = None
    for action in available_actions:
        delta = control.action_model.get(action)
        if delta is None:
            continue
        next_center = (
            controlled_center[0] + delta[0],
            controlled_center[1] + delta[1],
        )
        distance = abs(next_center[0] - target_center[0]) + abs(
            next_center[1] - target_center[1]
        )
        candidate = (distance, action)
        if best is None or candidate < best:
            best = candidate
    return best[1] if best else None


def choose_route_to_nonmatching_action(
    *,
    proposal: ProbeProposal,
    tracks: TrackUpdate,
    control: ControlEstimate,
    available_actions: tuple[int, ...],
) -> int | None:
    by_id = {item.object_id: item.detection for item in tracks.current}
    controlled = [
        by_id[object_id]
        for object_id in control.object_ids
        if object_id in by_id
    ]
    if not controlled:
        return None
    controlled_ids = set(control.object_ids)
    proposal_descriptors = Counter(proposal.contact_signature)
    proposal_area = sum(area for _value, area, _shape_id in proposal.contact_signature)
    max_candidate_area = max(1, proposal_area * 4)
    candidates = []
    for object_id, detection in by_id.items():
        if object_id in controlled_ids:
            continue
        descriptor = (detection.value, detection.area, detection.shape_id)
        if proposal_descriptors[descriptor] > 0:
            continue
        if detection.area > max_candidate_area:
            continue
        candidates.append(detection)
    if not candidates:
        return None

    controlled_center = _centroid(controlled)
    best = None
    for target in candidates:
        target_center = _centroid([target])
        for action in available_actions:
            delta = control.action_model.get(action)
            if delta is None:
                continue
            next_center = (
                controlled_center[0] + delta[0],
                controlled_center[1] + delta[1],
            )
            distance = abs(next_center[0] - target_center[0]) + abs(
                next_center[1] - target_center[1]
            )
            candidate = (distance, action)
            if best is None or candidate < best:
                best = candidate
    return best[1] if best else None


def _centroid(detections: list[Detection]) -> tuple[float, float]:
    area = sum(detection.area for detection in detections)
    return (
        sum(detection.centroid[0] * detection.area for detection in detections) / area,
        sum(detection.centroid[1] * detection.area for detection in detections) / area,
    )


def _matching_detections(
    signature: ContactSignature,
    detections: tuple[Detection, ...],
) -> list[Detection]:
    needed = Counter(signature)
    output = []
    for detection in detections:
        descriptor = (detection.value, detection.area, detection.shape_id)
        if needed[descriptor] <= 0:
            continue
        needed[descriptor] -= 1
        output.append(detection)
    return output if not +needed else []


class LiveContactProbeAgent:
    """Trace-assisted live probe controller for one neutral proposal.

    Optional bootstrap actions are public action IDs from admissible traces. Once
    the bootstrap prefix is exhausted, the agent uses only live observations and
    its learned action-motion belief to route to and probe the proposal.
    """

    def __init__(
        self,
        *,
        proposal: ProbeProposal,
        bootstrap_actions: tuple[int, ...] = (),
        positive_attempt_limit: int = 1,
        negative_control_limit: int = 1,
    ) -> None:
        self.proposal = proposal
        self.bootstrap_actions = bootstrap_actions
        self.positive_attempt_limit = positive_attempt_limit
        self.negative_control_limit = negative_control_limit
        self.parser = ObservationParser()
        self.tracker = ObjectTracker()
        self.detector = EventDetector()
        self.control_belief = ControlledEntityBelief()
        self.interaction_detector = InteractionDetector()
        self.actions = 0
        self._frame_index = 0
        self._bootstrap_index = 0
        self._tracks: TrackUpdate | None = None
        self._pending: PendingProbe | None = None
        self._positive_attempts = 0
        self._positive_successes = 0
        self._negative_controls = 0
        self._negative_control_violations = 0
        self._notes: list[str] = []
        self.done = False

    def act(self, observation: Observation) -> Decision:
        self._ensure_initialized(observation)
        if self._bootstrap_index < len(self.bootstrap_actions):
            action = self.bootstrap_actions[self._bootstrap_index]
            self._bootstrap_index += 1
            return self._decision(action, phase="bootstrap")

        if self.done:
            return self._decision(observation.available_actions[0], phase="done")

        tracks = self._tracks
        control = self._expanded_control(observation.available_actions)
        if tracks is None or control is None:
            action = observation.available_actions[self.actions % len(observation.available_actions)]
            return self._decision(action, phase="learn_control")

        if self._positive_attempts < self.positive_attempt_limit:
            action = choose_contact_action(
                proposal=self.proposal,
                tracks=tracks,
                control=control,
                available_actions=observation.available_actions,
            )
            if action is not None:
                self._pending = "positive"
                return self._decision(action, phase="positive_contact")
            action = choose_route_action(
                proposal=self.proposal,
                tracks=tracks,
                control=control,
                available_actions=observation.available_actions,
            )
            if action is not None:
                return self._decision(action, phase="route_positive")

        if self._negative_controls < self.negative_control_limit:
            action = choose_nonmatching_contact_action(
                proposal=self.proposal,
                tracks=tracks,
                control=control,
                available_actions=observation.available_actions,
            )
            if action is not None:
                self._pending = "negative"
                return self._decision(action, phase="negative_contact")
            action = choose_route_to_nonmatching_action(
                proposal=self.proposal,
                tracks=tracks,
                control=control,
                available_actions=observation.available_actions,
            )
            if action is not None:
                return self._decision(action, phase="route_negative")

        self.done = True
        return self._decision(observation.available_actions[0], phase="complete")

    def observe(self, transition: Transition) -> None:
        self.actions += 1
        scene = self.parser.parse(transition.after, frame_index=self._frame_index + 1)
        force_new = (
            transition.after.levels_completed != transition.before.levels_completed
            or transition.after.full_reset
        )
        tracks = self.tracker.update(scene, force_new=force_new)
        event = self.detector.detect(
            step=self.actions,
            before=transition.before,
            action=transition.decision.action,
            after=transition.after,
            tracks=tracks,
        )
        prior_control = self.control_belief.active
        expected_translation = self.control_belief.expected_delta(event.action)
        interaction = self.interaction_detector.detect(
            event=event,
            tracks=tracks,
            control=prior_control,
            expected_translation=expected_translation,
        )
        self._observe_pending(event, interaction)
        self.control_belief.observe(event)
        self._tracks = tracks
        self._frame_index += 1

    def result(self) -> InterventionResult:
        return InterventionResult(
            proposal=self.proposal,
            positive_attempts=self._positive_attempts,
            positive_successes=self._positive_successes,
            negative_controls=self._negative_controls,
            negative_control_violations=self._negative_control_violations,
        )

    def summary(self) -> LiveProbeSummary:
        return LiveProbeSummary(
            actions=self.actions,
            bootstrap_actions=self._bootstrap_index,
            positive_attempts=self._positive_attempts,
            positive_successes=self._positive_successes,
            negative_controls=self._negative_controls,
            negative_control_violations=self._negative_control_violations,
            completed=self.done,
            notes=tuple(self._notes),
        )

    def _ensure_initialized(self, observation: Observation) -> None:
        if self._tracks is not None:
            return
        scene = self.parser.parse(observation, frame_index=self._frame_index)
        self._tracks = self.tracker.update(scene)

    def _decision(self, action: int, *, phase: str) -> Decision:
        return Decision(
            action=action,
            reasoning={
                "agent": "0001-live-contact-probe",
                "phase": phase,
                "predicted_effect": self.proposal.predicted_effect,
            },
        )

    def _expanded_control(
        self,
        available_actions: tuple[int, ...],
    ) -> ControlEstimate | None:
        active = self.control_belief.active
        if active is None:
            return None
        action_model = dict(active.action_model)
        for action in available_actions:
            expected = self.control_belief.expected_delta(action)
            if expected is not None:
                action_model[action] = expected
        return ControlEstimate(
            level=active.level,
            object_ids=active.object_ids,
            support=active.support,
            distinct_actions=len(action_model),
            consistency=active.consistency,
            confidence=active.confidence,
            source=active.source,
            action_model=dict(sorted(action_model.items())),
            discovered_step=active.discovered_step,
            actions_into_level=active.actions_into_level,
        )

    def _observe_pending(
        self,
        event,
        interaction: InteractionEvent | None,
    ) -> None:
        if self._pending is None or interaction is None:
            self._pending = None
            return
        record = _record_from_event(event, interaction)
        signature = _interaction_signature(interaction)
        effects = contact_effect_symbols_from_record(record)
        if self._pending == "positive" and signature == self.proposal.contact_signature:
            self._positive_attempts += 1
            if self.proposal.predicted_effect in effects:
                self._positive_successes += 1
        elif self._pending == "negative" and signature != self.proposal.contact_signature:
            self._negative_controls += 1
            if self.proposal.predicted_effect in effects:
                self._negative_control_violations += 1
        else:
            self._notes.append(
                f"pending_{self._pending}_did_not_match_observed_contact"
            )
        self._pending = None
        if (
            self._positive_attempts >= self.positive_attempt_limit
            and self._negative_controls >= self.negative_control_limit
        ):
            self.done = True


def _interaction_signature(interaction: InteractionEvent) -> ContactSignature:
    return tuple(sorted(
        (item.value, item.object_area, item.shape_id)
        for item in interaction.contact_candidates
    ))


def _record_from_event(event, interaction: InteractionEvent) -> dict[str, object]:
    return {
        "action": event.action,
        "levels_before": event.levels_before,
        "levels_after": event.levels_after,
        "full_reset": event.full_reset,
        "created_object_ids": list(event.created_object_ids),
        "disappeared_object_ids": list(event.disappeared_object_ids),
        "reappeared_object_ids": list(event.reappeared_object_ids),
        "interaction": {
            "blocked": interaction.blocked,
            "changed_other_object_ids": list(interaction.changed_other_object_ids),
            "contact_candidates": [
                {
                    "value": item.value,
                    "object_area": item.object_area,
                    "shape_id": item.shape_id,
                }
                for item in interaction.contact_candidates
            ],
        },
        "object_deltas": [
            {
                "object_id": delta.object_id,
                "translation": list(delta.translation),
                "shape_transform": delta.shape_transform,
                "shape_changed": delta.shape_changed,
                "value_changed": delta.value_changed,
                "area_delta": delta.area_delta,
            }
            for delta in event.object_deltas
        ],
    }
