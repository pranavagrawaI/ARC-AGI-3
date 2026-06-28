"""First-segment trace alignment and closed-loop recovery for 0001."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Literal

from experiments.harness import Decision, Observation, Transition

from .control import ControlledEntityBelief
from .events import Delta, EventDetector
from .observation import Detection, ObservationParser
from .tracking import ObjectTracker, TrackUpdate
from .traces import TraceStep


ActionSource = Literal[
    "TRACE_PRIOR",
    "LOCAL_RECOVERY",
    "ALIGNMENT_PROBE",
    "FALLBACK_EXPLORE",
]

AlignmentStatus = Literal["CONFIDENT", "NO_CONFIDENT_ALIGNMENT"]
RecoveryStopReason = Literal[
    "aligned",
    "completed",
    "confidence_collapsed",
    "recovery_budget_exhausted",
    "control_conflict",
    "no_delta_reduces_offset",
]

ControlSignature = tuple[tuple[int, int, str], ...]


@dataclass(frozen=True)
class TraceState:
    step: int
    levels_completed: int
    action: int | None
    controlled_center: tuple[float, float] | None
    controlled_signature: ControlSignature
    nearby_signature: ControlSignature


@dataclass(frozen=True)
class TraceReference:
    states: tuple[TraceState, ...]
    segment_actions: tuple[int, ...]
    completion_step: int
    initial_level: int
    action_model: dict[int, Delta]

    @classmethod
    def from_steps(cls, steps: list[TraceStep] | tuple[TraceStep, ...]) -> "TraceReference":
        if not steps:
            raise ValueError("trace reference requires at least one transition")

        parser = ObservationParser()
        tracker = ObjectTracker()
        detector = EventDetector()
        control_belief = ControlledEntityBelief()
        initial_level = steps[0].before.levels_completed
        states: list[TraceState] = []
        actions: list[int] = []

        scene = parser.parse(steps[0].before, frame_index=0)
        tracks = tracker.update(scene)
        states.append(_trace_state(
            step=0,
            observation=steps[0].before,
            action=steps[0].action,
            tracks=tracks,
            control_ids=(),
        ))

        completion_step = len(steps)
        for index, trace_step in enumerate(steps, start=1):
            actions.append(trace_step.action)
            scene = parser.parse(trace_step.after, frame_index=index)
            force_new = (
                trace_step.after.levels_completed != trace_step.before.levels_completed
                or trace_step.after.full_reset
            )
            tracks = tracker.update(scene, force_new=force_new)
            event = detector.detect(
                step=index,
                before=trace_step.before,
                action=trace_step.action,
                after=trace_step.after,
                tracks=tracks,
            )
            control = control_belief.observe(event)
            next_action = steps[index].action if index < len(steps) else None
            states.append(_trace_state(
                step=index,
                observation=trace_step.after,
                action=next_action,
                tracks=tracks,
                control_ids=control.object_ids if control is not None else (),
            ))
            if trace_step.after.levels_completed > initial_level:
                completion_step = index
                break

        actions = actions[:completion_step]
        states = states[:completion_step + 1]
        action_model = {
            action: delta
            for action in sorted({item for state in states for item in (() if state.action is None else (state.action,))})
            if (delta := control_belief.expected_delta(action)) is not None
        }
        return cls(
            states=tuple(states),
            segment_actions=tuple(actions),
            completion_step=completion_step,
            initial_level=initial_level,
            action_model=action_model,
        )

    @classmethod
    def synthetic(
        cls,
        states: list[tuple[tuple[float, float], int | None]],
    ) -> "TraceReference":
        trace_states = tuple(
            TraceState(
                step=index,
                levels_completed=0,
                action=action,
                controlled_center=center,
                controlled_signature=((1, 1, "synthetic-controlled"),),
                nearby_signature=(),
            )
            for index, (center, action) in enumerate(states)
        )
        return cls(
            states=trace_states,
            segment_actions=tuple(
                action for _center, action in states if action is not None
            ),
            completion_step=len(states) - 1,
            initial_level=0,
            action_model={1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)},
        )

    def state(self, step: int) -> TraceState:
        return self.states[min(max(0, step), len(self.states) - 1)]


@dataclass(frozen=True)
class AlignmentResult:
    best_match_step: int
    best_score: float
    second_best_step: int | None
    second_best_score: float | None
    match_margin: float
    alignment_entropy: float
    alignment_status: AlignmentStatus
    window: tuple[int, int]


class TraceAlignment:
    def __init__(
        self,
        *,
        back_window: int = 2,
        forward_window: int = 4,
        confident_score: float = 8.0,
        confident_margin: float = 0.5,
    ) -> None:
        self.back_window = back_window
        self.forward_window = forward_window
        self.confident_score = confident_score
        self.confident_margin = confident_margin

    def align(
        self,
        *,
        current_center: tuple[float, float] | None,
        current_signature: ControlSignature,
        reference: TraceReference,
        last_matched_step: int,
    ) -> AlignmentResult:
        start = max(0, last_matched_step - self.back_window)
        end = min(len(reference.states) - 1, last_matched_step + self.forward_window)
        scored = [
            (self._score(current_center, current_signature, reference.states[index]), index)
            for index in range(start, end + 1)
        ]
        scored.sort()
        best_score, best_step = scored[0]
        second_score, second_step = scored[1] if len(scored) > 1 else (None, None)
        margin = (second_score - best_score) if second_score is not None else math.inf
        entropy = _entropy([score for score, _index in scored])
        status: AlignmentStatus = (
            "CONFIDENT"
            if best_score <= self.confident_score and margin >= self.confident_margin
            else "NO_CONFIDENT_ALIGNMENT"
        )
        return AlignmentResult(
            best_match_step=best_step,
            best_score=round(best_score, 4),
            second_best_step=second_step,
            second_best_score=round(second_score, 4) if second_score is not None else None,
            match_margin=round(margin, 4) if math.isfinite(margin) else margin,
            alignment_entropy=round(entropy, 4),
            alignment_status=status,
            window=(start, end),
        )

    @staticmethod
    def _score(
        current_center: tuple[float, float] | None,
        current_signature: ControlSignature,
        state: TraceState,
    ) -> float:
        if current_center is None or state.controlled_center is None:
            center_score = 100.0
        else:
            center_score = abs(current_center[0] - state.controlled_center[0]) + abs(
                current_center[1] - state.controlled_center[1]
            )
        signature_penalty = 0.0 if current_signature == state.controlled_signature else 1.0
        return center_score + signature_penalty


@dataclass(frozen=True)
class InjectionRecord:
    injected_step: int
    original_action: int
    injected_action: int
    reason: str
    valid: bool | None = None


class D1InjectionPolicy:
    def __init__(
        self,
        *,
        primary_step: int = 8,
        backup_steps: tuple[int, ...] = (9, 10),
        alternatives: dict[int, int] | None = None,
    ) -> None:
        self.primary_step = primary_step
        self.backup_steps = backup_steps
        self.alternatives = alternatives or {primary_step: 3, 9: 3, 10: 3}
        self.records: list[InjectionRecord] = []

    def candidate_steps(self) -> tuple[int, ...]:
        return (self.primary_step,) + self.backup_steps

    def action_for(self, *, live_step: int, original_action: int) -> int | None:
        if any(record.valid for record in self.records):
            return None
        if live_step not in self.candidate_steps():
            return None
        injected = self.alternatives.get(live_step)
        if injected is None or injected == original_action:
            return None
        return injected

    def record_for(
        self,
        *,
        live_step: int,
        original_action: int,
        controlled_displaced: bool,
    ) -> InjectionRecord:
        injected = self.alternatives[live_step]
        record = InjectionRecord(
            injected_step=live_step,
            original_action=original_action,
            injected_action=injected,
            reason="harmless_displacement",
            valid=controlled_displaced,
        )
        self.records.append(record)
        return record


class ControlVerifier:
    def __init__(self, *, conflict_limit: int = 2) -> None:
        self.conflict_limit = conflict_limit
        self._model: dict[int, Delta] = {}
        self._conflicts: Counter[int] = Counter()
        self.confirmations = 0
        self.conflicts = 0

    def seed(self, action_model: dict[int, Delta]) -> None:
        self._model.update(action_model)

    def observe(
        self,
        *,
        action: int,
        expected: Delta | None,
        observed: Delta | None,
    ) -> None:
        if expected is None or observed is None:
            return
        if expected == observed:
            self.confirmations += 1
            if self._conflicts[action] > 0:
                self._conflicts[action] -= 1
            self._model[action] = expected
            return
        self.conflicts += 1
        self._conflicts[action] += 1

    def available_model(self, available_actions: tuple[int, ...]) -> dict[int, Delta]:
        return {
            action: delta
            for action, delta in self._model.items()
            if action in available_actions and self._conflicts[action] < self.conflict_limit
        }

    def suppressed(self, action: int) -> bool:
        return self._conflicts[action] >= self.conflict_limit


@dataclass
class TraceAssistedReplayAgent:
    reference: TraceReference
    recovery_budget: int = 8
    alignment: TraceAlignment = field(default_factory=TraceAlignment)

    def __post_init__(self) -> None:
        self.parser = ObservationParser()
        self.tracker = ObjectTracker()
        self.detector = EventDetector()
        self.control_belief = ControlledEntityBelief()
        self.control_verifier = ControlVerifier()
        self.control_verifier.seed(self.reference.action_model)
        self.actions = 0
        self.last_matched_step = 0
        self._trace_cursor = 0
        self._recovery_target_step: int | None = None
        self._tracks: TrackUpdate | None = None
        self._last_alignment: AlignmentResult | None = None
        self._last_source: ActionSource | None = None
        self._recovery_actions = 0
        self._recovery_stop_reason: RecoveryStopReason | None = None
        self._records: list[dict[str, object]] = []
        self._action_sources: Counter[str] = Counter()

    def act(self, observation: Observation) -> Decision:
        self._ensure_initialized(observation)
        current_center, current_signature = self._controlled_state()
        alignment = self.alignment.align(
            current_center=current_center,
            current_signature=current_signature,
            reference=self.reference,
            last_matched_step=self.last_matched_step,
        )
        self._last_alignment = alignment
        nominal_step = min(self._trace_cursor, self.reference.completion_step)
        target_step = (
            self._recovery_target_step
            if self._recovery_target_step is not None
            else nominal_step
        )
        target_step = min(target_step, self.reference.completion_step)
        target = self.reference.state(nominal_step)
        verified_model = self.control_verifier.available_model(observation.available_actions)

        source: ActionSource
        action: int | None = None
        if observation.levels_completed > self.reference.initial_level:
            source = "FALLBACK_EXPLORE"
            action = observation.available_actions[0]
            self._recovery_stop_reason = "completed"
        elif current_center is None or target.controlled_center is None:
            self.last_matched_step = nominal_step
            source = "TRACE_PRIOR"
            action = self._trace_action(observation, nominal_step)
        elif alignment.alignment_status == "NO_CONFIDENT_ALIGNMENT":
            source = "ALIGNMENT_PROBE"
            target = self.reference.state(target_step)
            action = self.choose_recovery_action(
                current_center=current_center,
                target_center=target.controlled_center,
                available_actions=observation.available_actions,
                verified_action_model=verified_model,
            )
            if action is None:
                action = self._fallback(observation)
        else:
            self.last_matched_step = alignment.best_match_step
            target = self.reference.state(target_step)
            needs_recovery = self._needs_recovery(current_center, target.controlled_center)
            action = self.choose_recovery_action(
                current_center=current_center,
                target_center=target.controlled_center,
                available_actions=observation.available_actions,
                verified_action_model=verified_model,
            )
            if action is not None and needs_recovery:
                if self._recovery_target_step is None:
                    self._recovery_target_step = target_step
                source = "LOCAL_RECOVERY"
                self._recovery_actions += 1
                if self._recovery_actions > self.recovery_budget:
                    self._recovery_stop_reason = "recovery_budget_exhausted"
                    action = self._trace_action(observation, nominal_step)
                    source = "TRACE_PRIOR"
            else:
                if self._recovery_target_step is not None:
                    self._trace_cursor = self._recovery_target_step
                    self._recovery_target_step = None
                if self._recovery_actions and self._recovery_stop_reason is None:
                    self._recovery_stop_reason = "aligned"
                source = "TRACE_PRIOR"
                action = self._trace_action(observation, self._trace_cursor)

        if action is None:
            source = "FALLBACK_EXPLORE"
            action = self._fallback(observation)
        self._last_source = source
        self._action_sources[source] += 1
        return Decision(action=action, reasoning=self._reasoning(source, alignment))

    def observe(self, transition: Transition) -> None:
        self.actions += 1
        before_center, _before_signature = self._controlled_state()
        scene = self.parser.parse(transition.after, frame_index=self.actions)
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
        expected = self.control_verifier.available_model(
            transition.before.available_actions
        ).get(transition.decision.action)
        control = self.control_belief.observe(event)
        self._tracks = tracks
        after_center, _after_signature = self._controlled_state()
        observed = _delta_between(before_center, after_center)
        if expected is None:
            expected = self.control_belief.expected_delta(transition.decision.action)
        self.control_verifier.observe(
            action=transition.decision.action,
            expected=expected,
            observed=observed,
        )
        if (
            self._recovery_stop_reason is None
            and self._last_alignment is not None
            and self._last_alignment.alignment_status == "NO_CONFIDENT_ALIGNMENT"
            and self._last_source == "ALIGNMENT_PROBE"
        ):
            self._recovery_stop_reason = "confidence_collapsed"
        if (
            transition.after.levels_completed > self.reference.initial_level
            and self._recovery_actions
        ):
            self._recovery_stop_reason = "completed"
        if control is not None:
            self.control_verifier.seed({
                action: delta
                for action in transition.after.available_actions
                if (delta := self.control_belief.expected_delta(action)) is not None
            })
        if self._last_source == "TRACE_PRIOR":
            self._trace_cursor = min(
                self._trace_cursor + 1,
                self.reference.completion_step,
            )
        self._records.append({
            "step": self.actions,
            "action": transition.decision.action,
            "action_source": self._last_source,
            "levels_before": transition.before.levels_completed,
            "levels_after": transition.after.levels_completed,
            "state_after": transition.after.state,
            "alignment": _alignment_record(self._last_alignment),
            "control_confirmations": self.control_verifier.confirmations,
            "control_conflicts": self.control_verifier.conflicts,
        })

    def choose_recovery_action(
        self,
        *,
        current_center: tuple[float, float] | None,
        target_center: tuple[float, float] | None,
        available_actions: tuple[int, ...],
        verified_action_model: dict[int, Delta],
    ) -> int | None:
        if current_center is None or target_center is None:
            return None
        current_distance = _distance(current_center, target_center)
        best: tuple[float, int] | None = None
        for action in available_actions:
            delta = verified_action_model.get(action)
            if delta is None:
                continue
            next_center = (current_center[0] + delta[0], current_center[1] + delta[1])
            distance = _distance(next_center, target_center)
            if distance >= current_distance:
                continue
            candidate = (distance, action)
            if best is None or candidate < best:
                best = candidate
        return best[1] if best else None

    @property
    def records(self) -> tuple[dict[str, object], ...]:
        return tuple(self._records)

    def summary(self) -> dict[str, object]:
        return {
            "actions": self.actions,
            "action_source_distribution": dict(sorted(self._action_sources.items())),
            "recovery_actions": self._recovery_actions,
            "recovery_stop_reason": self._recovery_stop_reason,
            "live_control_confirmations": self.control_verifier.confirmations,
            "live_control_conflicts": self.control_verifier.conflicts,
            "max_alignment_uncertainty": max(
                (
                    record["alignment"]["alignment_entropy"]
                    for record in self._records
                    if record.get("alignment")
                ),
                default=0.0,
            ),
        }

    def _ensure_initialized(self, observation: Observation) -> None:
        if self._tracks is not None:
            return
        scene = self.parser.parse(observation, frame_index=0)
        self._tracks = self.tracker.update(scene)

    def _controlled_state(self) -> tuple[tuple[float, float] | None, ControlSignature]:
        if self._tracks is None or self.control_belief.active is None:
            return None, ()
        detections = _controlled_detections(self._tracks, self.control_belief.active.object_ids)
        return _center_and_signature(detections)

    @staticmethod
    def _needs_recovery(
        current_center: tuple[float, float] | None,
        target_center: tuple[float, float] | None,
    ) -> bool:
        return (
            current_center is not None
            and target_center is not None
            and _distance(current_center, target_center) > 0.25
        )

    def _trace_action(self, observation: Observation, nominal_step: int) -> int:
        action = self.reference.state(nominal_step).action
        if action in observation.available_actions:
            return action
        return self._fallback(observation)

    @staticmethod
    def _fallback(observation: Observation) -> int:
        return observation.available_actions[0]

    @staticmethod
    def _reasoning(source: ActionSource, alignment: AlignmentResult) -> dict[str, object]:
        return {
            "agent": "0001-first-segment-trace-alignment",
            "action_source": source,
            "alignment": _alignment_record(alignment),
        }


@dataclass
class PureReplayAgent:
    actions: tuple[int, ...]

    def __post_init__(self) -> None:
        self.index = 0
        self.records: list[dict[str, object]] = []

    def act(self, observation: Observation) -> Decision:
        action = self.actions[min(self.index, len(self.actions) - 1)]
        if action not in observation.available_actions:
            action = observation.available_actions[0]
        return Decision(
            action=action,
            reasoning={
                "agent": "0001-pure-replay-baseline",
                "action_source": "TRACE_PRIOR",
            },
        )

    def observe(self, transition: Transition) -> None:
        self.index += 1
        self.records.append({
            "step": self.index,
            "action": transition.decision.action,
            "action_source": "TRACE_PRIOR",
            "levels_before": transition.before.levels_completed,
            "levels_after": transition.after.levels_completed,
            "state_after": transition.after.state,
        })


def _trace_state(
    *,
    step: int,
    observation: Observation,
    action: int | None,
    tracks: TrackUpdate,
    control_ids: tuple[int, ...],
) -> TraceState:
    controlled = _controlled_detections(tracks, control_ids)
    center, signature = _center_and_signature(controlled)
    nearby = _nearby_signature(tracks, controlled)
    return TraceState(
        step=step,
        levels_completed=observation.levels_completed,
        action=action,
        controlled_center=center,
        controlled_signature=signature,
        nearby_signature=nearby,
    )


def _controlled_detections(
    tracks: TrackUpdate,
    control_ids: tuple[int, ...],
) -> list[Detection]:
    wanted = set(control_ids)
    return [
        item.detection
        for item in tracks.current
        if item.object_id in wanted
    ]


def _center_and_signature(
    detections: list[Detection],
) -> tuple[tuple[float, float] | None, ControlSignature]:
    if not detections:
        return None, ()
    area = sum(detection.area for detection in detections)
    center = (
        sum(detection.centroid[0] * detection.area for detection in detections) / area,
        sum(detection.centroid[1] * detection.area for detection in detections) / area,
    )
    signature = tuple(sorted(
        (detection.value, detection.area, detection.shape_id)
        for detection in detections
    ))
    return center, signature


def _nearby_signature(
    tracks: TrackUpdate,
    controlled: list[Detection],
    *,
    radius: float = 12.0,
) -> ControlSignature:
    center, _signature = _center_and_signature(controlled)
    if center is None:
        return ()
    controlled_pixels = {pixel for detection in controlled for pixel in detection.pixels}
    nearby = []
    for item in tracks.current:
        if item.detection.pixels & controlled_pixels:
            continue
        if _distance(center, item.detection.centroid) <= radius:
            nearby.append((item.detection.value, item.detection.area, item.detection.shape_id))
    return tuple(sorted(nearby))


def _entropy(scores: list[float]) -> float:
    if not scores:
        return 0.0
    weights = [math.exp(-min(score, 100.0)) for score in scores]
    total = sum(weights)
    if total == 0:
        return 0.0
    probabilities = [weight / total for weight in weights]
    return -sum(p * math.log(p) for p in probabilities if p > 0)


def _distance(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _delta_between(
    before: tuple[float, float] | None,
    after: tuple[float, float] | None,
) -> Delta | None:
    if before is None or after is None:
        return None
    return (int(round(after[0] - before[0])), int(round(after[1] - before[1])))


def _alignment_record(result: AlignmentResult | None) -> dict[str, object] | None:
    if result is None:
        return None
    return {
        "best_match_step": result.best_match_step,
        "best_score": result.best_score,
        "second_best_step": result.second_best_step,
        "second_best_score": result.second_best_score,
        "match_margin": result.match_margin,
        "alignment_entropy": result.alignment_entropy,
        "alignment_status": result.alignment_status,
        "window": list(result.window),
    }
