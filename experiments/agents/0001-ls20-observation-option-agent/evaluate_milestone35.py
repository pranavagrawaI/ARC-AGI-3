"""Scout for naturally available live semantic drift via legal actions.

This is an optional M3.5 bridge. It never mutates private environment state.
It tries public legal-action substitutions and asks whether the resulting public
observation becomes semantically incompatible with a nearby trace corridor.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
AGENT_ROOT = Path(__file__).resolve().parent
for path in (REPO, AGENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import arc_agi
from arc_agi.base import OperationMode
from arcengine import GameAction

from experiments.harness import Decision, Transition
from experiments.harness.runner import public_observation

from observation_option_agent import (
    ControlledEntityBelief,
    DriftClassifier,
    EventDetector,
    ObjectTracker,
    ObservationParser,
    SemanticSignature,
    TraceCorridor,
    TraceReference,
    TraceStep,
    iter_trace_steps,
)


DEFAULT_TRACE = (
    REPO
    / "recordings/ae20a3e4-ca6e-4f29-b463-042fd0dd6710"
    / "ls20-9607627b-b7f0efe3-1ae2-4f66-ad6e-178cf992b58f.jsonl"
)
DEFAULT_OUTPUT = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-35-summary.json"
)
DEFAULT_EVENTS = (
    REPO
    / "experiments/artifacts/0001-ls20-observation-option-agent"
    / "milestone-35-events.jsonl"
)
MAX_SEMANTIC_AREA = 200


@dataclass
class _LiveBelief:
    parser: ObservationParser
    tracker: ObjectTracker
    detector: EventDetector
    control: ControlledEntityBelief
    frame_index: int = 0
    tracks: Any | None = None

    @classmethod
    def create(cls):
        return cls(
            parser=ObservationParser(),
            tracker=ObjectTracker(),
            detector=EventDetector(),
            control=ControlledEntityBelief(),
        )

    def initialize(self, observation) -> None:
        scene = self.parser.parse(observation, frame_index=0)
        self.tracks = self.tracker.update(scene)

    def observe(self, transition: Transition) -> None:
        self.frame_index += 1
        scene = self.parser.parse(transition.after, frame_index=self.frame_index)
        force_new = (
            transition.after.levels_completed != transition.before.levels_completed
            or transition.after.full_reset
        )
        self.tracks = self.tracker.update(scene, force_new=force_new)
        event = self.detector.detect(
            step=self.frame_index,
            before=transition.before,
            action=transition.decision.action,
            after=transition.after,
            tracks=self.tracks,
        )
        self.control.observe(event)

    def controlled_center(self) -> tuple[float, float] | None:
        if self.tracks is None or self.control.active is None:
            return None
        detections = self._controlled_detections()
        if not detections:
            return None
        area = sum(item.area for item in detections)
        return (
            sum(item.centroid[0] * item.area for item in detections) / area,
            sum(item.centroid[1] * item.area for item in detections) / area,
        )

    def semantic_signature(self) -> SemanticSignature:
        if self.tracks is None or self.control.active is None:
            return SemanticSignature.from_labels(())
        controlled = set(self.control.active.object_ids)
        labels = []
        for item in self.tracks.current:
            if item.object_id in controlled:
                continue
            if item.detection.area > MAX_SEMANTIC_AREA:
                continue
            labels.append(
                _absolute_label(item.detection)
            )
        return SemanticSignature.from_labels(labels)

    def _controlled_detections(self):
        if self.tracks is None or self.control.active is None:
            return []
        controlled = set(self.control.active.object_ids)
        return [
            item.detection
            for item in self.tracks.current
            if item.object_id in controlled
        ]


def _absolute_label(detection) -> str:
    center = (
        int(round(detection.centroid[0])),
        int(round(detection.centroid[1])),
    )
    return (
        f"value={detection.value}|area={detection.area}|"
        f"shape={detection.shape_id}|center={center}"
    )


def _semantic_index(steps: tuple[TraceStep, ...]) -> dict[int, SemanticSignature]:
    parser = ObservationParser()
    tracker = ObjectTracker()
    detector = EventDetector()
    control = ControlledEntityBelief()
    output: dict[int, SemanticSignature] = {}

    scene = parser.parse(steps[0].before, frame_index=0)
    tracks = tracker.update(scene)
    output[0] = _signature_from_tracks(tracks, ())
    for index, step in enumerate(steps, start=1):
        scene = parser.parse(step.after, frame_index=index)
        force_new = (
            step.after.levels_completed != step.before.levels_completed
            or step.after.full_reset
        )
        tracks = tracker.update(scene, force_new=force_new)
        event = detector.detect(
            step=index,
            before=step.before,
            action=step.action,
            after=step.after,
            tracks=tracks,
        )
        estimate = control.observe(event)
        output[index] = _signature_from_tracks(
            tracks,
            estimate.object_ids if estimate is not None else (),
        )
        if step.after.levels_completed > steps[0].before.levels_completed:
            break
    return output


def _signature_from_tracks(tracks, controlled_ids: tuple[int, ...]) -> SemanticSignature:
    controlled = set(controlled_ids)
    labels = []
    for item in tracks.current:
        if item.object_id in controlled:
            continue
        if item.detection.area > MAX_SEMANTIC_AREA:
            continue
        labels.append(_absolute_label(item.detection))
    return SemanticSignature.from_labels(labels)


def _absolute_corridor(
    reference: TraceReference,
    semantic_index: dict[int, SemanticSignature],
    *,
    start_index: int,
    length: int = 5,
) -> TraceCorridor:
    states = []
    for index in range(start_index, min(reference.completion_step, start_index + length) + 1):
        state = reference.state(index)
        if state.controlled_center is None:
            continue
        signature = semantic_index.get(index, SemanticSignature.from_labels(()))
        states.append((
            index,
            (int(round(state.controlled_center[0])), int(round(state.controlled_center[1]))),
            tuple((label,) for label in signature.labels),
        ))
    return TraceCorridor.synthetic(states)


def _step_env(env, current, action_id: int, *, reason: str) -> tuple[Any, Transition]:
    action = GameAction.from_id(action_id)
    next_frame = env.step(
        action,
        reasoning={"agent": "0001-m35-live-semantic-drift-scout", "reason": reason},
    )
    following = public_observation(next_frame)
    transition = Transition(
        current,
        Decision(action=action_id, reasoning={"reason": reason}),
        following,
    )
    return following, transition


def _run_probe(
    *,
    game: str,
    reference: TraceReference,
    semantic_index: dict[int, SemanticSignature],
    prefix_actions: tuple[int, ...],
    probe_step: int,
    probe_action: int,
) -> dict[str, Any]:
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make(game)
    current = public_observation(env.reset())
    belief = _LiveBelief.create()
    belief.initialize(current)

    for action in prefix_actions:
        current, transition = _step_env(env, current, action, reason="trace_prefix")
        belief.observe(transition)

    before_level = current.levels_completed
    current, transition = _step_env(env, current, probe_action, reason="legal_substitution_probe")
    belief.observe(transition)

    corridor = _absolute_corridor(
        reference,
        semantic_index,
        start_index=probe_step,
        length=5,
    )
    result = DriftClassifier().classify(
        current_center=belief.controlled_center(),
        current_signature=belief.semantic_signature(),
        corridor=corridor,
        action_result=None,
    )
    return {
        "probe_step": probe_step,
        "probe_action": probe_action,
        "levels_before": before_level,
        "levels_after": current.levels_completed,
        "state_after": current.state,
        "drift_type": result.drift_type,
        "chosen_corridor_index": result.chosen_corridor_index,
        "failure_reason": result.failure_reason,
        "semantic_features_changed": list(result.semantic_features_changed),
        "corridor_candidates": list(result.corridor_candidates),
        "control_available": belief.control.active is not None,
    }


def evaluate(
    *,
    game: str,
    trace_path: Path,
    output_events: Path,
    max_probe_step: int,
) -> dict[str, Any]:
    steps = tuple(iter_trace_steps(trace_path))
    reference = TraceReference.from_steps(steps)
    semantic_index = _semantic_index(steps)
    trace_actions = reference.segment_actions
    output_events.parent.mkdir(parents=True, exist_ok=True)
    records = []
    found = None
    with output_events.open("w", encoding="utf-8") as stream:
        for probe_step in range(6, min(max_probe_step, reference.completion_step) + 1):
            if probe_step < 1 or probe_step > len(trace_actions):
                continue
            original = trace_actions[probe_step - 1]
            prefix = trace_actions[: probe_step - 1]
            for action in (1, 2, 3, 4):
                if action == original:
                    continue
                record = _run_probe(
                    game=game,
                    reference=reference,
                    semantic_index=semantic_index,
                    prefix_actions=prefix,
                    probe_step=probe_step,
                    probe_action=action,
                )
                record["original_action"] = original
                records.append(record)
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
                stream.write("\n")
                if (
                    found is None
                    and record["drift_type"] == "WORLD_DRIFT"
                    and record["semantic_features_changed"]
                    and record["levels_after"] == record["levels_before"]
                ):
                    found = record

    return {
        "milestone": "M3_5_LEGAL_ACTION_LIVE_SEMANTIC_DRIFT_SCOUT",
        "status": "available" if found is not None else "skipped_no_natural_live_semantic_drift_found",
        "mode": "public_legal_action_substitution_scout",
        "trace": str(trace_path),
        "event_artifact": str(output_events),
        "probes": len(records),
        "found": found,
        "reason": None if found is not None else (
            "No scanned first-segment legal-action substitution produced a clean "
            "non-terminal WORLD_DRIFT classification against the trace corridor."
        ),
        "known_limits": {
            "first_segment_only": True,
            "legal_actions_only": True,
            "no_private_environment_mutation": True,
            "scout_not_milestone_completion": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--max-probe-step", type=int, default=13)
    args = parser.parse_args()
    summary = evaluate(
        game=args.game,
        trace_path=args.trace,
        output_events=args.events,
        max_probe_step=args.max_probe_step,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
