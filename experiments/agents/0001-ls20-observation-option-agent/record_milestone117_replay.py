"""Record a replayable public trace from the M11.7 action sequence.

This utility is for visualization. It reconstructs the public action sequence
emitted by the M11.7 evaluator and writes a harness JSONL trace that the replay
viewer can open. It is not new evidence that the agent has learned a stronger
policy than the original evaluator demonstrated.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
AGENT_ROOT = Path(__file__).resolve().parent
for path in (REPO, AGENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import arc_agi
from arc_agi.base import OperationMode

from experiments.harness import Decision, JsonlTraceWriter, Observation, Transition, run_episode


ARTIFACT_DIR = REPO / "experiments/artifacts/0001-ls20-observation-option-agent"
DEFAULT_EVENTS = ARTIFACT_DIR / "milestone-117-events.jsonl"
DEFAULT_TRACE = (
    AGENT_ROOT
    / "artifacts/recordings"
    / "ls20-milestone-117-public-replay.jsonl"
)


class RecordedActionAgent:
    """Replay a known public action sequence through the harness boundary."""

    def __init__(
        self,
        actions: tuple[int, ...],
        *,
        source_events: Path,
        stop_after_actions: int | None,
    ) -> None:
        self._actions = actions
        self._source_events = source_events
        self._stop_after_actions = stop_after_actions
        self._index = 0

    def act(self, observation: Observation) -> Decision:
        if self._index >= len(self._actions):
            raise RuntimeError("recorded action sequence exhausted before run stopped")
        action = self._actions[self._index]
        self._index += 1
        return Decision(
            action=action,
            reasoning={
                "source": "reconstructed_milestone_117_public_action_sequence",
                "source_events": str(self._source_events),
                "sequence_index": self._index,
                "stop_after_actions": self._stop_after_actions,
                "visualization_trace_only": True,
            },
        )

    def observe(self, transition: Transition) -> None:
        return None


def extract_executed_actions(
    events_path: Path,
    *,
    stop_after_actions: int | None = None,
) -> tuple[int, ...]:
    """Extract a de-duplicated public action script from milestone events."""

    step_actions: dict[int, int] = {}
    with events_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{events_path}:{line_number}: invalid JSONL") from error
            for record in _iter_action_records(event):
                if not record.get("executed"):
                    continue
                step = int(record["step"])
                action = int(record["action"])
                prior = step_actions.get(step)
                if prior is not None and prior != action:
                    raise ValueError(
                        f"{events_path}:{line_number}: conflicting actions for step {step}: "
                        f"{prior} and {action}"
                    )
                step_actions[step] = action

    if not step_actions:
        raise ValueError(f"{events_path} contains no executed public actions")

    actions = tuple(action for step, action in sorted(step_actions.items()))
    if stop_after_actions is not None:
        if stop_after_actions <= 0:
            raise ValueError("stop_after_actions must be positive")
        actions = actions[:stop_after_actions]
    return actions


def _iter_action_records(value: Any):
    if isinstance(value, dict):
        if {"step", "action", "executed"}.issubset(value.keys()):
            yield value
        for item in value.values():
            yield from _iter_action_records(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_action_records(item)


def record_replay(
    *,
    game: str,
    events_path: Path,
    trace_path: Path,
    stop_after_actions: int | None,
) -> dict[str, Any]:
    actions = extract_executed_actions(
        events_path,
        stop_after_actions=stop_after_actions,
    )
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make(game)
    agent = RecordedActionAgent(
        actions,
        source_events=events_path,
        stop_after_actions=stop_after_actions,
    )
    with JsonlTraceWriter(trace_path) as trace:
        result = run_episode(env, agent, max_actions=len(actions), trace=trace)

    action_counts = Counter(transition.decision.action for transition in result.transitions)
    return {
        "agent": "0001-ls20-observation-option-agent",
        "game": game,
        "source": "milestone_117_public_action_sequence",
        "source_events": str(events_path),
        "trace": str(trace_path),
        "actions_requested": len(actions),
        "actions_recorded": result.actions,
        "action_counts": dict(sorted(action_counts.items())),
        "levels_completed": result.final.levels_completed,
        "state": result.final.state,
        "stop_reason": result.stop_reason,
        "visualization_trace_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument(
        "--stop-after-actions",
        type=int,
        help="truncate the replay, e.g. 21 to stop at the first public level transition",
    )
    args = parser.parse_args()

    summary = record_replay(
        game=args.game,
        events_path=args.events,
        trace_path=args.trace,
        stop_after_actions=args.stop_after_actions,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
