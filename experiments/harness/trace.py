"""JSONL traces containing only admissible observations and agent outputs."""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import TextIO

from .types import Observation, RunResult, Transition


def _observation_record(observation: Observation) -> dict[str, object]:
    return {
        "pixels": observation.pixels.tolist(),
        "available_actions": list(observation.available_actions),
        "levels_completed": observation.levels_completed,
        "state": observation.state,
        "full_reset": observation.full_reset,
    }


class JsonlTraceWriter:
    """Write an incrementally recoverable admissible trace."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream: TextIO = path.open("w", encoding="utf-8")

    def __enter__(self) -> JsonlTraceWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _write(self, record: dict[str, object]) -> None:
        self._stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
        self._stream.write("\n")
        self._stream.flush()

    def start(self, observation: Observation) -> None:
        self._write({"type": "observation", "step": 0, **_observation_record(observation)})

    def append(self, step: int, transition: Transition) -> None:
        self._write(
            {
                "type": "transition",
                "step": step,
                "action": transition.decision.action,
                "action_data": transition.decision.data,
                "reasoning": transition.decision.reasoning,
                "observation": _observation_record(transition.after),
            }
        )

    def append_suppressed(self, step: int, transition: Transition) -> None:
        if not transition.observation_suppressed:
            raise ValueError("suppressed trace record requires a suppressed transition")
        self._write(
            {
                "type": "suppressed_transition",
                "step": step,
                "action": transition.decision.action,
                "action_data": transition.decision.data,
                "reasoning": transition.decision.reasoning,
                "levels_completed": transition.after.levels_completed,
                "suppression_reason": "post_level_limit_observation",
            }
        )

    def finish(self, result: RunResult) -> None:
        self._write(
            {
                "type": "end",
                "actions": result.actions,
                "stop_reason": result.stop_reason,
                "level_action_counts": list(result.level_action_counts),
                "completion_action_counts": list(result.completion_action_counts),
                "reset_count": result.reset_count,
                "game_over_count": result.game_over_count,
                "levels_completed": result.final.levels_completed,
                "state": result.final.state,
            }
        )

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()
