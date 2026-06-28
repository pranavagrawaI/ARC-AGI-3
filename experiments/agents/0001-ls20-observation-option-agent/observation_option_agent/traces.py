"""Read admissible observation-action traces without retaining private fields."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.harness import Observation


@dataclass(frozen=True)
class TraceStep:
    before: Observation
    action: int
    after: Observation


def _observation(data: dict[str, object]) -> Observation:
    pixels = np.array(data["frame"] if "frame" in data else data["pixels"], copy=True)
    pixels.setflags(write=False)
    return Observation(
        pixels=pixels,
        available_actions=tuple(int(action) for action in data["available_actions"]),
        levels_completed=int(data["levels_completed"]),
        state=str(data["state"]),
        full_reset=bool(data["full_reset"]),
    )


def _action_number(action_id: object) -> int | None:
    if not isinstance(action_id, str) or not action_id.startswith("ACTION"):
        return None
    return int(action_id.removeprefix("ACTION"))


def iter_trace_steps(path: Path):
    """Yield aligned transitions from wrapper recordings or harness traces."""

    previous: Observation | None = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if "data" in record:
                data = record["data"]
                current = _observation(data)
                action = _action_number((data.get("action_input") or {}).get("id"))
                if previous is not None and action is not None:
                    yield TraceStep(previous, action, current)
                previous = current
                continue

            record_type = record.get("type")
            if record_type == "observation":
                previous = _observation(record)
            elif record_type == "transition":
                current = _observation(record["observation"])
                if previous is None:
                    raise ValueError("transition encountered before initial observation")
                yield TraceStep(previous, int(record["action"]), current)
                previous = current
