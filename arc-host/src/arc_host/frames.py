"""Project raw ARC frames onto the public, settled observation."""

from __future__ import annotations

from typing import Any

import numpy as np

from arc_host.actions import action_name
from arc_host.types import Observation


def public_observation(frame: Any) -> Observation:
    """Copy the settled grid and public metadata; discard no public semantics."""
    frames = np.asarray(frame.frame)
    if frames.ndim == 2:
        settled = frames
    elif frames.ndim == 3 and frames.shape[0] > 0:
        settled = frames[-1]
    else:
        raise ValueError(
            f"public frame must be a grid or frame sequence, got {frames.shape}"
        )
    if settled.ndim != 2 or 0 in settled.shape:
        raise ValueError(
            f"settled public frame must be a non-empty grid, got {settled.shape}"
        )
    if settled.shape[0] > 64 or settled.shape[1] > 64:
        raise ValueError(f"public grid exceeds 64x64: {settled.shape}")
    if np.any((settled < 0) | (settled > 15)):
        raise ValueError("public grid cells must be in [0, 15]")
    grid = tuple(tuple(int(cell) for cell in row) for row in settled)
    state = frame.state.value if hasattr(frame.state, "value") else str(frame.state)
    available_actions = [action_name(action) for action in frame.available_actions]
    available_actions = [
        "RESET",
        *(action for action in available_actions if action != "RESET"),
    ]
    return Observation(
        grid=grid,
        available_actions=tuple(available_actions),
        levels_completed=int(frame.levels_completed),
        state=state,
        full_reset=bool(frame.full_reset),
    )
