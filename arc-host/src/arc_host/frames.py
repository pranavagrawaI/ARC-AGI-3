"""Project raw ARC frames onto the public, settled observation."""

from __future__ import annotations

from typing import Any

import numpy as np

from arc_host.actions import action_name
from arc_host.types import (
    EvidenceEnvelope,
    EvidenceFrame,
    FrameSelectionProvenance,
    Grid,
)


def public_observation(
    frame: Any, *, previous_levels_completed: int | None = None
) -> EvidenceEnvelope:
    """Classify and copy every public frame without changing action semantics."""
    frames = np.asarray(frame.frame)
    if frames.ndim == 2:
        sequence = [frames]
    elif frames.ndim == 3 and frames.shape[0] > 0:
        sequence = list(frames)
    else:
        raise ValueError(
            f"public frame must be a grid or frame sequence, got {frames.shape}"
        )
    grids = tuple(_public_grid(grid) for grid in sequence)
    state = frame.state.value if hasattr(frame.state, "value") else str(frame.state)
    levels_completed = int(frame.levels_completed)
    level_advanced = (
        previous_levels_completed is not None
        and levels_completed > previous_levels_completed
    )
    terminal_state = state.upper().replace("GAMESTATE.", "") in {"WIN", "GAME_OVER"}
    terminal_index = 0 if level_advanced or terminal_state else None
    terminal_reason = (
        "level_advanced"
        if level_advanced
        else "terminal_state"
        if terminal_state
        else None
    )
    decision_index = len(grids) - 1
    event_indices = tuple(
        index
        for index in range(len(grids))
        if index not in {decision_index, terminal_index}
    )
    available_actions = [action_name(action) for action in frame.available_actions]
    available_actions = [
        "RESET",
        *(action for action in available_actions if action != "RESET"),
    ]
    return EvidenceEnvelope(
        decision_frame=grids[decision_index],
        terminal_frame=grids[terminal_index] if terminal_index is not None else None,
        event_frames=tuple(
            EvidenceFrame(kind="animation", source_index=index, frame=grids[index])
            for index in event_indices
        ),
        provenance=FrameSelectionProvenance(
            raw_frame_count=len(grids),
            decision_frame_index=decision_index,
            terminal_frame_index=terminal_index,
            event_frame_indices=event_indices,
            terminal_reason=terminal_reason,
        ),
        available_actions=tuple(available_actions),
        levels_completed=levels_completed,
        state=state,
        full_reset=bool(frame.full_reset),
    )


def _public_grid(frame: np.ndarray[Any, Any]) -> Grid:
    if frame.ndim != 2 or 0 in frame.shape:
        raise ValueError(f"public frame must be a non-empty grid, got {frame.shape}")
    if frame.shape[0] > 64 or frame.shape[1] > 64:
        raise ValueError(f"public grid exceeds 64x64: {frame.shape}")
    if np.any((frame < 0) | (frame > 15)):
        raise ValueError("public grid cells must be in [0, 15]")
    return tuple(tuple(int(cell) for cell in row) for row in frame)
