"""Offline source-template tracker used to adjudicate the recorded contradictions."""

from __future__ import annotations

import numpy as np


AVATAR_TEMPLATE = np.asarray([[12] * 5, [12] * 5, [9] * 5, [9] * 5, [9] * 5])


def avatar_top_left(frame: object) -> tuple[int, int]:
    """Return ``(row, column)`` for the unique LS20 avatar template."""

    grid = np.asarray(frame)
    if grid.ndim == 3:
        grid = grid[-1]
    matches: list[tuple[int, int]] = []
    for row in range(grid.shape[0] - 4):
        for column in range(grid.shape[1] - 4):
            if np.array_equal(grid[row : row + 5, column : column + 5], AVATAR_TEMPLATE):
                matches.append((row, column))
    if len(matches) != 1:
        raise ValueError(f"expected one avatar template, found {len(matches)}")
    return matches[0]


def track_translation(before: object, after: object) -> dict[str, object]:
    """Track a move from the avatar template, never from changed-half ordering."""

    source = avatar_top_left(before)
    destination = avatar_top_left(after)
    return {
        "source": source,
        "destination": destination,
        "delta": (destination[0] - source[0], destination[1] - source[1]),
    }
