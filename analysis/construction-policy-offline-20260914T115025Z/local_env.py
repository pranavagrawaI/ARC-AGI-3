"""Deterministic grid world for local gameplay without the ARC service."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LocalObservation:
    grid: list[list[int]]
    available_actions: tuple[str, ...]
    levels_completed: int = 0
    state: str = "NOT_FINISHED"
    full_reset: bool = False


class LocalGridWorld:
    """Block value 12; walls value 5; open floor 4. Motion is one cell."""

    def __init__(self, rows: int = 8, cols: int = 8, block: list[int] | None = None):
        self.rows = rows
        self.cols = cols
        self.block = list(block or [4, 0])
        self.steps: list[str] = []
        self._available = ("RESET", "UP", "DOWN", "LEFT", "RIGHT")

    def _grid(self) -> list[list[int]]:
        grid = [[4] * self.cols for _ in range(self.rows)]
        for c in range(self.cols):
            grid[0][c] = 5
            grid[self.rows - 1][c] = 5
        for r in range(self.rows):
            grid[r][0] = 5
            grid[r][self.cols - 1] = 5
        grid[0][0] = 5
        r, c = self.block
        grid[r][c] = 12
        return grid

    def _observation(self) -> LocalObservation:
        return LocalObservation(grid=self._grid(), available_actions=self._available)

    def reset(self) -> LocalObservation:
        self.steps = []
        return self._observation()

    def step(self, action: str) -> LocalObservation:
        self.steps.append(action)
        delta = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}.get(action, (0, 0))
        nr, nc = self.block[0] + delta[0], self.block[1] + delta[1]
        if 0 <= nr < self.rows and 0 <= nc < self.cols and self._grid()[nr][nc] != 5:
            self.block = [nr, nc]
        if self.block == [1, self.cols - 2]:
            obs = self._observation()
            obs.state = "WIN"
            obs.levels_completed = 1
            return obs
        return self._observation()
