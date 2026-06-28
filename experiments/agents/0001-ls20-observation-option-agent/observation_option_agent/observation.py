"""Neutral visual decomposition with no game-specific role vocabulary."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib

import numpy as np

from experiments.harness import Observation


Pixel = tuple[int, int]


@dataclass(frozen=True)
class BoundingBox:
    top: int
    left: int
    bottom: int
    right: int

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1

    @property
    def width(self) -> int:
        return self.right - self.left + 1


@dataclass(frozen=True)
class Detection:
    frame_index: int
    detection_index: int
    value: int
    pixels: frozenset[Pixel]
    bbox: BoundingBox
    centroid: tuple[float, float]
    shape: tuple[Pixel, ...]

    @property
    def area(self) -> int:
        return len(self.pixels)

    @property
    def shape_id(self) -> str:
        return hashlib.sha1(repr(self.shape).encode("utf-8")).hexdigest()[:12]

    @classmethod
    def from_pixels(
        cls,
        *,
        frame_index: int,
        detection_index: int,
        value: int,
        pixels: set[Pixel],
    ) -> Detection:
        rows = [pixel[0] for pixel in pixels]
        cols = [pixel[1] for pixel in pixels]
        top, left = min(rows), min(cols)
        bbox = BoundingBox(top, left, max(rows), max(cols))
        shape = tuple(sorted((row - top, col - left) for row, col in pixels))
        return cls(
            frame_index=frame_index,
            detection_index=detection_index,
            value=value,
            pixels=frozenset(pixels),
            bbox=bbox,
            centroid=(sum(rows) / len(rows), sum(cols) / len(cols)),
            shape=shape,
        )


@dataclass(frozen=True)
class VisualScene:
    frame_index: int
    shape: tuple[int, int]
    inferred_background: int
    detections: tuple[Detection, ...]


class ObservationParser:
    """Infer a background value and decompose the rest into components.

    Same-valued four-connected regions are merely detection proposals. They carry
    no fixed semantic role; temporal evidence decides which proposals matter.
    """

    def parse(self, observation: Observation, *, frame_index: int) -> VisualScene:
        if observation.pixels.ndim < 2 or observation.pixels.size == 0:
            raise ValueError("observation has no rendered pixels")
        grid = np.asarray(observation.pixels[-1])
        values, counts = np.unique(grid, return_counts=True)
        background = int(values[int(np.argmax(counts))])
        visited = np.zeros(grid.shape, dtype=bool)
        detections: list[Detection] = []
        height, width = grid.shape

        for row in range(height):
            for col in range(width):
                if visited[row, col] or int(grid[row, col]) == background:
                    visited[row, col] = True
                    continue
                value = int(grid[row, col])
                pixels: set[Pixel] = set()
                queue = deque([(row, col)])
                visited[row, col] = True
                while queue:
                    current_row, current_col = queue.popleft()
                    pixels.add((current_row, current_col))
                    for next_row, next_col in (
                        (current_row - 1, current_col),
                        (current_row + 1, current_col),
                        (current_row, current_col - 1),
                        (current_row, current_col + 1),
                    ):
                        if not (0 <= next_row < height and 0 <= next_col < width):
                            continue
                        if visited[next_row, next_col]:
                            continue
                        if int(grid[next_row, next_col]) != value:
                            continue
                        visited[next_row, next_col] = True
                        queue.append((next_row, next_col))
                detections.append(Detection.from_pixels(
                    frame_index=frame_index,
                    detection_index=len(detections),
                    value=value,
                    pixels=pixels,
                ))

        return VisualScene(
            frame_index=frame_index,
            shape=(height, width),
            inferred_background=background,
            detections=tuple(detections),
        )
