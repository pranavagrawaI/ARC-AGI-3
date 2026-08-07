"""Closed, game-general transformation inventory for acquisition probes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

Grid = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class RelationResult:
    relation: str
    matches: bool
    parameters: dict[str, object]


def grid(rows: Sequence[Sequence[int]]) -> Grid:
    value = tuple(tuple(int(cell) for cell in row) for row in rows)
    if not value or not value[0] or any(len(row) != len(value[0]) for row in value):
        raise ValueError("a grid must be non-empty and rectangular")
    return value


def rotate(value: Grid) -> Grid:
    return tuple(tuple(row) for row in zip(*value[::-1], strict=True))


def reflect(value: Grid) -> Grid:
    return tuple(tuple(reversed(row)) for row in value)


def orientations(value: Grid) -> tuple[Grid, ...]:
    rows = []
    current = value
    for _ in range(4):
        rows.extend((current, reflect(current)))
        current = rotate(current)
    return tuple(dict.fromkeys(rows))


def colour_permutation(left: Grid, right: Grid) -> bool:
    if (len(left), len(left[0])) != (len(right), len(right[0])):
        return False
    forward: dict[int, int] = {}
    reverse: dict[int, int] = {}
    for arow, brow in zip(left, right, strict=True):
        for a, b in zip(arow, brow, strict=True):
            if forward.setdefault(a, b) != b or reverse.setdefault(b, a) != a:
                return False
    return True


def nearest_scale(value: Grid, factor: int) -> Grid:
    return tuple(
        tuple(cell for cell in row for _ in range(factor))
        for row in value
        for _ in range(factor)
    )


def tiled(value: Grid, row_repeats: int, column_repeats: int) -> Grid:
    return tuple(
        tuple(cell for _ in range(column_repeats) for cell in row)
        for _ in range(row_repeats)
        for row in value
    )


def embedded(left: Grid, right: Grid) -> tuple[int, int] | None:
    for row in range(len(right) - len(left) + 1):
        for column in range(len(right[0]) - len(left[0]) + 1):
            if all(
                right[row + y][column : column + len(left[0])] == left[y]
                for y in range(len(left))
            ):
                return row, column
    return None


def algorithmic_progression(values: Sequence[int]) -> tuple[str, int] | None:
    if len(values) < 3:
        return None
    deltas = [b - a for a, b in zip(values, values[1:])]
    if len(set(deltas)) == 1:
        return "add", deltas[0]
    for modulus in range(2, 17):
        if any(value < 0 or value >= modulus for value in values):
            continue
        modular = [(b - a) % modulus for a, b in zip(values, values[1:])]
        if len(set(modular)) == 1:
            return f"advance_mod_{modulus}", modular[0]
    return None


def evaluate(left: Grid, right: Grid) -> tuple[RelationResult, ...]:
    results = [RelationResult("exact", left == right, {})]
    congruent = next((index for index, candidate in enumerate(orientations(left)) if candidate == right), None)
    results.append(RelationResult("dihedral_congruence", congruent is not None, {"orientation": congruent}))
    results.append(RelationResult("colour_permutation", colour_permutation(left, right), {}))
    scale = next((factor for factor in range(2, 6) if nearest_scale(left, factor) == right), None)
    results.append(RelationResult("integer_scale", scale is not None, {"factor": scale}))
    repeats = next(
        ((r, c) for r in range(1, 6) for c in range(1, 6) if (r, c) != (1, 1) and tiled(left, r, c) == right),
        None,
    )
    results.append(RelationResult("tiled_repetition", repeats is not None, {"repeats": repeats}))
    offset = embedded(left, right)
    results.append(RelationResult("embedded_subpattern", offset is not None, {"offset": offset}))
    lh, rh = Counter(cell for row in left for cell in row), Counter(cell for row in right for cell in row)
    ratio = len(right) * len(right[0]) / (len(left) * len(left[0]))
    proportional = set(lh) == set(rh) and all(rh[key] == lh[key] * ratio for key in lh)
    results.append(RelationResult("proportional_histogram", proportional, {"ratio": ratio}))
    return tuple(results)
