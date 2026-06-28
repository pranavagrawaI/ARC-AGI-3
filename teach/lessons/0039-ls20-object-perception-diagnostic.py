"""0039 - Diagnose ls20 object perception.

Run from the repo root:
    uv run python teach/lessons/0039-ls20-object-perception-diagnostic.py

This lesson diagnoses perception only. It does not solve ls20 and does not give
the agent private sprite labels. The private offline game state is used only as
an evaluation oracle so we can ask whether a frame-only representation recovers
the object candidates a human would notice before knowing their roles.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import sys as _sys
from dataclasses import dataclass
from pathlib import Path

import arc_agi
import numpy as np
from arc_agi.base import OperationMode
from arcengine import GameAction


REPO = Path(__file__).resolve().parents[2]
CELL = 5
ROW_ORIGIN = 0
COL_ORIGIN = 4


def _load_0010():
    path = REPO / "teach/lessons/0010-invariance-lattice-perception.py"
    spec = _ilu.spec_from_file_location("lopd39_il0010", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


il0010 = _load_0010()


@dataclass(frozen=True)
class OracleObject:
    name: str
    row: int
    col: int
    height: int
    width: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.row, self.col, self.row + self.height - 1, self.col + self.width - 1)

    @property
    def cell(self) -> tuple[int, int]:
        return _cell_top_left(self.row, self.col)


@dataclass(frozen=True)
class CellPatch:
    row: int
    col: int
    signature: tuple[tuple[int, ...], ...]
    colors: tuple[tuple[int, int], ...]
    non_background: int
    distinct_colors: int

    @property
    def objectness(self) -> tuple[int, int, int]:
        return (self.distinct_colors, self.non_background, -self.row)


def _grid(frame_data) -> np.ndarray:
    arr = np.asarray(frame_data.frame, dtype=np.int8)
    return arr[-1] if arr.ndim == 3 else arr


def _background(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _cell_top_left(row: int, col: int) -> tuple[int, int]:
    r = row - ((row - ROW_ORIGIN) % CELL)
    c = col - ((col - COL_ORIGIN) % CELL)
    return (r, c)


def _patch(grid: np.ndarray, row: int, col: int) -> np.ndarray:
    return grid[row : row + CELL, col : col + CELL]


def _patch_signature(patch: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in patch.tolist())


def _patch_summary(patch: np.ndarray) -> str:
    return "/".join("".join(f"{int(v):02d}" for v in row) for row in patch.tolist())


def _cell_patches(grid: np.ndarray) -> list[CellPatch]:
    bg = _background(grid)
    patches: list[CellPatch] = []
    height, width = grid.shape
    for row in range(ROW_ORIGIN, height - CELL + 1, CELL):
        for col in range(COL_ORIGIN, width - CELL + 1, CELL):
            patch = _patch(grid, row, col)
            vals, counts = np.unique(patch, return_counts=True)
            colors = tuple(sorted((int(v), int(c)) for v, c in zip(vals, counts)))
            non_background = int(patch.size - np.sum(patch == bg))
            patches.append(
                CellPatch(
                    row=row,
                    col=col,
                    signature=_patch_signature(patch),
                    colors=colors,
                    non_background=non_background,
                    distinct_colors=len(colors),
                )
            )
    return patches


def _oracle_objects(game) -> list[OracleObject]:
    objects = [
        OracleObject(
            "controlled_candidate_orange_blue",
            game.gudziatsk.y,
            game.gudziatsk.x,
            game.gudziatsk.height,
            game.gudziatsk.width,
        )
    ]
    for sprite in game.current_level.get_sprites_by_tag("rhsxkxzdjz"):
        objects.append(
            OracleObject("white_cross_candidate", sprite.y, sprite.x, sprite.height, sprite.width)
        )
    markers = []
    for tag in ("rjlbuycveu", "kvynsvxbpi"):
        markers.extend(game.current_level.get_sprites_by_tag(tag))
    if markers:
        r0 = min(sprite.y for sprite in markers)
        c0 = min(sprite.x for sprite in markers)
        r1 = max(sprite.y + sprite.height - 1 for sprite in markers)
        c1 = max(sprite.x + sprite.width - 1 for sprite in markers)
        objects.append(OracleObject("top_marked_candidate", r0, c0, r1 - r0 + 1, c1 - c0 + 1))
    return objects


def _view_hits_object(view, obj: OracleObject) -> int:
    r0, c0, r1, c1 = obj.bbox
    return sum(1 for row, col in view.pixels if r0 <= row <= r1 and c0 <= col <= c1)


def _component_report(grid: np.ndarray, objects: list[OracleObject]) -> list[str]:
    _bg, views = il0010.views_from_grid(grid, frame=0)
    lines = [f"blob_components total={len(views)}"]
    largest = sorted(views, key=lambda view: view.area, reverse=True)[:8]
    for view in largest:
        lines.append(
            "  blob "
            f"area={view.area:4d} bbox={view.bbox} colors={view.color_lattice['raw']}"
        )
    lines.append("blob coverage of salient first-frame candidates:")
    for obj in objects:
        hits = sorted(
            ((view.area, _view_hits_object(view, obj), view.bbox, view.color_lattice["raw"]) for view in views),
            key=lambda item: (-item[1], -item[0]),
        )
        covered = [item for item in hits if item[1] > 0][:4]
        lines.append(f"  {obj.name} bbox={obj.bbox}:")
        for area, hit, bbox, colors in covered:
            lines.append(f"    covered_by_blob hit_pixels={hit:2d} blob_area={area:4d} bbox={bbox} colors={colors}")
    return lines


def _cell_report(grid: np.ndarray, objects: list[OracleObject]) -> list[str]:
    patches = _cell_patches(grid)
    by_cell = {(patch.row, patch.col): patch for patch in patches}
    lines = [f"cell_patches lattice_origin=(row={ROW_ORIGIN}, col={COL_ORIGIN}) cell={CELL} total={len(patches)}"]
    lines.append("cell patches at salient first-frame candidates:")
    for obj in objects:
        patch = by_cell[obj.cell]
        lines.append(
            f"  {obj.name} object_bbox={obj.bbox} cell={obj.cell} "
            f"colors={patch.colors} signature={_patch_summary(np.array(patch.signature))}"
        )
    distinctive = sorted(
        [patch for patch in patches if patch.distinct_colors >= 2 and patch.non_background > 0],
        key=lambda patch: patch.objectness,
        reverse=True,
    )[:12]
    lines.append("most distinctive non-background cells:")
    for patch in distinctive:
        lines.append(
            f"  cell={(patch.row, patch.col)} colors={patch.colors} "
            f"non_bg={patch.non_background} signature={_patch_summary(np.array(patch.signature))}"
        )
    return lines


def _find_signature(grid: np.ndarray, signature: tuple[tuple[int, ...], ...]) -> list[tuple[int, int]]:
    return [
        (patch.row, patch.col)
        for patch in _cell_patches(grid)
        if patch.signature == signature
    ]


def _single_action_falsification(action: int) -> list[str]:
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make("ls20")
    frame0 = env.reset()
    game = env._game
    grid0 = _grid(frame0)
    objects = _oracle_objects(game)
    signatures = {
        obj.name: _patch_signature(_patch(grid0, *obj.cell))
        for obj in objects
    }
    frame1 = env.step(GameAction.from_id(action))
    grid1 = _grid(frame1)
    lines = [f"single_action_probe action=A{action}"]
    for obj in objects:
        before = _patch(grid0, *obj.cell)
        after_same_cell = _patch(grid1, *obj.cell)
        changed_at_original_cell = bool(np.any(before != after_same_cell))
        locations = _find_signature(grid1, signatures[obj.name])
        lines.append(
            f"  {obj.name}: original_cell={obj.cell} changed={changed_at_original_cell} "
            f"same_signature_locations_after={locations[:6]}"
        )
    changed_cells = []
    for patch in _cell_patches(grid0):
        if np.any(_patch(grid0, patch.row, patch.col) != _patch(grid1, patch.row, patch.col)):
            changed_cells.append((patch.row, patch.col))
    lines.append(f"  changed_lattice_cells={changed_cells[:20]} total={len(changed_cells)}")
    return lines


def run_diagnostic(action: int) -> list[str]:
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make("ls20")
    frame = env.reset()
    game = env._game
    grid = _grid(frame)
    objects = _oracle_objects(game)
    lines = [
        "ls20 perception diagnosis",
        f"frame_shape={grid.shape} background={_background(grid)} unique_colors={sorted(int(v) for v in np.unique(grid))}",
        "",
        "Hypothesis 1: connected components are too coarse for object candidates.",
    ]
    lines.extend(_component_report(grid, objects))
    lines.extend(
        [
            "",
            "Hypothesis 2: 5x5 cell patches preserve the salient object candidates.",
        ]
    )
    lines.extend(_cell_report(grid, objects))
    lines.extend(
        [
            "",
            "Hypothesis 3: one action falsifies which candidate is controllable.",
        ]
    )
    lines.extend(_single_action_falsification(action))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=int, default=1, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    for line in run_diagnostic(args.action):
        print(line)


if __name__ == "__main__":
    main()
