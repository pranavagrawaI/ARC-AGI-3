"""0041 - Verify cell-patch candidates across one frame per ls20 level.

Run from the repo root:
    uv run python teach/lessons/0041-ls20-cross-level-cell-patch-verifier.py

0040 verified cell-patch temporal falsification on level 1. This lesson checks
the frame-only half of that representation on one initial frame from every ls20
level. The private game state is used only as an evaluation oracle:

  - perception input: a 64x64 rendered observation frame,
  - perception output: top 5x5 cell-patch candidates,
  - oracle scoring: whether known salient sprites occupy detected candidate cells.

No role labels are fed into perception.
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


REPO = Path(__file__).resolve().parents[2]


def _load_0040():
    path = REPO / "teach/lessons/0040-ls20-cell-patch-temporal-falsifier.py"
    spec = _ilu.spec_from_file_location("clcpv41_falsifier40", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


falsifier40 = _load_0040()
diag39 = falsifier40.diag39


EVAL_TAGS = (
    "rhsxkxzdjz",  # rotation changer
    "soyhouuebz",  # colour changer
    "ttfwljgohq",  # shape changer
    "npxgalaybz",  # step pickup/reset-like object
    "rjlbuycveu",  # target/goal location
    "kvynsvxbpi",  # target marker shown with the goal
)


@dataclass(frozen=True)
class SpriteOracle:
    label: str
    cell: tuple[int, int]
    bbox: tuple[int, int, int, int]
    visible: bool


def _is_visible(sprite) -> bool:
    return bool(getattr(sprite, "is_visible", getattr(sprite, "visible", True)))


def _sprite_oracles(game) -> list[SpriteOracle]:
    out = [
        SpriteOracle(
            "controlled_candidate",
            diag39._cell_top_left(game.gudziatsk.y, game.gudziatsk.x),
            (
                game.gudziatsk.y,
                game.gudziatsk.x,
                game.gudziatsk.y + game.gudziatsk.height - 1,
                game.gudziatsk.x + game.gudziatsk.width - 1,
            ),
            _is_visible(game.gudziatsk),
        )
    ]
    for tag in EVAL_TAGS:
        for index, sprite in enumerate(game.current_level.get_sprites_by_tag(tag)):
            out.append(
                SpriteOracle(
                    f"{tag}[{index}]",
                    diag39._cell_top_left(sprite.y, sprite.x),
                    (
                        sprite.y,
                        sprite.x,
                        sprite.y + sprite.height - 1,
                        sprite.x + sprite.width - 1,
                    ),
                    _is_visible(sprite),
                )
            )
    return out


def _candidate_cells(grid: np.ndarray, limit: int) -> dict[tuple[int, int], falsifier40.Candidate]:
    return {
        candidate.cell: candidate
        for candidate in falsifier40.initial_candidates(grid, limit)
    }


def _level_grid(game, level_index: int) -> np.ndarray:
    game.set_level(level_index)
    # Use the same rendered observation surface as the wrapper. `get_pixels`
    # returns raw sprite pixels and bypasses level-7 fog.
    return np.asarray(game.camera.render(game.current_level._sprites), dtype=np.int8)


def verify_levels(limit: int) -> list[str]:
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make("ls20")
    env.reset()
    game = env._game
    n_levels = len(game._levels)
    lines = [
        "ls20 cross-level cell-patch verifier",
        f"levels={n_levels} candidate_limit={limit}",
        "",
    ]
    total_visible = 0
    total_hit = 0
    by_label_kind: dict[str, list[bool]] = {}

    for level_index in range(n_levels):
        grid = _level_grid(game, level_index)
        candidates = _candidate_cells(grid, limit)
        oracles = _sprite_oracles(game)
        visible_oracles = [oracle for oracle in oracles if oracle.visible]
        hits = [oracle for oracle in visible_oracles if oracle.cell in candidates]
        total_visible += len(visible_oracles)
        total_hit += len(hits)

        lines.append(
            f"level {level_index + 1}: visible_oracles={len(visible_oracles)} "
            f"candidate_hits={len(hits)} recall={len(hits)}/{len(visible_oracles)}"
        )
        for oracle in visible_oracles:
            ok = oracle.cell in candidates
            kind = oracle.label.split("[", 1)[0]
            by_label_kind.setdefault(kind, []).append(ok)
            status = "hit" if ok else "miss"
            extra = ""
            if ok:
                candidate = candidates[oracle.cell]
                extra = f" colors={candidate.colors} non_bg={candidate.non_background}"
            lines.append(
                f"  {status:4s} {oracle.label:24s} cell={oracle.cell} bbox={oracle.bbox}{extra}"
            )
        lines.append("")

    lines.append(f"overall_visible_recall={total_hit}/{total_visible}")
    lines.append("recall_by_oracle_kind:")
    for kind, oks in sorted(by_label_kind.items()):
        lines.append(f"  {kind}: {sum(oks)}/{len(oks)}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()

    for line in verify_levels(args.limit):
        print(line)


if __name__ == "__main__":
    main()
