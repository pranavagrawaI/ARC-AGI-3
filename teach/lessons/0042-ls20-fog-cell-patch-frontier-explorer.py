"""0042 - Fog-aware cell-patch frontier explorer for ls20.

Run from the repo root:
    uv run python teach/lessons/0042-ls20-fog-cell-patch-frontier-explorer.py --level 7 --steps 120

This is the first partial-observability layer. It does not try to solve ls20.
It maintains a persistent 5x5 cell-patch map from rendered observations, treats
unseen fog cells as unknown, learns the controlled patch/action model from
temporal falsification, and chooses frontier moves that reveal new cells.

Private game state is used only to select the test level and score coverage.
The explorer sees only rendered 64x64 frames.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import sys as _sys
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import arc_agi
import numpy as np
from arc_agi.base import OperationMode
from arcengine import GameAction


REPO = Path(__file__).resolve().parents[2]
FOG_COLOR = 5
MOVE_ACTIONS = (1, 2, 3, 4)


def _load_0041():
    path = REPO / "teach/lessons/0041-ls20-cross-level-cell-patch-verifier.py"
    spec = _ilu.spec_from_file_location("fcpfe42_verify41", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


verify41 = _load_0041()
falsifier40 = verify41.falsifier40
diag39 = verify41.diag39


@dataclass
class MapCell:
    cell: tuple[int, int]
    signature: tuple[tuple[int, ...], ...]
    colors: tuple[tuple[int, int], ...]
    first_seen: int
    last_seen: int
    seen_count: int = 1
    object_like: bool = False


@dataclass
class ExplorerState:
    known: dict[tuple[int, int], MapCell] = field(default_factory=dict)
    blocked_edges: set[tuple[tuple[int, int], int]] = field(default_factory=set)
    action_deltas: dict[int, tuple[int, int]] = field(default_factory=dict)
    controlled_signature: tuple[tuple[int, ...], ...] | None = None
    controlled_cell: tuple[int, int] | None = None
    tried_edges: set[tuple[tuple[int, int], int]] = field(default_factory=set)
    log: list[str] = field(default_factory=list)


def _rendered_grid(game) -> np.ndarray:
    return np.asarray(game.camera.render(game.current_level._sprites), dtype=np.int8)


def _set_level(env, level_index: int) -> np.ndarray:
    env.reset()
    env._game.set_level(level_index)
    return _rendered_grid(env._game)


def _patch_is_fog(patch: np.ndarray) -> bool:
    return bool(np.all(patch == FOG_COLOR))


def _visible_patches(grid: np.ndarray) -> list:
    patches = []
    for patch in diag39._cell_patches(grid):
        raw = np.array(patch.signature, dtype=np.int8)
        if _patch_is_fog(raw):
            continue
        patches.append(patch)
    return patches


def _is_object_like(patch) -> bool:
    return patch.distinct_colors >= 2 and patch.non_background > 0


def _signature_locations_visible(
    grid: np.ndarray,
    signature: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (patch.row, patch.col)
        for patch in _visible_patches(grid)
        if patch.signature == signature
    )


def _changed_visible_cells(grid0: np.ndarray, grid1: np.ndarray) -> tuple[tuple[int, int], ...]:
    cells = set()
    by0 = {(patch.row, patch.col): patch for patch in diag39._cell_patches(grid0)}
    by1 = {(patch.row, patch.col): patch for patch in diag39._cell_patches(grid1)}
    for cell in by0.keys() | by1.keys():
        p0 = np.array(by0[cell].signature, dtype=np.int8)
        p1 = np.array(by1[cell].signature, dtype=np.int8)
        if _patch_is_fog(p0) and _patch_is_fog(p1):
            continue
        if np.any(p0 != p1):
            cells.add(cell)
    return tuple(sorted(cells))


def _candidate_score(patch) -> tuple[int, int, int]:
    return (patch.distinct_colors, patch.non_background, -(patch.row + patch.col))


def _learn_initial_control(level_index: int) -> tuple[
    tuple[tuple[int, ...], ...] | None,
    dict[int, tuple[int, int]],
    set[int],
]:
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make("ls20")
    grid0 = _set_level(env, level_index)
    candidates = [
        patch for patch in _visible_patches(grid0)
        if _is_object_like(patch)
    ]
    candidates.sort(key=_candidate_score, reverse=True)
    evidence: dict[tuple[tuple[int, ...], ...], dict[int, tuple[int, int]]] = {}
    blocked = set()

    for action in MOVE_ACTIONS:
        before = _set_level(env, level_index)
        after = diag39._grid(env.step(GameAction.from_id(action)))
        changed = set(_changed_visible_cells(before, after))
        if not changed:
            blocked.add(action)
        for patch in candidates:
            locs_after = _signature_locations_visible(after, patch.signature)
            relocated = [cell for cell in locs_after if cell != (patch.row, patch.col)]
            if (patch.row, patch.col) in changed and len(relocated) == 1:
                new_cell = relocated[0]
                evidence.setdefault(patch.signature, {})[action] = (
                    new_cell[0] - patch.row,
                    new_cell[1] - patch.col,
                )

    if not evidence:
        return None, {}, blocked
    signature, deltas = max(evidence.items(), key=lambda item: len(item[1]))
    if len(deltas) < 2:
        return None, {}, blocked
    return signature, deltas, blocked


def _update_map(state: ExplorerState, grid: np.ndarray, step: int) -> int:
    new_cells = 0
    for patch in _visible_patches(grid):
        cell = (patch.row, patch.col)
        existing = state.known.get(cell)
        object_like = _is_object_like(patch)
        if existing is None:
            state.known[cell] = MapCell(
                cell=cell,
                signature=patch.signature,
                colors=patch.colors,
                first_seen=step,
                last_seen=step,
                object_like=object_like,
            )
            new_cells += 1
        else:
            existing.signature = patch.signature
            existing.colors = patch.colors
            existing.last_seen = step
            existing.seen_count += 1
            existing.object_like = existing.object_like or object_like
        if patch.signature == state.controlled_signature:
            state.controlled_cell = cell
    return new_cells


def _neighbors(
    cell: tuple[int, int],
    action_deltas: dict[int, tuple[int, int]],
    *,
    actions: tuple[int, ...] | None = None,
):
    action_iter = actions if actions is not None else tuple(action_deltas)
    for action in action_iter:
        if action not in action_deltas:
            continue
        dr, dc = action_deltas[action]
        yield action, (cell[0] + dr, cell[1] + dc)


def _in_lattice(cell: tuple[int, int]) -> bool:
    row, col = cell
    return 0 <= row <= 55 and 4 <= col <= 59 and row % 5 == 0 and (col - 4) % 5 == 0


def _frontier_score(state: ExplorerState, cell: tuple[int, int]) -> int:
    score = 0
    for _action, nxt in _neighbors(cell, state.action_deltas):
        if _in_lattice(nxt) and nxt not in state.known:
            score += 1
    return score


def _path_to_frontier(state: ExplorerState) -> tuple[int, ...] | None:
    start = state.controlled_cell
    if start is None:
        return None
    q = deque([(start, ())])
    seen = {start}
    best_path = None
    best_score = 0
    while q:
        cell, path = q.popleft()
        score = _frontier_score(state, cell)
        if score > best_score:
            best_score = score
            best_path = path
            if path:
                return path
        for action, nxt in _neighbors(cell, state.action_deltas):
            if not _in_lattice(nxt) or nxt in seen:
                continue
            if nxt not in state.known:
                continue
            if (cell, action) in state.blocked_edges:
                continue
            seen.add(nxt)
            q.append((nxt, path + (action,)))
    return best_path


def _choose_explore_action(state: ExplorerState) -> int:
    if state.controlled_cell is not None:
        for action in MOVE_ACTIONS:
            edge = (state.controlled_cell, action)
            if edge not in state.tried_edges and edge not in state.blocked_edges:
                return action
    path = _path_to_frontier(state)
    if path:
        return path[0]
    if state.controlled_cell is not None:
        for action, nxt in _neighbors(state.controlled_cell, state.action_deltas):
            if _in_lattice(nxt) and (state.controlled_cell, action) not in state.blocked_edges:
                return action
    for action in MOVE_ACTIONS:
        edge = (state.controlled_cell, action) if state.controlled_cell is not None else None
        if edge is None or edge not in state.blocked_edges:
            return action
    return MOVE_ACTIONS[0]


def _oracle_visible_cells_seen(game, state: ExplorerState) -> tuple[int, int, list[str]]:
    oracles = verify41._sprite_oracles(game)
    hits = []
    visible = []
    for oracle in oracles:
        if not oracle.visible:
            continue
        visible.append(oracle)
        if oracle.cell in state.known:
            hits.append(oracle)
    labels = [oracle.label for oracle in hits]
    return len(hits), len(visible), labels


def run_explorer(level: int, steps: int) -> dict[str, object]:
    level_index = level - 1
    controlled_sig, action_deltas, blocked_actions = _learn_initial_control(level_index)
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make("ls20")
    grid = _set_level(env, level_index)
    state = ExplorerState(
        controlled_signature=controlled_sig,
        action_deltas=action_deltas,
    )
    _update_map(state, grid, 0)
    for action in blocked_actions:
        if state.controlled_cell is not None:
            state.blocked_edges.add((state.controlled_cell, action))
    state.log.append(
        f"init known={len(state.known)} controlled_cell={state.controlled_cell} "
        f"actions={state.action_deltas}"
    )
    final_state = "NOT_FINISHED"
    levels_completed = 0

    for step in range(1, steps + 1):
        if state.controlled_cell is None or not state.action_deltas:
            state.log.append(f"{step}: blocked_no_control_model")
            break
        prev_cell = state.controlled_cell
        action = _choose_explore_action(state)
        state.tried_edges.add((prev_cell, action))
        frame = env.step(GameAction.from_id(action))
        grid_next = diag39._grid(frame)
        new_cells = _update_map(state, grid_next, step)
        moved = state.controlled_cell != prev_cell
        if moved and state.controlled_cell is not None and action not in state.action_deltas:
            state.action_deltas[action] = (
                state.controlled_cell[0] - prev_cell[0],
                state.controlled_cell[1] - prev_cell[1],
            )
        if not moved:
            state.blocked_edges.add((prev_cell, action))
        state.log.append(
            f"{step}:A{action}:moved={moved}:new={new_cells}:known={len(state.known)}:"
            f"cell={state.controlled_cell}:state={frame.state.value}:level_done={frame.levels_completed}"
        )
        final_state = frame.state.value
        levels_completed = frame.levels_completed
        if frame.state.value in {"WIN", "GAME_OVER"} or frame.levels_completed >= level:
            break

    oracle_hits, oracle_total, oracle_labels = _oracle_visible_cells_seen(env._game, state)
    object_like = [cell for cell, item in state.known.items() if item.object_like]
    return {
        "level": level,
        "steps": len([line for line in state.log if ":A" in line]),
        "known_cells": len(state.known),
        "object_like_cells": len(object_like),
        "controlled_cell": state.controlled_cell,
        "final_state": final_state,
        "levels_completed": levels_completed,
        "action_deltas": state.action_deltas,
        "blocked_edges": len(state.blocked_edges),
        "oracle_hits": oracle_hits,
        "oracle_total": oracle_total,
        "oracle_labels": oracle_labels,
        "log": state.log,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=7)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()

    result = run_explorer(args.level, args.steps)
    print("ls20 fog-aware cell-patch frontier explorer")
    print(
        f"level={result['level']} steps={result['steps']} known_cells={result['known_cells']} "
        f"object_like_cells={result['object_like_cells']} controlled_cell={result['controlled_cell']} "
        f"state={result['final_state']} levels_completed={result['levels_completed']}"
    )
    print(f"action_deltas={result['action_deltas']} blocked_edges={result['blocked_edges']}")
    print(
        f"oracle_sprite_cells_seen={result['oracle_hits']}/{result['oracle_total']} "
        f"labels={result['oracle_labels']}"
    )
    print("log:")
    for line in result["log"][:80]:
        print(f"  {line}")


if __name__ == "__main__":
    main()
