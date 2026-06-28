"""0044 - Cross-level controllable-role memory for ls20.

Run from the repo root:
    uv run python teach/lessons/0044-ls20-cross-level-control-memory.py --steps 120

0043 relearned the controlled object and action model independently per level.
That fails when the avatar's appearance changes or only one action is initially
unblocked. This lesson carries a soft prior across levels:

  - action semantics persist across levels,
  - the controlled role persists,
  - exact avatar appearance may change.

The prior is used only as a constraint for frame-only temporal evidence: a patch
is credited as controlled on a new level only when it moves according to the
already learned action delta.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import sys as _sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import arc_agi
import numpy as np
from arc_agi.base import OperationMode
from arcengine import GameAction


REPO = Path(__file__).resolve().parents[2]


def _load_0043():
    path = REPO / "teach/lessons/0043-ls20-value-aware-object-explorer.py"
    spec = _ilu.spec_from_file_location("clcm44_value43", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


value43 = _load_0043()
frontier42 = value43.frontier42
diag39 = value43.diag39
verify41 = value43.verify41


@dataclass
class CrossLevelMemory:
    action_deltas: dict[int, tuple[int, int]] = field(default_factory=dict)
    avatar_signatures: Counter = field(default_factory=Counter)


def _patch_at(grid: np.ndarray, cell: tuple[int, int]):
    for patch in frontier42._visible_patches(grid):
        if (patch.row, patch.col) == cell:
            return patch
    return None


def _candidate_control_from_prior(
    env,
    level_index: int,
    memory: CrossLevelMemory,
) -> tuple[tuple[tuple[int, ...], ...] | None, tuple[int, int] | None, dict[int, tuple[int, int]]]:
    if not memory.action_deltas:
        return None, None, {}

    votes: Counter = Counter()
    for action, delta in memory.action_deltas.items():
        before = frontier42._set_level(env, level_index)
        before_patches = [
            patch for patch in frontier42._visible_patches(before)
            if frontier42._is_object_like(patch)
        ]
        after = diag39._grid(env.step(GameAction.from_id(action)))
        changed = set(frontier42._changed_visible_cells(before, after))
        for patch in before_patches:
            cell = (patch.row, patch.col)
            if cell not in changed:
                continue
            expected = (cell[0] + delta[0], cell[1] + delta[1])
            if expected in frontier42._signature_locations_visible(after, patch.signature):
                votes[(cell, patch.signature)] += 1

    if not votes:
        return None, None, {}
    (cell, signature), count = votes.most_common(1)[0]
    if count < 1:
        return None, None, {}
    return signature, cell, dict(memory.action_deltas)


def _bootstrap_control(
    env,
    level_index: int,
    memory: CrossLevelMemory,
) -> tuple[tuple[tuple[int, ...], ...] | None, tuple[int, int] | None, dict[int, tuple[int, int]], str]:
    signature, cell, action_deltas = _candidate_control_from_prior(env, level_index, memory)
    if signature is not None and cell is not None:
        return signature, cell, action_deltas, "prior"

    signature, action_deltas, _blocked = frontier42._learn_initial_control(level_index)
    grid = frontier42._set_level(env, level_index)
    cell = None
    if signature is not None:
        locs = frontier42._signature_locations_visible(grid, signature)
        cell = locs[0] if locs else None
    return signature, cell, action_deltas, "scratch"


def _track_control_after_action(
    state: value43.ValueExplorerState,
    grid: np.ndarray,
    prev_cell: tuple[int, int],
    action: int,
) -> None:
    if action not in state.action_deltas:
        return
    delta = state.action_deltas[action]
    expected = (prev_cell[0] + delta[0], prev_cell[1] + delta[1])
    expected_patch = _patch_at(grid, expected)
    prev_patch = _patch_at(grid, prev_cell)
    if expected_patch is not None:
        if prev_patch is None or prev_patch.signature != state.controlled_signature:
            state.controlled_cell = expected
            state.controlled_signature = expected_patch.signature
            return
    if prev_patch is not None and prev_patch.signature == state.controlled_signature:
        state.controlled_cell = prev_cell


def _oracle_label_by_cell(game) -> dict[tuple[int, int], list[str]]:
    labels: dict[tuple[int, int], list[str]] = {}
    for oracle in verify41._sprite_oracles(game):
        labels.setdefault(oracle.cell, []).append(oracle.label)
    return labels


def run_level_with_memory(
    env,
    level: int,
    steps: int,
    memory: CrossLevelMemory,
) -> dict[str, object]:
    level_index = level - 1
    controlled_sig, controlled_cell, action_deltas, bootstrap = _bootstrap_control(env, level_index, memory)
    grid = frontier42._set_level(env, level_index)
    state = value43.ValueExplorerState(
        controlled_signature=controlled_sig,
        action_deltas=dict(action_deltas),
    )
    frontier42._update_map(state, grid, 0)
    if controlled_cell is not None:
        state.controlled_cell = controlled_cell
    state.log.append(
        f"init bootstrap={bootstrap} controlled_cell={state.controlled_cell} "
        f"actions={state.action_deltas}"
    )

    final_state = "NOT_FINISHED"
    levels_completed = 0
    for step in range(1, steps + 1):
        if state.controlled_cell is None or not state.action_deltas:
            state.log.append(f"{step}:blocked_no_control_model")
            break
        prev_grid = grid
        prev_cell = state.controlled_cell
        action, reason = value43._choose_target_action(state)
        state.tried_edges.add((prev_cell, action))
        frame = env.step(GameAction.from_id(action))
        grid = diag39._grid(frame)
        changed = frontier42._changed_visible_cells(prev_grid, grid)
        new_cells = frontier42._update_map(state, grid, step)
        _track_control_after_action(state, grid, prev_cell, action)
        moved = state.controlled_cell != prev_cell
        if moved and state.controlled_cell is not None and action not in state.action_deltas:
            state.action_deltas[action] = (
                state.controlled_cell[0] - prev_cell[0],
                state.controlled_cell[1] - prev_cell[1],
            )
        if not moved:
            state.blocked_edges.add((prev_cell, action))

        entered = state.controlled_cell if moved else None
        if entered in state.known and entered not in state.tested_cells:
            level_advanced = frame.levels_completed > levels_completed
            state.tested_cells[entered] = value43.TestRecord(
                cell=entered,
                step=step,
                changed_cells=len(changed),
                new_cells=new_cells,
                level_advanced=level_advanced,
                game_state=frame.state.value,
            )
            if state.current_target == entered:
                state.current_target = None
                state.current_target_age = 0

        final_state = frame.state.value
        levels_completed = frame.levels_completed
        state.log.append(
            f"{step}:A{action}:{reason}:moved={moved}:new={new_cells}:changed={len(changed)}:"
            f"known={len(state.known)}:tested={len(state.tested_cells)}:"
            f"cell={state.controlled_cell}:state={final_state}:level_done={levels_completed}"
        )
        if final_state in {"WIN", "GAME_OVER"} or levels_completed >= level:
            break

    if state.action_deltas:
        memory.action_deltas.update(state.action_deltas)
    if state.controlled_signature is not None:
        memory.avatar_signatures[state.controlled_signature] += 1

    oracle_hits, oracle_total, oracle_labels = frontier42._oracle_visible_cells_seen(env._game, state)
    labels_by_cell = _oracle_label_by_cell(env._game)
    tested_labels = {
        cell: labels_by_cell.get(cell, [])
        for cell in state.tested_cells
        if labels_by_cell.get(cell)
    }
    return {
        "level": level,
        "bootstrap": bootstrap,
        "steps": len([line for line in state.log if ":A" in line]),
        "known_cells": len(state.known),
        "tested_cells": len(state.tested_cells),
        "controlled_cell": state.controlled_cell,
        "final_state": final_state,
        "levels_completed": levels_completed,
        "action_deltas": dict(state.action_deltas),
        "oracle_hits": oracle_hits,
        "oracle_total": oracle_total,
        "oracle_labels": oracle_labels,
        "tested_oracle_labels": tested_labels,
        "log": state.log,
    }


def run_all_levels(steps: int) -> list[dict[str, object]]:
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make("ls20")
    env.reset()
    memory = CrossLevelMemory()
    results = []
    for level in range(1, len(env._game._levels) + 1):
        results.append(run_level_with_memory(env, level, steps, memory))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()

    results = run_all_levels(args.steps)
    print("ls20 cross-level control-memory explorer")
    for result in results:
        print(
            f"level={result['level']} bootstrap={result['bootstrap']} steps={result['steps']} "
            f"state={result['final_state']} known={result['known_cells']} tested={result['tested_cells']} "
            f"controlled={result['controlled_cell']} actions={result['action_deltas']}"
        )
        print(
            f"  oracle_seen={result['oracle_hits']}/{result['oracle_total']} "
            f"tested_oracles={result['tested_oracle_labels']}"
        )


if __name__ == "__main__":
    main()
