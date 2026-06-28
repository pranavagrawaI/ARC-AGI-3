"""0043 - Value-aware cell-patch object explorer for fogged ls20.

Run from the repo root:
    uv run python teach/lessons/0043-ls20-value-aware-object-explorer.py --level 7 --steps 120

0042 explored fog by maximizing map coverage. This lesson adds a small value
prior: once an object-like cell is discovered, route to it and test it before
returning to frontier expansion. Roles are still not assumed. A tested cell is
just "the controlled patch entered this cell and we observed the consequence."

Private game state is used only for final scoring labels.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import sys as _sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import arc_agi
import numpy as np
from arc_agi.base import OperationMode
from arcengine import GameAction


REPO = Path(__file__).resolve().parents[2]


def _load_0042():
    path = REPO / "teach/lessons/0042-ls20-fog-cell-patch-frontier-explorer.py"
    spec = _ilu.spec_from_file_location("vaoe43_frontier42", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


frontier42 = _load_0042()
verify41 = frontier42.verify41
diag39 = frontier42.diag39


@dataclass
class TestRecord:
    cell: tuple[int, int]
    step: int
    changed_cells: int
    new_cells: int
    level_advanced: bool
    game_state: str

    @property
    def effectful(self) -> bool:
        return (
            self.level_advanced
            or self.game_state != "NOT_FINISHED"
            or self.changed_cells >= 60
            or self.new_cells >= 10
        )


@dataclass
class ValueExplorerState(frontier42.ExplorerState):
    tested_cells: dict[tuple[int, int], TestRecord] = field(default_factory=dict)
    unreachable_targets: set[tuple[int, int]] = field(default_factory=set)
    current_target: tuple[int, int] | None = None
    current_target_age: int = 0
    target_history: list[tuple[int, tuple[int, int]]] = field(default_factory=list)


def _cell_score(state: ValueExplorerState, cell: tuple[int, int]) -> tuple[int, int, int, int]:
    item = state.known[cell]
    recently_seen = 1 if item.last_seen >= max(0, len(state.log) - 20) else 0
    distinct = len(item.colors)
    non_bg = sum(count for color, count in item.colors if color != frontier42.FOG_COLOR)
    dist = 0
    if state.controlled_cell is not None:
        dist = abs(state.controlled_cell[0] - cell[0]) + abs(state.controlled_cell[1] - cell[1])
    return (-dist, recently_seen, distinct, non_bg)


def _object_targets(state: ValueExplorerState) -> list[tuple[int, int]]:
    targets = []
    for cell, item in state.known.items():
        if cell == state.controlled_cell:
            continue
        if cell in state.tested_cells or cell in state.unreachable_targets:
            continue
        if not item.object_like:
            continue
        if item.signature == state.controlled_signature:
            continue
        # HUD is visually object-like but not part of the maze; keep this first
        # pass focused on world cells.
        if cell[0] >= 55:
            continue
        targets.append(cell)
    targets.sort(key=lambda cell: _cell_score(state, cell), reverse=True)
    return targets


def _path_to_cell(
    state: ValueExplorerState,
    target: tuple[int, int],
) -> tuple[int, ...] | None:
    start = state.controlled_cell
    if start is None:
        return None
    if start == target:
        return ()
    q = deque([(start, ())])
    seen = {start}
    while q:
        cell, path = q.popleft()
        for action, nxt in frontier42._neighbors(cell, state.action_deltas):
            if not frontier42._in_lattice(nxt) or nxt in seen:
                continue
            if nxt not in state.known:
                continue
            if (cell, action) in state.blocked_edges:
                continue
            npath = path + (action,)
            if nxt == target:
                return npath
            seen.add(nxt)
            q.append((nxt, npath))
    return None


def _choose_target_action(state: ValueExplorerState) -> tuple[int, str]:
    if state.controlled_cell is not None and len(state.action_deltas) < len(frontier42.MOVE_ACTIONS):
        for action in frontier42.MOVE_ACTIONS:
            edge = (state.controlled_cell, action)
            if edge not in state.tried_edges and edge not in state.blocked_edges:
                return action, "edge_probe"

    if state.current_target is not None:
        if state.current_target_age > 20:
            state.unreachable_targets.add(state.current_target)
            state.current_target = None
            state.current_target_age = 0
        else:
            state.current_target_age += 1
    if state.current_target is not None:
        path = _path_to_cell(state, state.current_target)
        if path is None:
            state.unreachable_targets.add(state.current_target)
            state.current_target = None
            state.current_target_age = 0
        elif path:
            return path[0], f"target:{state.current_target}"
        else:
            state.current_target = None
            state.current_target_age = 0

    for target in _object_targets(state):
        path = _path_to_cell(state, target)
        if path is None:
            state.unreachable_targets.add(target)
            continue
        state.current_target = target
        state.current_target_age = 0
        state.target_history.append((len(state.log), target))
        if path:
            return path[0], f"target:{target}"
        state.current_target = None
        state.current_target_age = 0
        break

    if state.controlled_cell is not None:
        for action in frontier42.MOVE_ACTIONS:
            edge = (state.controlled_cell, action)
            if edge not in state.tried_edges and edge not in state.blocked_edges:
                return action, "edge_probe"

    return frontier42._choose_explore_action(state), "frontier"


def _oracle_label_by_cell(game) -> dict[tuple[int, int], list[str]]:
    labels: dict[tuple[int, int], list[str]] = {}
    for oracle in verify41._sprite_oracles(game):
        labels.setdefault(oracle.cell, []).append(oracle.label)
    return labels


def run_value_explorer(level: int, steps: int) -> dict[str, object]:
    level_index = level - 1
    controlled_sig, action_deltas, blocked_actions = frontier42._learn_initial_control(level_index)
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make("ls20")
    grid = frontier42._set_level(env, level_index)
    state = ValueExplorerState(
        controlled_signature=controlled_sig,
        action_deltas=action_deltas,
    )
    frontier42._update_map(state, grid, 0)
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
            state.log.append(f"{step}:blocked_no_control_model")
            break
        prev_grid = grid
        prev_cell = state.controlled_cell
        action, reason = _choose_target_action(state)
        state.tried_edges.add((prev_cell, action))
        frame = env.step(GameAction.from_id(action))
        grid = diag39._grid(frame)
        changed = frontier42._changed_visible_cells(prev_grid, grid)
        new_cells = frontier42._update_map(state, grid, step)
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
            state.tested_cells[entered] = TestRecord(
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

    oracle_hits, oracle_total, oracle_labels = frontier42._oracle_visible_cells_seen(env._game, state)
    labels_by_cell = _oracle_label_by_cell(env._game)
    tested_labels = {
        cell: labels_by_cell.get(cell, [])
        for cell in state.tested_cells
        if labels_by_cell.get(cell)
    }
    effectful = [record for record in state.tested_cells.values() if record.effectful]
    return {
        "level": level,
        "steps": len([line for line in state.log if ":A" in line]),
        "known_cells": len(state.known),
        "tested_cells": len(state.tested_cells),
        "effectful_tests": len(effectful),
        "controlled_cell": state.controlled_cell,
        "final_state": final_state,
        "levels_completed": levels_completed,
        "action_deltas": state.action_deltas,
        "blocked_edges": len(state.blocked_edges),
        "oracle_hits": oracle_hits,
        "oracle_total": oracle_total,
        "oracle_labels": oracle_labels,
        "tested_oracle_labels": tested_labels,
        "target_history": state.target_history,
        "log": state.log,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=7)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()

    result = run_value_explorer(args.level, args.steps)
    print("ls20 value-aware cell-patch object explorer")
    print(
        f"level={result['level']} steps={result['steps']} known_cells={result['known_cells']} "
        f"tested_cells={result['tested_cells']} effectful_tests={result['effectful_tests']} "
        f"controlled_cell={result['controlled_cell']} state={result['final_state']} "
        f"levels_completed={result['levels_completed']}"
    )
    print(f"action_deltas={result['action_deltas']} blocked_edges={result['blocked_edges']}")
    print(
        f"oracle_sprite_cells_seen={result['oracle_hits']}/{result['oracle_total']} "
        f"labels={result['oracle_labels']}"
    )
    print(f"tested_oracle_labels={result['tested_oracle_labels']}")
    print(f"target_history={result['target_history'][:20]}")
    print("log:")
    for line in result["log"][:90]:
        print(f"  {line}")


if __name__ == "__main__":
    main()
