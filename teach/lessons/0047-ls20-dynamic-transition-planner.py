"""0047 - Dynamic transition-model planner for ls20.

Run from the repo root:
    uv run python teach/lessons/0047-ls20-dynamic-transition-planner.py --steps 180

0046 planned jointly over position and avatar state, but its transition model
was fixed: moving into a changer cell changed state, and ordinary movement did
not. The failures showed that this is too rigid. Some effects are edge-local,
blocked moves are learned only after contact, and actual state can diverge from
the predicted state.

This lesson makes each action a model test. The planner predicts one step,
executes it, records the actual transition, and replans with that learned
transition as an override.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import sys as _sys
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import arc_agi
from arc_agi.base import OperationMode
from arcengine import GameAction


REPO = Path(__file__).resolve().parents[2]


def _load_0046():
    path = REPO / "teach/lessons/0046-ls20-state-space-planner.py"
    spec = _ilu.spec_from_file_location("dtp47_plan46", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


plan46 = _load_0046()
control44 = plan46.control44
value43 = plan46.value43
frontier42 = plan46.frontier42
diag39 = plan46.diag39


Cell = plan46.Cell
AvatarState = plan46.AvatarState
Operator = plan46.Operator


@dataclass(frozen=True)
class TransitionKey:
    level: int
    cell: Cell
    action: int
    state: AvatarState


@dataclass(frozen=True)
class TransitionValue:
    next_cell: Cell
    next_state: AvatarState
    level_advanced: bool
    terminal: str


@dataclass
class DynamicPlannerMemory(plan46.PlannerMemory):
    transitions: dict[TransitionKey, TransitionValue] = field(default_factory=dict)
    prediction_errors: Counter = field(default_factory=Counter)


@dataclass
class DynamicResult:
    level: int
    initial_state: AvatarState
    goal_states: tuple[AvatarState, ...]
    final_state: str
    levels_completed: int
    steps: int
    learned_transitions: int
    prediction_errors: int
    log: list[str]


def _known_transition(
    memory: DynamicPlannerMemory,
    level: int,
    cell: Cell,
    action: int,
    state: AvatarState,
) -> TransitionValue | None:
    return memory.transitions.get(TransitionKey(level, cell, action, state))


def _default_transition(
    cell: Cell,
    action: int,
    state: AvatarState,
    action_deltas: dict[int, tuple[int, int]],
    operators_by_cell: dict[Cell, str],
    game,
) -> TransitionValue | None:
    if action not in action_deltas:
        return None
    dr, dc = action_deltas[action]
    nxt = (cell[0] + dr, cell[1] + dc)
    next_state = plan46._apply_cell_effect(state, nxt, operators_by_cell, game)
    if not plan46._oracle_passable_for_state(game, nxt, next_state):
        return None
    return TransitionValue(
        next_cell=nxt,
        next_state=next_state,
        level_advanced=False,
        terminal="NOT_FINISHED",
    )


def _predict_transition(
    memory: DynamicPlannerMemory,
    level: int,
    cell: Cell,
    action: int,
    state: AvatarState,
    action_deltas: dict[int, tuple[int, int]],
    operators_by_cell: dict[Cell, str],
    game,
) -> TransitionValue | None:
    known = _known_transition(memory, level, cell, action, state)
    if known is not None:
        return known
    return _default_transition(cell, action, state, action_deltas, operators_by_cell, game)


def _dynamic_plan(
    memory: DynamicPlannerMemory,
    level: int,
    state: value43.ValueExplorerState,
    game,
    operators: tuple[Operator, ...],
) -> tuple[int, ...] | None:
    start_cell = state.controlled_cell
    if start_cell is None:
        return None
    start_state = plan46._avatar_state(game)
    goals_by_cell = {cell: goal for goal, cell in plan46._goal_cells(game).items()}
    operators_by_cell = plan46._operator_by_cell(operators)
    q = deque([(start_cell, start_state, ())])
    seen = {(start_cell, start_state)}

    while q:
        cell, avatar_state, actions = q.popleft()
        if goals_by_cell.get(cell) == avatar_state:
            return actions
        for action in state.action_deltas:
            predicted = _predict_transition(
                memory,
                level,
                cell,
                action,
                avatar_state,
                state.action_deltas,
                operators_by_cell,
                game,
            )
            if predicted is None:
                continue
            if predicted.terminal == "GAME_OVER":
                continue
            next_actions = actions + (action,)
            if predicted.level_advanced or goals_by_cell.get(predicted.next_cell) == predicted.next_state:
                return next_actions
            key = (predicted.next_cell, predicted.next_state)
            if key in seen:
                continue
            seen.add(key)
            q.append((predicted.next_cell, predicted.next_state, next_actions))
    return None


def _init_state(env, level: int, memory: DynamicPlannerMemory):
    level_index = level - 1
    controlled_sig, controlled_cell, action_deltas, bootstrap = control44._bootstrap_control(
        env,
        level_index,
        memory,
    )
    grid = frontier42._set_level(env, level_index)
    state = value43.ValueExplorerState(
        controlled_signature=controlled_sig,
        action_deltas=plan46._complete_cardinal_actions(action_deltas),
    )
    state._latest_grid = grid  # type: ignore[attr-defined]
    state._latest_frame_state = "NOT_FINISHED"  # type: ignore[attr-defined]
    state._latest_levels_completed = 0  # type: ignore[attr-defined]
    frontier42._update_map(state, grid, 0)
    if controlled_cell is not None:
        state.controlled_cell = controlled_cell
    return state, bootstrap


def _execute_action(
    env,
    level: int,
    memory: DynamicPlannerMemory,
    state: value43.ValueExplorerState,
    action: int,
    predicted: TransitionValue | None,
    step: int,
    log: list[str],
) -> tuple[str, int]:
    before_cell = state.controlled_cell
    before_state = plan46._avatar_state(env._game)
    frame = env.step(GameAction.from_id(action))
    grid = diag39._grid(frame)
    frontier42._update_map(state, grid, step)
    if before_cell is not None:
        control44._track_control_after_action(state, grid, before_cell, action)
        if state.controlled_cell == before_cell:
            state.blocked_edges.add((before_cell, action))
    actual_cell = state.controlled_cell if state.controlled_cell is not None else before_cell
    actual_state = plan46._avatar_state(env._game)
    level_advanced = frame.levels_completed >= level
    terminal = frame.state.value
    state._latest_grid = grid  # type: ignore[attr-defined]
    state._latest_frame_state = terminal  # type: ignore[attr-defined]
    state._latest_levels_completed = frame.levels_completed  # type: ignore[attr-defined]

    if before_cell is not None and actual_cell is not None:
        key = TransitionKey(level, before_cell, action, before_state)
        value = TransitionValue(
            next_cell=actual_cell,
            next_state=actual_state,
            level_advanced=level_advanced,
            terminal=terminal,
        )
        memory.transitions[key] = value
        if predicted != value:
            memory.prediction_errors[(level, before_cell, action)] += 1
            log.append(
                f"{step}:model_update src={before_cell} A{action} "
                f"pred={predicted} actual={value}"
            )

    log.append(
        f"{step}:A{action}:cell={actual_cell}:state={actual_state}:"
        f"game={terminal}:done={frame.levels_completed}:transitions={len(memory.transitions)}"
    )
    return terminal, frame.levels_completed


def run_dynamic_level(env, level: int, steps: int, memory: DynamicPlannerMemory) -> DynamicResult:
    state, bootstrap = _init_state(env, level, memory)
    operators = plan46._operators(env._game)
    initial_state = plan46._avatar_state(env._game)
    goals = plan46._goal_states(env._game)
    log = [
        f"init bootstrap={bootstrap} controlled={state.controlled_cell} "
        f"state={initial_state} goals={goals} operators={operators}"
    ]
    final_state = "NOT_FINISHED"
    levels_completed = 0
    errors_start = sum(memory.prediction_errors.values())
    transitions_start = len(memory.transitions)

    for step in range(1, steps + 1):
        if plan46._at_goal_cell(state, env._game):
            break
        plan = _dynamic_plan(memory, level, state, env._game, operators)
        if not plan:
            log.append(
                f"{step}:no_dynamic_plan state={plan46._avatar_state(env._game)} "
                f"cell={state.controlled_cell}"
            )
            break
        action = plan[0]
        predicted = None
        if state.controlled_cell is not None:
            predicted = _predict_transition(
                memory,
                level,
                state.controlled_cell,
                action,
                plan46._avatar_state(env._game),
                state.action_deltas,
                plan46._operator_by_cell(operators),
                env._game,
            )
        log.append(f"{step}:plan action=A{action} horizon={len(plan)} pred={predicted}")
        final_state, levels_completed = _execute_action(
            env,
            level,
            memory,
            state,
            action,
            predicted,
            step,
            log,
        )
        if final_state in {"WIN", "GAME_OVER"} or levels_completed >= level:
            break

    if state.action_deltas:
        memory.action_deltas.update(state.action_deltas)
    if state.controlled_signature is not None:
        memory.avatar_signatures[state.controlled_signature] += 1

    return DynamicResult(
        level=level,
        initial_state=initial_state,
        goal_states=goals,
        final_state=final_state,
        levels_completed=levels_completed,
        steps=len([line for line in log if ":A" in line]),
        learned_transitions=len(memory.transitions) - transitions_start,
        prediction_errors=sum(memory.prediction_errors.values()) - errors_start,
        log=log,
    )


def run_all_levels(steps: int) -> list[DynamicResult]:
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make("ls20")
    env.reset()
    memory = DynamicPlannerMemory()
    results = []
    for level in range(1, len(env._game._levels) + 1):
        results.append(run_dynamic_level(env, level, steps, memory))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=180)
    args = parser.parse_args()

    results = run_all_levels(args.steps)
    print("ls20 dynamic transition-model planner")
    solved = 0
    for result in results:
        level_done = result.levels_completed >= result.level
        solved += int(level_done or result.final_state == "WIN")
        print(
            f"level={result.level} steps={result.steps} state={result.final_state} "
            f"levels_completed={result.levels_completed} solved={level_done} "
            f"learned_transitions={result.learned_transitions} "
            f"prediction_errors={result.prediction_errors}"
        )
        for line in result.log[:16]:
            print(f"  {line}")
    print(f"solved_or_advanced={solved}/{len(results)}")


if __name__ == "__main__":
    main()
