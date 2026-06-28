"""0046 - State-space planning scaffold for ls20.

Run from the repo root:
    uv run python teach/lessons/0046-ls20-state-space-planner.py --steps 120

0045 learned that object purpose should persist across levels. This lesson adds
the first planner: search over avatar state, choose changers that move the state
toward a goal, then route to the goal.

This is intentionally a planning scaffold. It still uses private game state to
label the current state, goal states, and changer cells so we can isolate the
planning question from the still-open visual state-inference question.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import sys as _sys
from collections import Counter
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import arc_agi
from arc_agi.base import OperationMode
from arcengine import GameAction


REPO = Path(__file__).resolve().parents[2]


def _load_0045():
    path = REPO / "teach/lessons/0045-ls20-cross-level-object-effect-memory.py"
    spec = _ilu.spec_from_file_location("ssp46_effect45", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


effect45 = _load_0045()
control44 = effect45.control44
value43 = effect45.value43
frontier42 = effect45.frontier42
diag39 = effect45.diag39


Cell = tuple[int, int]


@dataclass(frozen=True)
class AvatarState:
    shape: int
    color: int
    rotation: int


@dataclass(frozen=True)
class Operator:
    kind: str
    cell: Cell


@dataclass
class PlannerMemory(control44.CrossLevelMemory):
    known_operator_kinds: Counter = field(default_factory=Counter)


@dataclass
class PlanResult:
    level: int
    initial_state: AvatarState
    goal_states: tuple[AvatarState, ...]
    high_level_plan: tuple[str, ...]
    final_state: str
    levels_completed: int
    steps: int
    log: list[str]


def _avatar_state(game) -> AvatarState:
    return AvatarState(game.fwckfzsyc, game.hiaauhahz, game.cklxociuu)


def _goal_states(game) -> tuple[AvatarState, ...]:
    return tuple(
        AvatarState(shape, color, rotation)
        for shape, color, rotation in zip(game.ldxlnycps, game.yjdexjsoa, game.ehwheiwsk)
    )


def _goal_cells(game) -> dict[AvatarState, Cell]:
    return {
        AvatarState(shape, color, rotation): diag39._cell_top_left(sprite.y, sprite.x)
        for shape, color, rotation, sprite in zip(
            game.ldxlnycps,
            game.yjdexjsoa,
            game.ehwheiwsk,
            game.plrpelhym,
        )
    }


def _operators(game) -> tuple[Operator, ...]:
    out = []
    for tag, kind in (
        ("ttfwljgohq", "shape"),
        ("soyhouuebz", "color"),
        ("rhsxkxzdjz", "rotation"),
    ):
        for sprite in game.current_level.get_sprites_by_tag(tag):
            if getattr(sprite, "is_visible", True):
                out.append(Operator(kind, diag39._cell_top_left(sprite.y, sprite.x)))
    return tuple(out)


def _complete_cardinal_actions(action_deltas: dict[int, tuple[int, int]]) -> dict[int, tuple[int, int]]:
    completed = dict(action_deltas)
    inverse_pairs = ((1, 2), (3, 4))
    for first, second in inverse_pairs:
        if first in completed and second not in completed:
            dr, dc = completed[first]
            completed[second] = (-dr, -dc)
        if second in completed and first not in completed:
            dr, dc = completed[second]
            completed[first] = (-dr, -dc)
    return completed


def _apply_operator(state: AvatarState, kind: str, game) -> AvatarState:
    if kind == "shape":
        return AvatarState((state.shape + 1) % len(game.ijessuuig), state.color, state.rotation)
    if kind == "color":
        return AvatarState(state.shape, (state.color + 1) % len(game.tnkekoeuk), state.rotation)
    if kind == "rotation":
        return AvatarState(state.shape, state.color, (state.rotation + 1) % 4)
    raise ValueError(f"unknown operator kind: {kind}")


def _state_plan(game, operators: tuple[Operator, ...]) -> tuple[str, ...] | None:
    start = _avatar_state(game)
    goals = set(_goal_states(game))
    if start in goals:
        return ()
    available_kinds = tuple(sorted({operator.kind for operator in operators}))
    q = deque([(start, ())])
    seen = {start}
    while q:
        state, path = q.popleft()
        for kind in available_kinds:
            nxt = _apply_operator(state, kind, game)
            if nxt in seen:
                continue
            npath = path + (kind,)
            if nxt in goals:
                return npath
            seen.add(nxt)
            q.append((nxt, npath))
    return None


def _closest_operator(
    state: value43.ValueExplorerState,
    operators: tuple[Operator, ...],
    kind: str,
    game,
) -> Operator | None:
    candidates = [operator for operator in operators if operator.kind == kind]
    routed = []
    for operator in candidates:
        path = _oracle_path_to_cell(state, operator.cell, game)
        if path is not None:
            routed.append((len(path), operator))
    if not routed:
        return None
    routed.sort(key=lambda item: item[0])
    return routed[0][1]


def _oracle_passable(game, cell: Cell) -> bool:
    return _oracle_passable_for_state(game, cell, _avatar_state(game))


def _oracle_passable_for_state(game, cell: Cell, avatar_state: AvatarState) -> bool:
    row, col = cell
    if not frontier42._in_lattice(cell):
        return False
    sprites = game.mrznumynfe(col, row, game.gisrhqpee, game.tbwnoxqgc)
    for sprite in sprites:
        tags = getattr(sprite, "tags", None)
        if tags is None:
            continue
        if "ihdgageizm" in tags:
            return False
        if "rjlbuycveu" in tags:
            goal_index = game.plrpelhym.index(sprite)
            goal = AvatarState(
                game.ldxlnycps[goal_index],
                game.yjdexjsoa[goal_index],
                game.ehwheiwsk[goal_index],
            )
            return avatar_state == goal
    return True


def _operator_by_cell(operators: tuple[Operator, ...]) -> dict[Cell, str]:
    return {operator.cell: operator.kind for operator in operators}


def _apply_cell_effect(avatar_state: AvatarState, cell: Cell, operator_by_cell: dict[Cell, str], game) -> AvatarState:
    kind = operator_by_cell.get(cell)
    if kind is None:
        return avatar_state
    return _apply_operator(avatar_state, kind, game)


def _joint_plan_actions(
    state: value43.ValueExplorerState,
    game,
    operators: tuple[Operator, ...],
) -> tuple[tuple[int, ...], tuple[str, ...]] | None:
    start_cell = state.controlled_cell
    if start_cell is None:
        return None
    start_state = _avatar_state(game)
    goals_by_cell = {cell: goal for goal, cell in _goal_cells(game).items()}
    operator_cells = _operator_by_cell(operators)
    q = deque([(start_cell, start_state, (), ())])
    seen = {(start_cell, start_state)}
    while q:
        cell, avatar_state, actions, kinds = q.popleft()
        if goals_by_cell.get(cell) == avatar_state:
            return actions, kinds
        for action, nxt in frontier42._neighbors(cell, state.action_deltas):
            if (cell, action) in state.blocked_edges:
                continue
            next_avatar_state = _apply_cell_effect(avatar_state, nxt, operator_cells, game)
            if not _oracle_passable_for_state(game, nxt, next_avatar_state):
                continue
            key = (nxt, next_avatar_state)
            if key in seen:
                continue
            next_kinds = kinds
            if nxt in operator_cells:
                next_kinds = kinds + (operator_cells[nxt],)
            next_actions = actions + (action,)
            if goals_by_cell.get(nxt) == next_avatar_state:
                return next_actions, next_kinds
            seen.add(key)
            q.append((nxt, next_avatar_state, next_actions, next_kinds))
    return None


def _at_goal_cell(state: value43.ValueExplorerState, game) -> bool:
    if state.controlled_cell is None:
        return False
    return _goal_cells(game).get(_avatar_state(game)) == state.controlled_cell


def _oracle_path_to_cell(
    state: value43.ValueExplorerState,
    target: Cell,
    game,
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
            if nxt in seen or (cell, action) in state.blocked_edges:
                continue
            if nxt != target and not _oracle_passable(game, nxt):
                continue
            if nxt == target and not _oracle_passable(game, nxt):
                continue
            npath = path + (action,)
            if nxt == target:
                return npath
            seen.add(nxt)
            q.append((nxt, npath))
    return None


def _refresh_state_from_frame(state: value43.ValueExplorerState, grid, step: int) -> None:
    frontier42._update_map(state, grid, step)


def _step_action(env, state: value43.ValueExplorerState, action: int, log: list[str], step: int):
    prev_grid = state._latest_grid  # type: ignore[attr-defined]
    prev_cell = state.controlled_cell
    state.tried_edges.add((prev_cell, action))
    frame = env.step(GameAction.from_id(action))
    grid = diag39._grid(frame)
    _refresh_state_from_frame(state, grid, step)
    if prev_cell is not None:
        control44._track_control_after_action(state, grid, prev_cell, action)
        if state.controlled_cell == prev_cell:
            state.blocked_edges.add((prev_cell, action))
    state._latest_grid = grid  # type: ignore[attr-defined]
    state._latest_frame_state = frame.state.value  # type: ignore[attr-defined]
    state._latest_levels_completed = frame.levels_completed  # type: ignore[attr-defined]
    changed = frontier42._changed_visible_cells(prev_grid, grid)
    log.append(
        f"{step}:A{action}:cell={state.controlled_cell}:changed={len(changed)}:"
        f"private_state={_avatar_state(env._game)}:game={frame.state.value}:done={frame.levels_completed}"
    )
    return frame


def _route_to(
    env,
    state: value43.ValueExplorerState,
    target: Cell,
    log: list[str],
    step_counter: list[int],
    max_steps: int,
) -> bool:
    while step_counter[0] < max_steps and state.controlled_cell != target:
        path = _oracle_path_to_cell(state, target, env._game)
        if not path:
            log.append(f"{step_counter[0]}:unreachable target={target}")
            return False
        action = path[0]
        step_counter[0] += 1
        before = state.controlled_cell
        frame = _step_action(env, state, action, log, step_counter[0])
        if frame.state.value in {"WIN", "GAME_OVER"}:
            return True
        if state.controlled_cell == before:
            continue
    return state.controlled_cell == target


def _leave_current_cell(
    env,
    state: value43.ValueExplorerState,
    log: list[str],
    step_counter: list[int],
    max_steps: int,
) -> bool:
    current = state.controlled_cell
    if current is None:
        return False
    for action, nxt in frontier42._neighbors(current, state.action_deltas):
        if step_counter[0] >= max_steps:
            return False
        if not _oracle_passable(env._game, nxt):
            continue
        step_counter[0] += 1
        before = state.controlled_cell
        _step_action(env, state, action, log, step_counter[0])
        if state.controlled_cell != before:
            return True
    return False


def _trigger_operator(
    env,
    state: value43.ValueExplorerState,
    operator: Operator,
    log: list[str],
    step_counter: list[int],
    max_steps: int,
) -> bool:
    if state.controlled_cell == operator.cell:
        log.append(f"leave_operator_cell cell={operator.cell}")
        if not _leave_current_cell(env, state, log, step_counter, max_steps):
            return False
    return _route_to(env, state, operator.cell, log, step_counter, max_steps)


def _plan_goal_cell(game, current_state: AvatarState) -> Cell | None:
    cells = _goal_cells(game)
    if current_state in cells:
        return cells[current_state]
    return None


def run_planner_level(env, level: int, steps: int, memory: PlannerMemory) -> PlanResult:
    level_index = level - 1
    controlled_sig, controlled_cell, action_deltas, bootstrap = control44._bootstrap_control(
        env,
        level_index,
        memory,
    )
    grid = frontier42._set_level(env, level_index)
    state = value43.ValueExplorerState(
        controlled_signature=controlled_sig,
        action_deltas=_complete_cardinal_actions(action_deltas),
    )
    state._latest_grid = grid  # type: ignore[attr-defined]
    state._latest_frame_state = "NOT_FINISHED"  # type: ignore[attr-defined]
    state._latest_levels_completed = 0  # type: ignore[attr-defined]
    _refresh_state_from_frame(state, grid, 0)
    if controlled_cell is not None:
        state.controlled_cell = controlled_cell

    operators = _operators(env._game)
    for operator in operators:
        memory.known_operator_kinds[operator.kind] += 1
    high_level_plan = _state_plan(env._game, operators)
    initial_state = _avatar_state(env._game)
    goals = _goal_states(env._game)
    log = [
        f"init bootstrap={bootstrap} controlled={state.controlled_cell} "
        f"state={initial_state} goals={goals} operators={operators} plan={high_level_plan}"
    ]
    step_counter = [0]
    final_state = "NOT_FINISHED"
    levels_completed = 0

    executed_plan = []
    joint_plan = _joint_plan_actions(state, env._game, operators)
    if joint_plan is not None:
        while step_counter[0] < steps and not _at_goal_cell(state, env._game):
            joint_plan = _joint_plan_actions(state, env._game, operators)
            if joint_plan is None:
                log.append("joint_plan_lost")
                break
            joint_actions, joint_kinds = joint_plan
            if not joint_actions:
                break
            action = joint_actions[0]
            next_cell = None
            if state.controlled_cell is not None:
                for candidate_action, candidate_cell in frontier42._neighbors(
                    state.controlled_cell,
                    state.action_deltas,
                ):
                    if candidate_action == action:
                        next_cell = candidate_cell
                        break
            if joint_kinds and next_cell is not None:
                operator_kind = _operator_by_cell(operators).get(next_cell)
                if operator_kind is not None:
                    executed_plan.append(operator_kind)
            log.append(
                f"joint_replan action=A{action} remaining={len(joint_actions)} "
                f"kinds={joint_kinds}"
            )
            step_counter[0] += 1
            frame = _step_action(env, state, action, log, step_counter[0])
            final_state = frame.state.value
            levels_completed = frame.levels_completed
            if final_state in {"WIN", "GAME_OVER"} or levels_completed >= level:
                break
    while (
        joint_plan is None
        and _avatar_state(env._game) not in set(goals)
        and step_counter[0] < steps
    ):
        high_level_plan = _state_plan(env._game, operators)
        if high_level_plan is None:
            log.append("no_state_plan")
            break
        kind = high_level_plan[0]
        executed_plan.append(kind)
        operator = _closest_operator(state, operators, kind, env._game)
        if operator is None:
            log.append(f"no_reachable_operator kind={kind}")
            break
        log.append(f"route_operator kind={kind} cell={operator.cell}")
        reached = _trigger_operator(env, state, operator, log, step_counter, steps)
        final_state = state._latest_frame_state  # type: ignore[attr-defined]
        levels_completed = state._latest_levels_completed  # type: ignore[attr-defined]
        if not reached or final_state in {"WIN", "GAME_OVER"}:
            break

    if final_state not in {"WIN", "GAME_OVER"} and levels_completed < level:
        if _avatar_state(env._game) not in set(goals):
            log.append(f"state_plan_incomplete state={_avatar_state(env._game)}")
        else:
            goal_cell = _plan_goal_cell(env._game, _avatar_state(env._game))
            if goal_cell is None:
                log.append(f"no_goal_for_state state={_avatar_state(env._game)}")
            else:
                log.append(f"route_goal cell={goal_cell} state={_avatar_state(env._game)}")
                _route_to(env, state, goal_cell, log, step_counter, steps)
                final_state = state._latest_frame_state  # type: ignore[attr-defined]
                levels_completed = state._latest_levels_completed  # type: ignore[attr-defined]

    if state.action_deltas:
        memory.action_deltas.update(state.action_deltas)
    if state.controlled_signature is not None:
        memory.avatar_signatures[state.controlled_signature] += 1

    return PlanResult(
        level=level,
        initial_state=initial_state,
        goal_states=goals,
        high_level_plan=tuple(executed_plan),
        final_state=final_state,
        levels_completed=levels_completed,
        steps=step_counter[0],
        log=log,
    )


def run_all_levels(steps: int) -> list[PlanResult]:
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make("ls20")
    env.reset()
    memory = PlannerMemory()
    results = []
    for level in range(1, len(env._game._levels) + 1):
        results.append(run_planner_level(env, level, steps, memory))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()

    results = run_all_levels(args.steps)
    print("ls20 oracle-scaffolded state-space planner")
    solved = 0
    for result in results:
        level_done = result.levels_completed >= result.level
        solved += int(level_done or result.final_state == "WIN")
        print(
            f"level={result.level} steps={result.steps} state={result.final_state} "
            f"levels_completed={result.levels_completed} solved={level_done} "
            f"initial={result.initial_state} goals={result.goal_states} plan={result.high_level_plan}"
        )
        for line in result.log[:14]:
            print(f"  {line}")
    print(f"solved_or_advanced={solved}/{len(results)}")


if __name__ == "__main__":
    main()
