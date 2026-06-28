"""0049 - Option-level simulator-backed planner for ls20.

Run from the repo root:
    uv run python teach/lessons/0049-ls20-option-simulator-planner.py --levels 5

0048 used the wrong ontology for later LS20 levels: static cell transitions.
This lesson plans over semantic options, but validates every option by stepping a
cloned game state with the real engine and no rendering.
"""

from __future__ import annotations

import argparse
import copy
import heapq
import importlib.util as _ilu
import itertools
import sys as _sys
import time
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
from typing import Callable, Iterable

import arc_agi
from arc_agi.base import OperationMode
from arcengine import ActionInput, GameAction


REPO = Path(__file__).resolve().parents[2]
MOVE_ACTIONS = (1, 2, 3, 4)
ACTION_INPUTS = {action: ActionInput(id=GameAction.from_id(action)) for action in MOVE_ACTIONS}
ATTR_TAGS = {
    "shape": "ttfwljgohq",
    "color": "soyhouuebz",
    "rotation": "rhsxkxzdjz",
}
ATTR_MODULUS = {
    "rotation": 4,
}


def _load_0046():
    path = REPO / "teach/lessons/0046-ls20-state-space-planner.py"
    spec = _ilu.spec_from_file_location("osp49_plan46", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


plan46 = _load_0046()
AvatarState = plan46.AvatarState
Cell = plan46.Cell


@dataclass(frozen=True)
class Goal:
    index: int
    state: AvatarState
    cell: Cell


@dataclass(frozen=True)
class Option:
    kind: str
    attr: str | None = None
    target_value: int | None = None
    goal_index: int | None = None


@dataclass
class SearchStats:
    expanded: int = 0
    generated: int = 0
    pruned_game_over: int = 0
    pruned_regress: int = 0
    cache_hits: int = 0
    elapsed: float = 0.0


@dataclass
class OptionResult:
    option: Option
    success: bool
    game: object | None
    actions: tuple[int, ...] = ()
    stats: SearchStats = field(default_factory=SearchStats)
    reason: str = ""


@dataclass
class LevelResult:
    level: int
    solved: bool
    levels_completed: int
    final_state: str
    actions: tuple[int, ...]
    options: tuple[str, ...]
    log: list[str]
    elapsed: float


def clone_game(game):
    return copy.deepcopy(game)


def step_no_render(game, action: int) -> tuple[str, int]:
    """Advance one external action on a game clone without rendering frames."""
    game._set_action(ACTION_INPUTS[action])
    frames = 0
    while not game.is_action_complete():
        frames += 1
        if frames > 200:
            raise RuntimeError("action took too many internal frames")
        if game._next_level:
            game._really_set_next_level()
        else:
            game.step()
    return game._state.value, game._score


def replay_no_render(game, actions: Iterable[int]):
    out = clone_game(game)
    for action in actions:
        state, _score = step_no_render(out, action)
        if state in {"WIN", "GAME_OVER"}:
            break
    return out


def avatar_cell(game) -> Cell:
    return (game.gudziatsk.y, game.gudziatsk.x)


def avatar_state(game) -> AvatarState:
    return AvatarState(game.fwckfzsyc, game.hiaauhahz, game.cklxociuu)


def goals(game) -> tuple[Goal, ...]:
    return tuple(
        Goal(
            index=i,
            state=AvatarState(game.ldxlnycps[i], game.yjdexjsoa[i], game.ehwheiwsk[i]),
            cell=(sprite.y, sprite.x),
        )
        for i, sprite in enumerate(game.plrpelhym)
        if i >= len(game.lvrnuajbl) or not game.lvrnuajbl[i]
    )


def attr_modulus(game, attr: str) -> int:
    if attr == "shape":
        return len(game.ijessuuig)
    if attr == "color":
        return len(game.tnkekoeuk)
    return ATTR_MODULUS[attr]


def attr_value(state: AvatarState, attr: str) -> int:
    return getattr(state, attr)


def attr_debt(game, goal: Goal, attr: str) -> int:
    current = attr_value(avatar_state(game), attr)
    target = attr_value(goal.state, attr)
    return (target - current) % attr_modulus(game, attr)


def total_debt(game, goal: Goal) -> int:
    return sum(attr_debt(game, goal, attr) for attr in ATTR_TAGS)


def manhattan_cells(a: Cell, b: Cell) -> int:
    return (abs(a[0] - b[0]) + abs(a[1] - b[1])) // 5


def changer_cells(game, attr: str) -> tuple[Cell, ...]:
    cells = []
    for sprite in game.current_level.get_sprites_by_tag(ATTR_TAGS[attr]):
        if getattr(sprite, "is_visible", True):
            cells.append((sprite.y, sprite.x))
    return tuple(cells)


def nearest_changer_distance(game, attr: str) -> int:
    cells = changer_cells(game, attr)
    if not cells:
        return 99
    here = avatar_cell(game)
    return min(manhattan_cells(here, cell) for cell in cells)


def min_safe_steps_to_goal(game, goal: Goal) -> int:
    # LS20 level timers often decrement by 2 per action. Keep a small buffer so
    # option boundaries do not stop on a matching-but-doomed zero-timer state.
    return 2 * manhattan_cells(avatar_cell(game), goal.cell) + 4


def remaining_clock_pickups(game) -> int:
    return len(game.current_level.get_sprites_by_tag("npxgalaybz"))


def static_path_length(game, target: Cell) -> int | None:
    path = static_path_actions(game, target)
    if path is None:
        return None
    return len(path)


def static_path_actions(game, target: Cell) -> tuple[int, ...] | None:
    start = avatar_cell(game)
    if start == target:
        return ()
    deltas = {1: (-5, 0), 2: (5, 0), 3: (0, -5), 4: (0, 5)}
    q = deque([(start, ())])
    seen = {start}
    state = avatar_state(game)
    while q:
        cell, actions = q.popleft()
        for action, (dr, dc) in deltas.items():
            nxt = (cell[0] + dr, cell[1] + dc)
            if nxt in seen:
                continue
            if not plan46._oracle_passable_for_state(game, nxt, state):
                continue
            next_actions = actions + (action,)
            if nxt == target:
                return next_actions
            seen.add(nxt)
            q.append((nxt, next_actions))
    return None


def sprite_cell(sprite) -> Cell:
    row = (sprite.y // 5) * 5
    col = 4 + ((sprite.x - 4) // 5) * 5
    return (row, col)


def clock_pickup_cells(game) -> tuple[Cell, ...]:
    return tuple(sprite_cell(sprite) for sprite in game.current_level.get_sprites_by_tag("npxgalaybz"))


def replay_actions(game, actions: tuple[int, ...]):
    out = clone_game(game)
    for action in actions:
        state, _score = step_no_render(out, action)
        if state in {"WIN", "GAME_OVER"}:
            break
    return out


def exact_goal_route_proposal(game, target_level: int, goal: Goal) -> tuple[object | None, tuple[int, ...], str]:
    direct = static_path_actions(game, goal.cell)
    if direct is not None:
        out = replay_actions(game, direct)
        if out._score >= target_level:
            return out, direct, "static_direct"

    here = avatar_cell(game)
    pickups = sorted(clock_pickup_cells(game), key=lambda cell: manhattan_cells(here, cell))
    for pickup in pickups:
        to_pickup = static_path_actions(game, pickup)
        if to_pickup is None:
            continue
        after_pickup = replay_actions(game, to_pickup)
        if after_pickup._state.value == "GAME_OVER":
            continue
        if total_debt(after_pickup, goal) > total_debt(game, goal):
            continue
        to_goal = static_path_actions(after_pickup, goal.cell)
        if to_goal is None:
            continue
        actions = to_pickup + to_goal
        out = replay_actions(game, actions)
        if out._score >= target_level:
            return out, actions, f"static_via_clock:{pickup}"
    return None, (), "no_static_route"


def inverse_action(action: int) -> int:
    return {1: 2, 2: 1, 3: 4, 4: 3}[action]


def exact_set_attr_proposal(game, goal: Goal, attr: str) -> tuple[object | None, tuple[int, ...], str]:
    """Cheap option proposal for common LS20 changer interactions.

    It proposes route-like macros and validates the complete action sequence by
    replaying the exact engine. It does not assume that movement into the changer
    succeeds; it only trusts the replayed attribute delta.
    """
    target = attr_value(goal.state, attr)
    if attr_value(avatar_state(game), attr) == target:
        return clone_game(game), (), "already_set"

    prefixes: list[tuple[str, tuple[int, ...], object]] = [("direct", (), clone_game(game))]
    for pickup in sorted(clock_pickup_cells(game), key=lambda cell: manhattan_cells(avatar_cell(game), cell)):
        path = static_path_actions(game, pickup)
        if path is None:
            continue
        after = replay_actions(game, path)
        if after._state.value == "GAME_OVER":
            continue
        if remaining_clock_pickups(after) < remaining_clock_pickups(game):
            prefixes.append((f"clock:{pickup}", path, after))

    candidates = []
    for prefix_name, prefix_actions, prefix_game in prefixes:
        for changer in sorted(changer_cells(prefix_game, attr), key=lambda cell: manhattan_cells(avatar_cell(prefix_game), cell)):
            route = static_path_actions(prefix_game, changer)
            if route is None:
                continue
            current = replay_actions(prefix_game, route)
            actions = prefix_actions + route
            if current._state.value == "GAME_OVER":
                continue

            # Repeated touches are usually leave-and-return cycles around the
            # same changer. Try all inverse pairs and keep any exact replay that
            # reduces this attribute's debt.
            guard = 0
            while attr_value(avatar_state(current), attr) != target and guard < attr_modulus(current, attr) + 2:
                guard += 1
                before_debt = attr_debt(current, goal, attr)
                best = None
                for leave in MOVE_ACTIONS:
                    pair = (leave, inverse_action(leave))
                    trial = replay_actions(current, pair)
                    if trial._state.value == "GAME_OVER":
                        continue
                    after_debt = attr_debt(trial, goal, attr)
                    if after_debt < before_debt:
                        best = (pair, trial, after_debt)
                        break
                if best is None:
                    break
                pair, current, _after_debt = best
                actions += pair

            if attr_value(avatar_state(current), attr) != target:
                continue
            if total_debt(current, goal) >= total_debt(game, goal):
                continue
            if not has_goal_attempt_resources(current, goal):
                continue
            candidates.append((len(actions), goal_resource_shortfall(current, goal), current, actions, prefix_name, changer))

    if not candidates:
        return None, (), "no_static_set_attr"
    _length, _shortfall, out, actions, prefix_name, changer = min(candidates, key=lambda item: (item[1], item[0]))
    return out, actions, f"static_set_attr:{attr}:{prefix_name}:changer={changer}"


def has_goal_attempt_resources(game, goal: Goal) -> bool:
    route_len = static_path_length(game, goal.cell)
    if route_len is not None and game._step_counter_ui.current_steps >= 2 * route_len + 2:
        return True
    for pickup in clock_pickup_cells(game):
        to_pickup = static_path_actions(game, pickup)
        if to_pickup is None:
            continue
        after_pickup = replay_actions(game, to_pickup)
        if after_pickup._state.value == "GAME_OVER":
            continue
        if total_debt(after_pickup, goal) > total_debt(game, goal):
            continue
        if after_pickup._step_counter_ui.current_steps <= game._step_counter_ui.current_steps:
            continue
        to_goal = static_path_actions(after_pickup, goal.cell)
        if to_goal is None:
            continue
        if after_pickup._step_counter_ui.current_steps >= 2 * len(to_goal) + 2:
            return True
    return False


def goal_resource_shortfall(game, goal: Goal) -> int:
    route_len = static_path_length(game, goal.cell)
    if route_len is None:
        return 100
    shortfall = max(0, 2 * route_len + 2 - game._step_counter_ui.current_steps)
    if has_goal_attempt_resources(game, goal):
        shortfall = max(0, shortfall - 20)
    return shortfall


def choose_goal(game) -> Goal | None:
    candidates = goals(game)
    if not candidates:
        return None
    here = avatar_cell(game)
    return min(candidates, key=lambda goal: (total_debt(game, goal), manhattan_cells(here, goal.cell)))


def dynamic_signature(game, *, strict: bool = True) -> tuple:
    """Compact enough for search pruning, strict enough to distinguish phases."""
    sprites = []
    for sprite in game.current_level._sprites:
        tags = tuple(sorted(sprite.tags or ()))
        if not tags:
            continue
        if strict or set(tags) & {
            "rhsxkxzdjz",
            "soyhouuebz",
            "ttfwljgohq",
            "rjlbuycveu",
            "gbvqrjtaqo",
            "npxgalaybz",
            "sfqyzhzkij",
            "vjotnebuqo",
        }:
            sprites.append((sprite.name, sprite.x, sprite.y, getattr(sprite, "is_visible", True)))
    return (
        game.level_index,
        game._score,
        avatar_cell(game),
        avatar_state(game),
        tuple(game.lvrnuajbl),
        game._step_counter_ui.current_steps,
        tuple(sorted(sprites)),
    )


def progress_regressed(
    before,
    after,
    goal: Goal,
    *,
    protect_matched: bool,
    allow_debt_regress: bool = False,
) -> bool:
    if after._score < before._score:
        return True
    if after._state.value == "GAME_OVER":
        return True
    if after.level_index < before.level_index:
        return True
    if not allow_debt_regress and total_debt(after, goal) > total_debt(before, goal):
        return True
    if protect_matched:
        before_state = avatar_state(before)
        after_state = avatar_state(after)
        for attr in ATTR_TAGS:
            if attr_value(before_state, attr) == attr_value(goal.state, attr):
                if attr_value(after_state, attr) != attr_value(goal.state, attr):
                    return True
    return False


def _search(
    start_game,
    goal: Goal,
    condition: Callable[[object], bool],
    heuristic: Callable[[object], int | float],
    *,
    max_nodes: int,
    max_depth: int,
    weight: float = 4.0,
    protect_matched: bool = True,
    allow_debt_regress: bool = False,
) -> tuple[object | None, tuple[int, ...], SearchStats, str]:
    start_time = time.monotonic()
    stats = SearchStats()
    counter = itertools.count()
    start = clone_game(start_game)
    queue = [(weight * heuristic(start), 0, next(counter), (), start)]
    seen = {dynamic_signature(start): 0}

    while queue and stats.expanded < max_nodes:
        _priority, depth, _seq, actions, game = heapq.heappop(queue)
        stats.expanded += 1
        if condition(game):
            stats.elapsed = time.monotonic() - start_time
            return game, actions, stats, "ok"
        if depth >= max_depth:
            continue

        for action in MOVE_ACTIONS:
            nxt = clone_game(game)
            state, _score = step_no_render(nxt, action)
            stats.generated += 1
            if state == "GAME_OVER":
                stats.pruned_game_over += 1
                continue
            if progress_regressed(
                game,
                nxt,
                goal,
                protect_matched=protect_matched,
                allow_debt_regress=allow_debt_regress,
            ):
                stats.pruned_regress += 1
                continue
            sig = dynamic_signature(nxt)
            next_depth = depth + 1
            old_depth = seen.get(sig)
            if old_depth is not None and old_depth <= next_depth:
                stats.cache_hits += 1
                continue
            seen[sig] = next_depth
            next_actions = actions + (action,)
            priority = next_depth + weight * heuristic(nxt)
            heapq.heappush(queue, (priority, next_depth, next(counter), next_actions, nxt))

    stats.elapsed = time.monotonic() - start_time
    if queue:
        return None, (), stats, "node_budget"
    return None, (), stats, "exhausted"


def set_attr_option(
    game,
    goal: Goal,
    attr: str,
    *,
    max_nodes: int,
    max_depth: int,
) -> OptionResult:
    option = Option("SET_ATTR", attr=attr, target_value=attr_value(goal.state, attr), goal_index=goal.index)
    proposed_game, proposed_actions, proposed_reason = exact_set_attr_proposal(game, goal, attr)
    if proposed_game is not None:
        return OptionResult(
            option,
            True,
            proposed_game,
            proposed_actions,
            SearchStats(elapsed=0.0),
            proposed_reason,
        )
    target = attr_value(goal.state, attr)
    start_total = total_debt(game, goal)

    def condition(candidate) -> bool:
        return (
            attr_value(avatar_state(candidate), attr) == target
            and total_debt(candidate, goal) < start_total
            and candidate._step_counter_ui.current_steps >= min_safe_steps_to_goal(candidate, goal)
            and has_goal_attempt_resources(candidate, goal)
        )

    def heuristic(candidate) -> int:
        debt = attr_debt(candidate, goal, attr)
        if debt == 0:
            step_shortfall = max(0, min_safe_steps_to_goal(candidate, goal) - candidate._step_counter_ui.current_steps)
            route_len = static_path_length(candidate, goal.cell) or 99
            return total_debt(candidate, goal) * 3 + step_shortfall + goal_resource_shortfall(candidate, goal) + route_len
        step_shortfall = max(0, min_safe_steps_to_goal(candidate, goal) - candidate._step_counter_ui.current_steps)
        return (
            debt * 12
            + nearest_changer_distance(candidate, attr)
            + total_debt(candidate, goal) * 3
            + step_shortfall
            + goal_resource_shortfall(candidate, goal)
        )

    out, actions, stats, reason = _search(
        game,
        goal,
        condition,
        heuristic,
        max_nodes=max_nodes,
        max_depth=max_depth,
        weight=5.0,
        protect_matched=True,
    )
    return OptionResult(option, out is not None, out, actions, stats, reason)


def reduce_debt_option(
    game,
    goal: Goal,
    *,
    max_nodes: int,
    max_depth: int,
) -> OptionResult:
    option = Option("REDUCE_DEBT", goal_index=goal.index)
    start_total = total_debt(game, goal)

    def condition(candidate) -> bool:
        return total_debt(candidate, goal) < start_total and has_goal_attempt_resources(candidate, goal)

    def heuristic(candidate) -> int:
        debt = total_debt(candidate, goal)
        needed_distances = [
            nearest_changer_distance(candidate, attr)
            for attr in ATTR_TAGS
            if attr_debt(candidate, goal, attr)
        ]
        nearest = min(needed_distances) if needed_distances else manhattan_cells(avatar_cell(candidate), goal.cell)
        return debt * 14 + nearest + goal_resource_shortfall(candidate, goal)

    out, actions, stats, reason = _search(
        game,
        goal,
        condition,
        heuristic,
        max_nodes=max_nodes,
        max_depth=max_depth,
        weight=5.0,
        protect_matched=True,
    )
    return OptionResult(option, out is not None, out, actions, stats, reason)


def goal_attempt_option(
    game,
    goal: Goal,
    target_level: int,
    *,
    max_nodes: int,
    max_depth: int,
    protect_matched: bool = True,
    allow_debt_regress: bool = False,
) -> OptionResult:
    option = Option("GOAL_ATTEMPT", goal_index=goal.index)
    proposed_game, proposed_actions, proposed_reason = exact_goal_route_proposal(game, target_level, goal)
    if proposed_game is not None:
        return OptionResult(
            option,
            True,
            proposed_game,
            proposed_actions,
            SearchStats(elapsed=0.0),
            proposed_reason,
        )

    def condition(candidate) -> bool:
        return candidate._score >= target_level

    def heuristic(candidate) -> int:
        candidate_goal = goal
        if goal.index >= len(candidate.plrpelhym) or (
            goal.index < len(candidate.lvrnuajbl) and candidate.lvrnuajbl[goal.index]
        ):
            maybe = choose_goal(candidate)
            if maybe is None:
                return 0
            candidate_goal = maybe
        debt = total_debt(candidate, candidate_goal)
        distance = manhattan_cells(avatar_cell(candidate), candidate_goal.cell)
        return debt * 25 + distance

    for pickup in sorted(clock_pickup_cells(game), key=lambda cell: manhattan_cells(avatar_cell(game), cell)):
        to_pickup = static_path_actions(game, pickup)
        if to_pickup is None:
            continue
        after_pickup = replay_actions(game, to_pickup)
        if after_pickup._state.value == "GAME_OVER":
            continue
        if not allow_debt_regress and total_debt(after_pickup, goal) > total_debt(game, goal):
            continue
        out, tail_actions, stats, reason = _search(
            after_pickup,
            goal,
            condition,
            heuristic,
            max_nodes=max_nodes,
            max_depth=max_depth,
            weight=6.0,
            protect_matched=protect_matched,
            allow_debt_regress=allow_debt_regress,
        )
        if out is not None:
            return OptionResult(
                option,
                True,
                out,
                to_pickup + tail_actions,
                stats,
                f"clock_seeded:{pickup}:{reason}",
            )

    out, actions, stats, reason = _search(
        game,
        goal,
        condition,
        heuristic,
        max_nodes=max_nodes,
        max_depth=max_depth,
        weight=6.0,
        protect_matched=protect_matched,
        allow_debt_regress=allow_debt_regress,
    )
    if reason == "exhausted":
        reason = proposed_reason
    return OptionResult(option, out is not None, out, actions, stats, reason)


def option_label(result: OptionResult) -> str:
    opt = result.option
    if opt.kind == "SET_ATTR":
        name = f"SET_ATTR({opt.attr}={opt.target_value})"
    elif opt.kind == "REDUCE_DEBT":
        name = f"REDUCE_DEBT(goal={opt.goal_index})"
    else:
        name = f"{opt.kind}(goal={opt.goal_index})"
    return (
        f"{name}:success={result.success}:actions={len(result.actions)}:"
        f"expanded={result.stats.expanded}:generated={result.stats.generated}:"
        f"regress={result.stats.pruned_regress}:gameover={result.stats.pruned_game_over}:"
        f"cache={result.stats.cache_hits}:time={result.stats.elapsed:.2f}s:reason={result.reason}"
    )


def solve_game_with_options(
    game,
    target_level: int,
    *,
    option_budget: int = 12,
    attr_nodes: int = 1500,
    attr_depth: int = 65,
    goal_nodes: int = 9000,
    goal_depth: int = 90,
) -> tuple[object, tuple[int, ...], tuple[str, ...], list[str]]:
    current = clone_game(game)
    all_actions: list[int] = []
    option_logs: list[str] = []
    labels: list[str] = []

    for option_step in range(1, option_budget + 1):
        if current._score >= target_level:
            break
        goal = choose_goal(current)
        if goal is None:
            break
        debts = {attr: attr_debt(current, goal, attr) for attr in ATTR_TAGS}
        option_logs.append(
            f"option_step={option_step}:cell={avatar_cell(current)}:state={avatar_state(current)}:"
            f"goal={goal.index}@{goal.cell}:{goal.state}:debts={debts}:score={current._score}"
        )

        if sum(debts.values()) == 0:
            result = goal_attempt_option(
                current,
                goal,
                target_level,
                max_nodes=goal_nodes,
                max_depth=goal_depth,
                protect_matched=True,
            )
            option_logs.append(option_label(result))
            if not result.success:
                # If every matched-attribute path is blocked by moving hazards, allow
                # temporary attribute loss only as a last resort. Real completion is
                # still validated by the simulator.
                relaxed = goal_attempt_option(
                    current,
                    goal,
                    target_level,
                    max_nodes=goal_nodes,
                    max_depth=goal_depth,
                    protect_matched=False,
                    allow_debt_regress=True,
                )
                option_logs.append("relaxed:" + option_label(relaxed))
                result = relaxed
            if not result.success:
                break
        else:
            attrs = sorted(
                (attr for attr, debt in debts.items() if debt),
                key=lambda attr: (-debts[attr], nearest_changer_distance(current, attr)),
            )
            result = None
            for attr in attrs:
                proposed_game, proposed_actions, proposed_reason = exact_set_attr_proposal(current, goal, attr)
                candidate = OptionResult(
                    Option("SET_ATTR", attr=attr, target_value=attr_value(goal.state, attr), goal_index=goal.index),
                    proposed_game is not None,
                    proposed_game,
                    proposed_actions,
                    SearchStats(elapsed=0.0),
                    proposed_reason,
                )
                option_logs.append(option_label(candidate))
                if candidate.success:
                    result = candidate
                    break
            if result is None:
                result = reduce_debt_option(
                    current,
                    goal,
                    max_nodes=attr_nodes,
                    max_depth=attr_depth,
                )
                option_logs.append(option_label(result))
                if not result.success:
                    break

        labels.append(option_label(result))
        all_actions.extend(result.actions)
        current = result.game

    return current, tuple(all_actions), tuple(labels), option_logs


def run_level(level: int, *, steps: int = 240) -> LevelResult:
    start_time = time.monotonic()
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make("ls20")
    env.reset()
    env._game.set_level(level - 1)
    initial_score = env._game._score
    target_score = initial_score + 1
    planned_game, actions, options, log = solve_game_with_options(env._game, target_score)
    replay_actions = actions[:steps]
    final_state = "NOT_FINISHED"
    levels_completed = env._game._score
    for action in replay_actions:
        frame = env.step(GameAction.from_id(action))
        final_state = frame.state.value
        levels_completed = frame.levels_completed
        if final_state in {"WIN", "GAME_OVER"} or levels_completed >= target_score:
            break
    solved = levels_completed >= target_score or final_state == "WIN"
    log.append(
        f"planned_score={planned_game._score}:planned_state={planned_game._state.value}:"
        f"planned_cell={avatar_cell(planned_game)}:planned_avatar={avatar_state(planned_game)}"
    )
    log.append(
        f"replay_score={levels_completed}:replay_state={final_state}:"
        f"replay_cell={avatar_cell(env._game)}:replay_avatar={avatar_state(env._game)}"
    )
    return LevelResult(
        level=level,
        solved=solved,
        levels_completed=levels_completed,
        final_state=final_state,
        actions=tuple(replay_actions),
        options=options,
        log=log,
        elapsed=time.monotonic() - start_time,
    )


def run_sequence(levels: int, *, steps_per_level: int = 360) -> list[LevelResult]:
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make("ls20")
    env.reset()
    results = []
    for target_score in range(1, levels + 1):
        start_time = time.monotonic()
        planned_game, actions, options, log = solve_game_with_options(env._game, target_score)
        replay_actions = actions[:steps_per_level]
        final_state = "NOT_FINISHED"
        levels_completed = env._game._score
        for action in replay_actions:
            frame = env.step(GameAction.from_id(action))
            final_state = frame.state.value
            levels_completed = frame.levels_completed
            if final_state in {"WIN", "GAME_OVER"} or levels_completed >= target_score:
                break
        solved = levels_completed >= target_score or final_state == "WIN"
        log.append(
            f"planned_score={planned_game._score}:planned_state={planned_game._state.value}:"
            f"planned_cell={avatar_cell(planned_game)}:planned_avatar={avatar_state(planned_game)}"
        )
        log.append(
            f"replay_score={levels_completed}:replay_state={final_state}:"
            f"replay_cell={avatar_cell(env._game)}:replay_avatar={avatar_state(env._game)}"
        )
        results.append(
            LevelResult(
                level=target_score,
                solved=solved,
                levels_completed=levels_completed,
                final_state=final_state,
                actions=tuple(replay_actions),
                options=options,
                log=log,
                elapsed=time.monotonic() - start_time,
            )
        )
        if not solved:
            break
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()

    if args.sequential:
        results = run_sequence(args.levels, steps_per_level=args.steps)
    else:
        level_list = [args.level] if args.level is not None else list(range(1, args.levels + 1))
        results = [run_level(level, steps=args.steps) for level in level_list]
    print("ls20 option-level simulator planner", flush=True)
    solved = 0
    for result in results:
        solved += int(result.solved)
        print(
            f"level={result.level} solved={result.solved} steps={len(result.actions)} "
            f"state={result.final_state} levels_completed={result.levels_completed} "
            f"options={len(result.options)} time={result.elapsed:.2f}s",
            flush=True,
        )
        for line in result.log[:40]:
            print(f"  {line}")
    print(f"solved={solved}/{len(results)}")


if __name__ == "__main__":
    main()
