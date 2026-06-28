"""Lesson 0006 companion: from reflex to plan — BFS over a learned map.

Run from the repo root:  uv run python teach/lessons/0006-systematic-explorer.py

Lesson 0005 left you with a wall-aware *wanderer*: it never re-bumps a wall,
but its movement is still uniform-random. It explores the map by luck. This
lesson gives it a brain stem upgrade — DELIBERATION:

    1. Remember not just walls, but every successful move -> a movement GRAPH.
    2. Find the FRONTIER: cells you've reached that still have an untried exit.
    3. BFS the graph to the nearest frontier and walk there on purpose.

The new skill is breadth-first search. Everything else is review (your 0004
perception, your 0005 wall memory). The payoff is measured as COVERAGE: how
much of the reachable map you map per budget, vs the random baseline.

Honest spoiler (we decoded ls20.py together): this still clears ~0 levels.
A door only opens when your shape+color+rotation match it. Mapping the room
is necessary but not sufficient — that gate is lesson 0007. Watch the
coverage column soar and the levels column stay flat: that's the exact shape
of "necessary but not sufficient."
"""

import importlib.util
import random
from collections import deque
from types import SimpleNamespace

import numpy as np

import arc_agi
from arc_agi.base import OperationMode
from arcengine import GameAction

# Your verified perception stack from lesson 0004 — never rewrite working eyes.
_spec = importlib.util.spec_from_file_location(
    "perception", "teach/lessons/0004-perception.py"
)
perception = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(perception)

BASELINES = [22, 123, 73, 84, 96, 192, 186]
EPISODES = 5
MAX_ACTIONS = 5000
# Level 1's room is only ~36 cells, so 5000 steps is enough for even a random
# walk to map it. Planning's edge is SPEED, not final coverage — so we snapshot
# how much is mapped after a tight budget, where the difference is stark.
COVERAGE_AT = 150


def player_pos(grid: np.ndarray) -> tuple | None:
    """Robust position from lesson 0005: (row, col) tuple, or None."""
    try:
        return perception.find_player(grid)
    except ValueError:
        return None


# ----------------------------------------------------------------------------
# Exercise 1 — BFS, the heart of the lesson.
# graph maps pos -> {action_id: neighbour_pos}. Find the SHORTEST sequence of
# action ids that walks from `start` to ANY position in `goals`.
#   - return [] if start is already a goal
#   - return None if no goal is reachable
# Classic textbook BFS: a queue of (pos, path), a visited set, expand in order.
# ----------------------------------------------------------------------------
def bfs(graph: dict, start: tuple, goals: set) -> list | None:
    if start in goals:
        return []
    seen = {start}
    queue = deque([(start, [])])
    while queue:
        pos, path = queue.popleft()
        for action, nxt in graph.get(pos, {}).items():
            if nxt in seen:
                continue
            new_path = path + [action]
            if nxt in goals:
                return new_path
            seen.add(nxt)
            queue.append((nxt, new_path))
    return None


# ----------------------------------------------------------------------------
# Exercise 2 — the frontier.
# A frontier cell is a KNOWN position that still has at least one available
# action we have never tried there. Those are the edges of the map — the only
# places where new ground can be discovered. Return the set of such positions.
# `tried` maps pos -> set of action ids already attempted there.
# ----------------------------------------------------------------------------
def frontier_positions(tried: dict, available: list) -> set:
    avail = set(available)
    return {pos for pos, acts in tried.items() if avail - acts}


# ----------------------------------------------------------------------------
# The agent. Three memories, all keyed by player position:
#   moves[pos][action] = resulting pos   (the movement graph — BFS runs on it)
#   blocked[pos]       = set of wall actions
#   tried[pos]         = set of attempted actions  (frontier = tried but open)
# ----------------------------------------------------------------------------
class SystematicExplorer:
    def __init__(self):
        self.moves: dict[tuple, dict[int, tuple]] = {}
        self.blocked: dict[tuple, set[int]] = {}
        self.tried: dict[tuple, set[int]] = {}

    def choose(self, frame) -> int:
        grid = np.array(frame.frame[-1], dtype=np.int8)
        pos = player_pos(grid)
        if pos is None:
            return random.choice(frame.available_actions)

        # 1. Local frontier: an untried action right here? Expand it now.
        untried = [a for a in frame.available_actions if a not in self.tried.get(pos, set())]
        if untried:
            return random.choice(untried)

        # 2. Remote frontier: BFS the known graph to the nearest open cell.
        goals = frontier_positions(self.tried, frame.available_actions) - {pos}
        path = bfs(self.moves, pos, goals)
        if path:
            return path[0]

        # 3. Map fully explored (or frontier walled off): wander.
        return random.choice(frame.available_actions)

    def observe(self, prev, action_id, nxt, same_level):
        if not same_level:
            self.moves, self.blocked, self.tried = {}, {}, {}
            return
        pos0 = player_pos(prev)
        if pos0 is None:
            return
        self.tried.setdefault(pos0, set()).add(action_id)
        pos1 = player_pos(nxt)
        changed = perception.change_mask(prev, nxt).sum()
        if pos1 is not None and pos0 != pos1:
            self.moves.setdefault(pos0, {})[action_id] = pos1
        elif 0 < changed < 100 and pos0 == pos1:
            # A real bump twitches a few px (the budget-bar tick). A startup
            # no-op changes ZERO px — excluding 0 fixes the spawn poisoning
            # you diagnosed in lesson 0005.
            self.blocked.setdefault(pos0, set()).add(action_id)


class RandomAgent:
    def choose(self, frame) -> int:
        return random.choice(frame.available_actions)

    def observe(self, prev, action_id, nxt, same_level):
        pass


# ----------------------------------------------------------------------------
# Harness — GIVEN. Adds COVERAGE (unique cells visited) to the 0005 metrics.
# ----------------------------------------------------------------------------
def run_episode(arc, agent, max_actions: int) -> dict:
    env = arc.make("ls20")
    frame = env.reset()
    level_actions = [0] * 7
    level = 0
    actions = 0
    visited: set = set()
    coverage_at = 0

    while actions < max_actions and frame.state.value != "WIN":
        if frame.state.value == "GAME_OVER":
            frame = env.reset()
        else:
            prev_grid = np.array(frame.frame[-1], dtype=np.int8)
            prev_levels = frame.levels_completed
            action_id = agent.choose(frame)
            frame = env.step(GameAction.from_id(action_id))
            if frame.frame and frame.state.value not in ("GAME_OVER", "WIN"):
                nxt_grid = np.array(frame.frame[-1], dtype=np.int8)
                same = frame.levels_completed == prev_levels
                if same:
                    p1 = player_pos(nxt_grid)
                    if p1 is not None:
                        visited.add(p1)
                agent.observe(prev_grid, action_id, nxt_grid, same)
        actions += 1
        if actions == COVERAGE_AT:
            coverage_at = len(visited)
        level_actions[min(level, 6)] += 1
        level = max(level, frame.levels_completed)

    scores = [
        min((BASELINES[i] / level_actions[i]) ** 2, 1.15)
        if i < level and level_actions[i]
        else 0.0
        for i in range(7)
    ]
    if frame.state.value == "WIN":
        scores[6] = min((BASELINES[6] / level_actions[6]) ** 2, 1.15)
    game_score = sum((i + 1) * s for i, s in enumerate(scores)) / 28

    return {
        "won": frame.state.value == "WIN",
        "levels": level,
        "actions": actions,
        "game_score": game_score,
        "coverage": len(visited),
        "coverage_at": coverage_at or len(visited),
    }


# ----------------------------------------------------------------------------
# Check harness — instant feedback, no environment needed. Do not edit below.
# ----------------------------------------------------------------------------
def _synthetic_grid(player_row=20, player_col=10):
    g = np.full((64, 64), 3, dtype=np.int8)
    g[player_row : player_row + 2, player_col : player_col + 5] = 12  # orange cap
    g[player_row + 2 : player_row + 5, player_col : player_col + 5] = 9  # blue body
    return g


def _check(name, fn, expect_fn):
    try:
        result = fn()
    except NotImplementedError:
        print(f"  [ ] {name}: not implemented yet")
        return False
    except Exception as e:
        print(f"  [✗] {name}: raised {type(e).__name__}: {e}")
        return False
    ok, detail = expect_fn(result)
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    # A tiny 4-cell corridor graph:  A -(4)-> B -(4)-> C -(1)-> D
    A, B, C, D = (0, 0), (0, 5), (0, 10), (-5, 10)
    graph = {A: {4: B}, B: {4: C, 3: A}, C: {1: D, 3: B}, D: {2: C}}

    def fn_bfs():
        return (
            bfs(graph, A, {D}),       # shortest route across the corridor
            bfs(graph, A, {A}),       # already there
            bfs(graph, A, {(9, 9)}),  # unreachable
        )

    def chk_bfs(r):
        far, here, none = r
        ok = far == [4, 4, 1] and here == [] and none is None
        return ok, f"A->D {far} (expected [4,4,1]); A->A {here}; unreachable {none}"

    def fn_frontier():
        tried = {A: {1, 2, 3, 4}, B: {4}, C: {1, 2, 3, 4}}
        return frontier_positions(tried, [1, 2, 3, 4])

    def chk_frontier(r):
        ok = r == {B}
        return ok, f"open cells {sorted(r)} (expected [{B}] — only B has untried actions)"

    def fn_choose_local():
        agent = SystematicExplorer()
        g = _synthetic_grid()
        pos = player_pos(g)
        agent.tried = {pos: {1, 2, 3}}  # only action 4 untried here
        frame = SimpleNamespace(frame=[g.tolist()], available_actions=[1, 2, 3, 4])
        return {agent.choose(frame) for _ in range(25)}

    def chk_choose_local(r):
        ok = r == {4}
        return ok, f"untried-here picks {sorted(r)} (expected [4])"

    def fn_observe():
        agent = SystematicExplorer()
        g = _synthetic_grid()
        moved = _synthetic_grid(player_col=15)  # same row, 5 cols right = action 4
        pos0 = player_pos(g)
        agent.observe(g, 4, moved, True)         # a real move -> graph edge
        edge = agent.moves.get(pos0, {}).get(4) == player_pos(moved)
        agent.observe(g, 2, g.copy(), True)      # startup no-op: 0 px changed
        spawn_clean = pos0 not in agent.blocked  # must NOT be called a wall
        bump = _synthetic_grid()
        bump[61, 30] = 11                        # real bump: a few px twitch
        agent.observe(g, 2, bump, True)
        walled = 2 in agent.blocked.get(pos0, set())
        return edge, spawn_clean, walled

    def chk_observe(r):
        edge, spawn_clean, walled = r
        ok = edge and spawn_clean and walled
        return ok, (
            f"move->edge: {edge}, startup no-op NOT walled: {spawn_clean}, "
            f"real bump walled: {walled}"
        )

    results = [
        _check("Exercise 1: bfs", fn_bfs, chk_bfs),
        _check("Exercise 2: frontier_positions", fn_frontier, chk_frontier),
        _check("Exercise 3: choose expands local frontier", fn_choose_local, chk_choose_local),
        _check("Exercise 4: observe builds graph, no spawn poison", fn_observe, chk_observe),
    ]
    return all(results)


if __name__ == "__main__":
    print("Offline checks (synthetic graph + frames, instant):\n")
    if not _run_checks():
        print("\nFix the checks above, then run again for the live comparison.")
        raise SystemExit(1)

    print(
        f"\nAll checks pass. Live comparison on ls20 — write your prediction"
        f"\nfirst: after only {COVERAGE_AT} steps, how much of the room will"
        f"\nplanned search have mapped vs random, and will it clear levels?\n"
    )
    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    for make_agent in (RandomAgent, SystematicExplorer):
        name = make_agent.__name__
        print(f"{name}: {EPISODES} episodes x {MAX_ACTIONS} actions")
        for ep in range(EPISODES):
            agent = make_agent()
            r = run_episode(arc, agent, MAX_ACTIONS)
            print(
                f"  episode {ep + 1}: levels={r['levels']}/7  "
                f"coverage@{COVERAGE_AT}={r['coverage_at']} cells  "
                f"final={r['coverage']} cells  score={r['game_score']:.2%}"
            )
        print()

    print(
        f"Compare coverage@{COVERAGE_AT} — that's what planning bought you:"
        "\nthe whole room mapped in a fraction of the steps. The FINAL column"
        "\nconverges (random gets there eventually) — speed is the win."
        "\nLevels stay flat: mapping the room is NECESSARY but not SUFFICIENT."
        "\nThe door is gated on matching shape+color+rotation — that's lesson"
        "\n0007. Bring your prediction and your explanation of the gap."
    )
