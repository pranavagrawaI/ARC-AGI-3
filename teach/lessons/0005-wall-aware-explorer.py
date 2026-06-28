"""Lesson 0005 companion: the wall-aware explorer — perception becomes agent state.

Run from the repo root:  uv run python teach/lessons/0005-wall-aware-explorer.py

Your lesson-0004 functions stop being exercises today: this script imports
YOUR find_player and change_mask and wires them into a live agent. The agent
interface grows one method:

    choose(frame) -> action id          (as before, but now it may look)
    observe(prev_grid, action_id, next_grid, same_level)   (NEW: it may learn)

The harness calls observe() after every step — the same (prev, action, next,
same_level) tuple shape as lesson 0004's triples, but arriving live.

Exercises 1-3 build the naive explorer. Exercise 4 is a trap that the naive
rule walks into; the check tells you when you've fallen in. Before running
the comparison at the end, write down your prediction: will avoiding known
walls clear level 1 more often than random?
"""

import importlib.util
import random
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


# ----------------------------------------------------------------------------
# Exercise 1 — robust position.
# find_player raises ValueError when the player isn't visible. An agent's
# perception must degrade gracefully, never crash the loop.
# Return the player's (row, col) as a tuple of ints, or None if not found.
# ----------------------------------------------------------------------------
def player_pos(grid: np.ndarray) -> tuple | None:
    try:
        return perception.find_player(grid)
    except ValueError:
        return None


# ----------------------------------------------------------------------------
# The agent. blocked maps position -> set of action ids known to be walls.
# ----------------------------------------------------------------------------
class WallAwareExplorer:
    def __init__(self):
        self.blocked: dict[tuple, set[int]] = {}

    # ------------------------------------------------------------------
    # Exercise 2 — choose: random among actions NOT known to be blocked here.
    # If everything is blocked (or the player isn't visible), fall back to
    # all available actions — never return nothing.
    # ------------------------------------------------------------------
    def choose(self, frame) -> int:
        grid = np.array(frame.frame[-1], dtype=np.int8)
        pos = player_pos(grid)
        if pos is None:
            return random.choice(frame.available_actions)
        available = [
            a for a in frame.available_actions if a not in self.blocked.get(pos, set())
        ]
        return (
            random.choice(available)
            if available
            else random.choice(frame.available_actions)
        )

    # ------------------------------------------------------------------
    # Exercise 3 — observe: learn walls. Naive rule to start with:
    #   if the player's position didn't change, that action is a wall here.
    # Clear all memory when the level changes (new map, old walls are lies).
    #
    # Exercise 4 — the naive rule is WRONG in one important case. The check
    # below feeds you a step where the player stayed put but ~90 pixels
    # changed elsewhere on screen. A real wall bump changes almost nothing
    # (run the comparison and study what a bump looks like — or recall
    # exercise 1 of lesson 0004). Use change_mask to block only on real bumps.
    # ------------------------------------------------------------------
    def observe(
        self, prev: np.ndarray, action_id: int, nxt: np.ndarray, same_level: bool
    ):
        if not same_level:
            self.blocked = {}
            return
        last_pos = player_pos(prev)
        next_pos = player_pos(nxt)
        if last_pos is None:
            return
        if perception.change_mask(prev, nxt).sum() < 100 and last_pos == next_pos:
            # Player didn't move, so this action is a wall
            self.blocked.setdefault(last_pos, set()).add(action_id)


class RandomAgent:
    def choose(self, frame) -> int:
        return random.choice(frame.available_actions)

    def observe(self, prev, action_id, nxt, same_level):
        pass


# ----------------------------------------------------------------------------
# Harness — GIVEN. Same scoring as lesson 0003, plus the observe() hook and
# bump-rate instrumentation.
# ----------------------------------------------------------------------------
def run_episode(arc, agent, max_actions: int) -> dict:
    env = arc.make("ls20")
    frame = env.reset()
    level_actions = [0] * 7
    level = 0
    actions = 0
    bumps = 0
    moves = 0

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
                    moves += 1
                    p0, p1 = player_pos(prev_grid), player_pos(nxt_grid)
                    if p0 is not None and p0 == p1:
                        bumps += 1
                agent.observe(prev_grid, action_id, nxt_grid, same)
        actions += 1
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
        "bump_rate": bumps / max(moves, 1),
    }


def audit_memory(agent) -> list:
    """Positions claiming 4 blocked directions cannot exist on a reachable map."""
    blocked = getattr(agent, "blocked", {})
    return [pos for pos, acts in blocked.items() if len(acts) >= 4]


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
    g = _synthetic_grid()
    empty = np.full((64, 64), 3, dtype=np.int8)

    def chk_pos(r):
        want = (21, 12)
        return r == want, f"player_pos = {r} (expected {want}); empty grid -> None"

    def fn_pos():
        if player_pos(empty) is not None:
            return "should be None on empty grid"
        return player_pos(g)

    def fn_choose():
        agent = WallAwareExplorer()
        agent.blocked = {(21, 12): {1, 2, 3}}
        frame = SimpleNamespace(frame=[g.tolist()], available_actions=[1, 2, 3, 4])
        picks = {agent.choose(frame) for _ in range(25)}
        agent.blocked = {(21, 12): {1, 2, 3, 4}}
        fallback = {agent.choose(frame) for _ in range(25)}
        return picks, fallback

    def chk_choose(r):
        picks, fallback = r
        ok = picks == {4} and len(fallback) >= 2
        return ok, (
            f"with 1/2/3 blocked picks {sorted(picks)} (expected [4]); "
            f"all-blocked fallback explores {sorted(fallback)}"
        )

    def fn_observe():
        agent = WallAwareExplorer()
        agent.observe(g, 2, g.copy(), True)  # true bump: nothing changed
        bumped = agent.blocked.get((21, 12), set()) == {2}
        agent.observe(g, 1, g.copy(), False)  # level change
        cleared = agent.blocked == {}
        return bumped, cleared

    def chk_observe(r):
        bumped, cleared = r
        ok = bumped and cleared
        return ok, f"bump recorded: {bumped}, cleared on level change: {cleared}"

    def fn_trap():
        agent = WallAwareExplorer()
        reset_frame = g.copy()
        reset_frame[58:62, 5:55] = 11  # move bar refills: big change, no movement
        agent.observe(g, 3, reset_frame, True)
        return agent.blocked

    def chk_trap(r):
        ok = r == {}
        return ok, (
            "level-reset step (player still, ~200 px changed) not mistaken for a wall"
            if ok
            else f"blocked {r} — but the player was teleported, not walled! "
            "Hint: how many pixels does a REAL bump change?"
        )

    results = [
        _check("Exercise 1: player_pos", fn_pos, chk_pos),
        _check("Exercise 2: choose avoids known walls", fn_choose, chk_choose),
        _check("Exercise 3: observe learns walls", fn_observe, chk_observe),
        _check("Exercise 4: don't trust 'didn't move'", fn_trap, chk_trap),
    ]
    return all(results)


if __name__ == "__main__":
    print("Offline checks (synthetic frames, instant):\n")
    if not _run_checks():
        print("\nFix the checks above, then run again for the live comparison.")
        raise SystemExit(1)

    print(
        "\nAll checks pass. Live comparison on ls20 — write your prediction"
        "\nfirst: will the explorer clear level 1 more often than random?\n"
    )
    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    for make_agent in (RandomAgent, WallAwareExplorer):
        name = make_agent.__name__
        print(f"{name}: {EPISODES} episodes x {MAX_ACTIONS} actions")
        for ep in range(EPISODES):
            agent = make_agent()
            r = run_episode(arc, agent, MAX_ACTIONS)
            ghosts = audit_memory(agent)
            print(
                f"  episode {ep + 1}: levels={r['levels']}/7  "
                f"bump_rate={r['bump_rate']:.0%}  score={r['game_score']:.2%}"
                + (f"  impossible cells: {ghosts}" if ghosts else "")
            )
        print()

    print(
        "Compare the bump_rate columns — that's what perception bought you."
        "\nCompare the levels columns — that's what it didn't. Bring both"
        "\nnumbers and your explanation of the gap to your teacher."
    )
