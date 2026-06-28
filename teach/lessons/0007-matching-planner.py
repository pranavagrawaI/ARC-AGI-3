"""Lesson 0007 companion: the matching planner — the first agent that clears a
level on PURPOSE.

Run from the repo root:  uv run python teach/lessons/0007-matching-planner.py

Lesson 0006 proved that mapping the room is necessary but not sufficient: the
goal door only opens when the player's shape+color+rotation match it (you read
this in ls20.py yourself). This lesson builds the agent that satisfies that
gate deliberately.

ONE new idea, three small skills:
  - taps_needed: cyclers CYCLE one-way, so reaching a target state is modular
    arithmetic, not detection — (required - current) mod N.
  - match-then-route: fix each attribute at its cycler, THEN path to the door
    while AVOIDING cyclers (re-entering one would un-match you).
  - reach a blocked cell: the door isn't walkable until matched, so you can't
    BFS *onto* it — BFS to a neighbour and take the final step manually.

The cheat, stated plainly: this planner reads the map and goal straight from
the engine (`read_oracle`). Perceiving them from pixels is lesson 0008 — one
new skill at a time, exactly as we built perception on a recording first.

Honest result: this clears LEVEL 1 every run (deterministic), turning the
0.12% luck of lesson 0006 into a full 1.15 score. Level 2+ introduces a new
constraint — a 42-move budget per life, refuelled only by collecting coins —
which turns the problem into survival-routing. That's the next frontier.
"""

import importlib.util
from collections import deque

import arc_agi
from arc_agi.base import OperationMode
from arcengine import GameAction

# Reuse the systematic explorer (lesson 0006) purely as the baseline to beat.
_spec = importlib.util.spec_from_file_location(
    "explorer", "teach/lessons/0006-systematic-explorer.py"
)
explorer_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(explorer_mod)

BASELINES = [22, 123, 73, 84, 96, 192, 186]
EPISODES = 3
MAX_ACTIONS = 5000
S = 5  # one cell = 5 sprite units (you measured this in lesson 0004)
# action id -> (dx, dy) in sprite coords (decoded from ls20.py step())
DELTA = {1: (0, -S), 2: (0, S), 3: (-S, 0), 4: (S, 0)}
CYCLER_TAGS = {"shape": "ttfwljgohq", "color": "soyhouuebz", "rotation": "rhsxkxzdjz"}


# ----------------------------------------------------------------------------
# Exercise 1 — taps_needed. A cycler advances one attribute by +1 (mod N) per
# entry; it cannot go backwards. How many entries turn `cur` into `req`?
# ----------------------------------------------------------------------------
def taps_needed(cur: int, req: int, n: int) -> int:
    return (req - cur) % n


# ----------------------------------------------------------------------------
# Exercise 2 — bfs (your lesson 0006 routine, with an `avoid` set so we can
# forbid stepping onto cycler cells). Shortest action path start -> any goal.
# ----------------------------------------------------------------------------
def bfs(graph: dict, start: tuple, goals: set, avoid=frozenset()) -> list | None:
    if start in goals:
        return []
    seen, q = {start}, deque([(start, [])])
    while q:
        pos, path = q.popleft()
        for action, nxt in graph.get(pos, {}).items():
            if nxt in seen or nxt in avoid:
                continue
            if nxt in goals:
                return path + [action]
            seen.add(nxt)
            q.append((nxt, path + [action]))
    return None


# ----------------------------------------------------------------------------
# Exercise 3 — build_graph. Flood the 5-unit lattice from `start`; a cell is a
# node unless it is a wall or in `blocked` (we block door cells: not walkable
# until matched). Edges are labelled with the action that traverses them.
# ----------------------------------------------------------------------------
def build_graph(walls: set, start: tuple, blocked: set) -> dict:
    G, seen, q = {}, {start}, deque([start])
    while q:
        c = q.popleft()
        G[c] = {}
        for action, (dx, dy) in DELTA.items():
            nb = (c[0] + dx, c[1] + dy)
            if nb in walls or nb in blocked:
                continue
            G[c][action] = nb
            if nb not in seen:
                seen.add(nb)
                q.append(nb)
    return G


# ----------------------------------------------------------------------------
# The oracle (the cheat). Reads the live puzzle from the engine. Lesson 0008
# replaces every line of this with perception over the 64x64 frame.
# ----------------------------------------------------------------------------
def read_oracle(game) -> dict:
    walls = {(s.x, s.y) for s in game.current_level.get_sprites_by_tag("ihdgageizm")}
    cyclers = {
        k: [(s.x, s.y) for s in game.current_level.get_sprites_by_tag(t)]
        for k, t in CYCLER_TAGS.items()
    }
    unsolved = [i for i in range(len(game.plrpelhym)) if not game.lvrnuajbl[i]]
    doors = {
        i: {
            "cell": (game.plrpelhym[i].x, game.plrpelhym[i].y),
            "req": {
                "shape": game.ldxlnycps[i],
                "color": game.yjdexjsoa[i],
                "rotation": game.ehwheiwsk[i],
            },
        }
        for i in unsolved
    }
    return {
        "pos": (game.gudziatsk.x, game.gudziatsk.y),
        "attrs": {
            "shape": game.fwckfzsyc,
            "color": game.hiaauhahz,
            "rotation": game.cklxociuu,
        },
        "lens": {
            "shape": len(game.ijessuuig),
            "color": len(game.tnkekoeuk),
            "rotation": len(game.dhksvilbb),
        },
        "walls": walls,
        "cyclers": cyclers,
        "doors": doors,
    }


# ----------------------------------------------------------------------------
# The agent — reactive: every step it re-reads the world and emits ONE action.
# Reactivity is what makes it survive budget resets (which teleport the player
# to spawn AND reset its attributes): it simply replans from wherever it is.
# ----------------------------------------------------------------------------
class MatchingPlanner:
    def __init__(self, game):
        self.game = game

    def choose(self, frame) -> int:
        o = read_oracle(self.game)
        if not o["doors"]:
            return frame.available_actions[0]
        all_cyclers = {c for v in o["cyclers"].values() for c in v}
        door_cells = {d["cell"] for d in o["doors"].values()}
        G = build_graph(o["walls"], o["pos"], blocked=door_cells)

        def approach(door_cell):
            """Shortest (path, step-in action) to a door neighbour, no cyclers."""
            best = None
            for action, (dx, dy) in DELTA.items():
                nb = (door_cell[0] - dx, door_cell[1] - dy)
                if nb not in G:
                    continue
                p = bfs(G, o["pos"], {nb}, avoid=all_cyclers - {o["pos"]})
                if p is not None and (best is None or len(p) < len(best[0])):
                    best = (p, action)
            return best

        # commit to the nearest unsolved door
        i, dt = min(
            ((i, approach(d["cell"])) for i, d in o["doors"].items()),
            key=lambda t: 1e9 if t[1] is None else len(t[1][0]),
        )
        req = o["doors"][i]["req"]

        # 1. fix the first mismatched attribute at its cycler
        for name in ("shape", "color", "rotation"):
            if o["attrs"][name] == req[name] or not o["cyclers"][name]:
                continue
            C = o["cyclers"][name][0]
            if o["pos"] == C:  # on it already: step off so we can re-enter (= +1 tap)
                off = next((a for a, nb in G[C].items() if nb not in all_cyclers), None)
                return off if off is not None else frame.available_actions[0]
            path = bfs(G, o["pos"], {C})  # stepping ONTO C taps it once
            return path[0] if path else frame.available_actions[0]

        # 2. matched: walk to the door (avoiding cyclers) and step in
        if dt is None:
            return frame.available_actions[0]
        path, step_in = dt
        return step_in if not path else path[0]

    def observe(self, *a):  # the oracle is always live; nothing to learn
        pass


# ----------------------------------------------------------------------------
# Harness — GIVEN. Same scoring as lessons 0003/0006.
# ----------------------------------------------------------------------------
def run_episode(arc, make_agent, max_actions: int) -> dict:
    env = arc.make("ls20")
    frame = env.reset()
    agent = make_agent(env)
    level_actions = [0] * 7
    level = 0
    actions = 0
    while actions < max_actions and frame.state.value != "WIN":
        if frame.state.value == "GAME_OVER":
            frame = env.reset()
            agent = make_agent(env)
        else:
            frame = env.step(GameAction.from_id(agent.choose(frame)))
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
        "levels": level,
        "game_score": game_score,
        "l1_actions": level_actions[0] if level >= 1 else None,
    }


# ----------------------------------------------------------------------------
# Check harness — instant feedback, no environment needed. Do not edit below.
# ----------------------------------------------------------------------------
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
    def fn_taps():
        return (taps_needed(3, 0, 4), taps_needed(0, 3, 4), taps_needed(2, 2, 4))

    def chk_taps(r):
        ok = r == (1, 3, 0)
        return ok, f"3->0={r[0]} (exp 1), 0->3={r[1]} (exp 3), 2->2={r[2]} (exp 0)"

    # A 3x3 grid of cells with a wall in the middle; door at top-right corner.
    walls = {(5, 5)}
    start = (0, 0)

    def fn_graph():
        G = build_graph(walls, start, blocked={(10, 0)})  # block one cell
        return (5, 5) not in G and (10, 0) not in G and (0, 5) in G

    def chk_graph(r):
        return r, f"wall & blocked excluded, open cell present: {r}"

    def fn_bfs():
        G = build_graph(walls, start, blocked=set())
        straight = bfs(G, (0, 0), {(10, 0)})  # detour around wall? no, clear row
        around = bfs(G, (0, 0), {(10, 10)})  # must route around (5,5)
        avoided = bfs(
            G, (0, 0), {(10, 10)}, avoid={(5, 0), (0, 5)}
        )  # both near routes blocked
        return straight, around, avoided

    def chk_bfs(r):
        straight, around, avoided = r
        ok = (
            straight == [4, 4]
            and around is not None
            and len(around) == 4
            and avoided is None
        )
        return (
            ok,
            f"straight {straight}; around len {None if around is None else len(around)}; over-avoided {avoided}",
        )

    def fn_approach():
        # door at (10,10); a neighbour (5,10) is reachable, step-in action is 4 (+x)
        G = build_graph(set(), (0, 0), blocked={(10, 10)})
        nb = (5, 10)
        return nb in G and bfs(G, (0, 0), {nb}) is not None

    def chk_approach(r):
        return r, f"door neighbour reachable while door itself is blocked: {r}"

    return all(
        [
            _check("Exercise 1: taps_needed (modular)", fn_taps, chk_taps),
            _check(
                "Exercise 2: build_graph excludes walls/blocked", fn_graph, chk_graph
            ),
            _check("Exercise 3: bfs shortest + avoid", fn_bfs, chk_bfs),
            _check(
                "Exercise 4: reach a blocked door via neighbour",
                fn_approach,
                chk_approach,
            ),
        ]
    )


if __name__ == "__main__":
    print("Offline checks (synthetic maze, instant):\n")
    if not _run_checks():
        print("\nFix the checks above, then run again for the live comparison.")
        raise SystemExit(1)

    print(
        "\nAll checks pass. Live comparison on ls20 — predict first: which"
        "\nlevels will the planner clear, and what level-1 score (vs 0006's"
        "\n~0.12% luck)?\n"
    )
    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)

    def make_explorer(env):
        return explorer_mod.SystematicExplorer()

    for name, factory in (
        ("SystematicExplorer (0006)", make_explorer),
        ("MatchingPlanner (0007)", MatchingPlanner),
    ):
        print(f"{name}: {EPISODES} episodes x {MAX_ACTIONS} actions")
        for ep in range(EPISODES):
            r = run_episode(arc, factory, MAX_ACTIONS)
            l1 = f"  L1 in {r['l1_actions']} moves" if r["l1_actions"] else ""
            print(
                f"  episode {ep + 1}: levels={r['levels']}/7  "
                f"score={r['game_score']:.2%}{l1}"
            )
        print()

    print(
        "The planner clears level 1 deterministically and near-optimally"
        "\n(baseline 22). The levels column finally moved ON PURPOSE."
        "\nIt stalls at level 2: the 42-move budget resets position AND"
        "\nattributes, and level 2's solve (par 123) needs coins to refuel."
        "\nSurvival-routing through coins is the next frontier; perceiving"
        "\nthe map+goal from pixels (replacing read_oracle) is lesson 0008."
    )
