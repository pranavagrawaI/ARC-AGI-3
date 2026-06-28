"""Lesson 0008 companion: the first LEARNING agent — discover what you control.

Run from the repo root:  uv run python teach/lessons/0008-object-controllability.py

This is the turn the whole mission pivots on (MISSION rewritten; LR 0009/0010).
Everything before this HANDED the agent its rules. Lesson 0004's `find_player`
told it "the player is color 12 over color 9" — a fact WE read out of the game.
Lesson 0007's planner read the win condition straight from `ls20.py`. None of
that is allowed on the real benchmark: an ARC-AGI-3 agent gets no instructions
and must learn each game's rules by playing it.

So we throw the color prior away and keep only TWO Core-Knowledge priors —
the ones ARC says every solver may assume:

  GIVEN  (objectness):  the world is a set of objects = connected regions of one
                        colour on a background. We never name a colour.
  GIVEN  (agentness):   *some* object responds to my actions. We don't know which.

  LEARNED:              WHICH object(s) I control, and what each action does to
                        them — induced purely from the experiment loop
                        act -> diff the objects -> attribute consistent motion.

Why object-centric (the architectural bet, see RESOURCES): if state is a SET of
objects and "what do I control?" is a per-object question, then a game with one
controlled entity and a game with five are the SAME mechanism at a different
count. A pixel policy would relearn from scratch; this transfers for free. The
offline checks prove exactly that — the identical code finds 1 avatar in a
1-avatar world and 2 in a 2-avatar world, no change.

Honest scope: this lesson only finds the avatar and its action-semantics. It
does NOT yet learn the cyclers/door/win-condition (those are dynamics induction,
lesson 0009). The avatar's signature is assumed stable here — true during early
level-1 exploration, before any attribute-cycling. That limit motivates 0009.
"""

import random
from collections import defaultdict, deque

import numpy as np

import arc_agi
from arc_agi.base import OperationMode
from arcengine import GameAction

CELL = 5  # one cell = 5 sprite units (you measured this in lesson 0004)
ACTIONS = (1, 2, 3, 4)
DIR = {1: "up", 2: "down", 3: "left", 4: "right"}


# ----------------------------------------------------------------------------
# Exercise 1 — segment_objects.  The objectness prior, made concrete.
# An object is a maximal 4-connected region of ONE colour, excluding the
# background (the most frequent colour). Return a list of objects, each a dict
# {color, size, centroid:(row,col)}. No colour is ever named — fully generic.
# ----------------------------------------------------------------------------
def segment_objects(grid: np.ndarray, background: int | None = None) -> list[dict]:
    if background is None:
        vals, counts = np.unique(grid, return_counts=True)
        background = int(vals[counts.argmax()])
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    objects = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or grid[r, c] == background:
                continue
            color = int(grid[r, c])
            cells, q = [], deque([(r, c)])
            seen[r, c] = True
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < H
                        and 0 <= nx < W
                        and not seen[ny, nx]
                        and grid[ny, nx] == color
                    ):
                        seen[ny, nx] = True
                        q.append((ny, nx))
            ys = [p[0] for p in cells]
            xs = [p[1] for p in cells]
            n = len(cells)
            objects.append(
                {"color": color, "size": n, "centroid": (sum(ys) / n, sum(xs) / n)}
            )
    return objects


def signature(obj: dict) -> tuple:
    """Translation-invariant identity. (colour, size) survives any move, so an
    object that keeps it is a rigid body; one that changes it has transformed."""
    return (obj["color"], obj["size"])


# ----------------------------------------------------------------------------
# Exercise 2 — match_objects.  Tie each object in `prev` to the same object in
# `nxt` (same signature, nearest centroid) and report its (d_row, d_col) shift.
# A jump bigger than `max_move` is a teleport (budget reset / level change), not
# a step — drop it so it can't poison the motion statistics.
# ----------------------------------------------------------------------------
def match_objects(prev_objs: list, nxt_objs: list, max_move: int = 2 * CELL) -> list:
    by_sig = defaultdict(list)
    for o in nxt_objs:
        by_sig[signature(o)].append(o)
    used, matches = set(), []
    for p in prev_objs:
        best, best_d = None, 1e9
        for o in by_sig[signature(p)]:
            if id(o) in used:
                continue
            d = abs(o["centroid"][0] - p["centroid"][0]) + abs(
                o["centroid"][1] - p["centroid"][1]
            )
            if d < best_d:
                best, best_d = o, d
        if best is not None and best_d <= max_move:
            used.add(id(best))
            dr = best["centroid"][0] - p["centroid"][0]
            dc = best["centroid"][1] - p["centroid"][1]
            matches.append((signature(p), (dr, dc)))
    return matches


# ----------------------------------------------------------------------------
# Exercise 3 — the learner.  Accumulate (signature, action) -> displacements.
# An object is CONTROLLED if, for at least one action, it reliably translates:
# a consistent non-trivial median shift. Static scenery (walls, doors, coins)
# never translates -> never flagged, even if it shares the avatar's colour.
# Objects that morph (the shrinking budget bar) change signature -> never
# accumulate a stable sample -> never flagged. Both filtered for free.
# ----------------------------------------------------------------------------
class ControllabilityLearner:
    def __init__(self):
        self.samples: dict = defaultdict(
            lambda: defaultdict(list)
        )  # sig -> action -> [(dr,dc)]

    def observe(self, prev_grid: np.ndarray, action: int, nxt_grid: np.ndarray):
        for sig, disp in match_objects(
            segment_objects(prev_grid), segment_objects(nxt_grid)
        ):
            self.samples[sig][action].append(disp)

    def controlled(self, min_samples=2, min_move=CELL * 0.5, consistency=0.6) -> dict:
        result = {}
        for sig, per_action in self.samples.items():
            motion, is_ctrl = {}, False
            for a, ds in per_action.items():
                if len(ds) < min_samples:
                    continue
                arr = np.array(ds)
                med = np.median(arr, axis=0)
                motion[a] = (round(float(med[0]), 1), round(float(med[1]), 1))
                mag = abs(med[0]) + abs(med[1])
                close = float(np.mean(np.abs(arr - med).sum(axis=1) <= 2.0))
                if mag >= min_move and close >= consistency:
                    is_ctrl = True
            if is_ctrl:
                result[sig] = motion
        return result


# ----------------------------------------------------------------------------
# Live discovery on ls20 — GIVEN. Random-walk a budget of steps and learn.
# ----------------------------------------------------------------------------
def discover_on_ls20(arc, steps: int) -> dict:
    env = arc.make("ls20")
    frame = env.reset()
    learner = ControllabilityLearner()
    prev = np.array(frame.frame[-1], dtype=np.int8)
    for _ in range(steps):
        a = random.choice(frame.available_actions)
        frame = env.step(GameAction.from_id(a))
        if frame.state.value in ("GAME_OVER", "WIN") or not frame.frame:
            frame = env.reset()
            prev = np.array(frame.frame[-1], dtype=np.int8)
            continue
        nxt = np.array(frame.frame[-1], dtype=np.int8)
        learner.observe(prev, a, nxt)
        prev = nxt
    return learner.controlled()


# ----------------------------------------------------------------------------
# Offline checks — synthetic worlds, instant, no engine. Do not edit below.
# The toy "game": N avatars (2x2 blocks) all translate by the action; one 3x3
# block is static scenery. This is where the transfer claim is proven.
# ----------------------------------------------------------------------------
DELTAS = {1: (-CELL, 0), 2: (CELL, 0), 3: (0, -CELL), 4: (0, CELL)}


def _scene(avatars: list, static=(4, 4)) -> np.ndarray:
    g = np.zeros((64, 64), dtype=np.int8)  # background 0
    sr, sc = static
    g[sr : sr + 3, sc : sc + 3] = 99  # static scenery (colour 99, size 9)
    for i, (r, c) in enumerate(avatars):
        g[r : r + 2, c : c + 2] = 7 + i  # avatar i (colour 7+i, size 4)
    return g


def _simulate(n_avatars: int, steps=40, seed=0) -> ControllabilityLearner:
    rng = random.Random(seed)
    learner = ControllabilityLearner()
    avs = [[15 + 12 * i, 25] for i in range(n_avatars)]  # spaced so they never collide
    prev = _scene([tuple(a) for a in avs])
    for _ in range(steps):
        a = rng.choice(ACTIONS)
        dr, dc = DELTAS[a]
        for av in avs:
            av[0] = min(max(av[0] + dr, 8), 50)  # stay off the borders
            av[1] = min(max(av[1] + dc, 8), 50)
        nxt = _scene([tuple(a) for a in avs])
        learner.observe(prev, a, nxt)
        prev = nxt
    return learner


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


def _semantics_ok(motion: dict) -> bool:
    """Every sampled action maps to its correct unit direction."""
    signs = {a: (np.sign(d[0]), np.sign(d[1])) for a, d in motion.items()}
    want = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    return all(signs.get(a) == want[a] for a in motion)


def _run_checks() -> bool:
    def fn_seg():
        g = _scene([(20, 20)])  # 1 avatar + 1 static block
        return segment_objects(g)

    def chk_seg(objs):
        sigs = sorted(signature(o) for o in objs)
        ok = sigs == [(7, 4), (99, 9)]  # background 0 excluded
        return ok, f"objects {sigs} (expected [(7,4),(99,9)], background dropped)"

    def fn_match():
        a = segment_objects(_scene([(20, 20)]))
        b = segment_objects(_scene([(20, 25)]))  # avatar moved +5 cols (right)
        return match_objects(a, b)

    def chk_match(m):
        d = dict(m)
        ok = d.get((7, 4)) == (0.0, 5.0) and d.get((99, 9)) == (0.0, 0.0)
        return ok, f"shifts {sorted(m)} (avatar (0,+5), static (0,0))"

    def fn_one():
        return _simulate(1).controlled()

    def chk_one(ctrl):
        ok = set(ctrl) == {(7, 4)} and _semantics_ok(ctrl[(7, 4)])
        learned = {DIR[a]: v for a, v in ctrl.get((7, 4), {}).items()}
        return (
            ok,
            f"controlled={sorted(ctrl)} (expected the avatar only); semantics {learned}",
        )

    def fn_two():
        return _simulate(2).controlled()

    def chk_two(ctrl):
        ok = set(ctrl) == {(7, 4), (8, 4)}  # SAME code, two avatars found
        return ok, f"controlled={sorted(ctrl)} (expected BOTH avatars, static excluded)"

    return all(
        [
            _check("Exercise 1: segment_objects (objectness)", fn_seg, chk_seg),
            _check("Exercise 2: match_objects (motion)", fn_match, chk_match),
            _check("Exercise 3: discover the 1 avatar + semantics", fn_one, chk_one),
            _check("Transfer: SAME code finds 2 avatars", fn_two, chk_two),
        ]
    )


if __name__ == "__main__":
    print("Offline checks (synthetic worlds, instant):\n")
    if not _run_checks():
        print("\nFix the checks above, then run again for live discovery on ls20.")
        raise SystemExit(1)

    print(
        "\nAll checks pass — including the transfer test: ONE algorithm, no edits,"
        "\nfound 1 avatar in a 1-avatar world and 2 in a 2-avatar world. That is"
        "\nthe architectural prior earning its keep.\n"
        "\nNow live on ls20. Predict first: with the colour prior THROWN AWAY,"
        "\nwhich object(s) will it discover that you control, and will it recover"
        "\nthe up/down/left/right map you hand-built in lesson 0004?\n"
    )
    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    controlled = discover_on_ls20(arc, steps=400)

    if not controlled:
        print("Discovered nothing — widen the budget or loosen thresholds.")
    else:
        print("Discovered avatar (signature -> learned action map):")
        for sig, motion in sorted(controlled.items()):
            sem = {DIR[a]: v for a, v in sorted(motion.items())}
            print(f"  colour {sig[0]:>2}, size {sig[1]:>2}:  {sem}")
    print(
        "\nNo colour was named. The agent learned its own body and its own"
        "\ncontrols from interaction alone — the first thing on this benchmark"
        "\nit was not simply told. Bring the discovered table to your teacher:"
        "\nnext (0009) we learn the RULES — what the cyclers and the door do."
    )
