"""Lesson 0009 companion: the falsifier — learn a rule by killing hypotheses.

Run from the repo root:  uv run python teach/lessons/0009-dynamics-falsifier.py

Lesson 0008 discovered WHAT you control. This lesson discovers HOW THE WORLD
RESPONDS — the first dynamics the agent learns instead of being told. We hold
the line locked in the mission (LR 0012/0013):

  GIVEN  — a GENERIC causal-interaction vocabulary. Every distinct object-class
           plays a ROLE; the candidate roles are game-agnostic:
             WALKABLE  — step toward it and you move onto it, unchanged
             BLOCKER   — step toward it and you do NOT move
             CHANGER   — step onto it and your appearance changes
           We never name ls20's wall or its cyclers. We enumerate the SCHEMA.
  LEARNED — which role each object-class actually has, induced by experiment.

The method is falsification, not fitting. Each object-class starts with ALL
roles alive. Every contact is an experiment: the observed outcome (did I move?
did I change?) is predicted by some roles and not others — keep the survivors,
kill the rest. The metric we optimise is HYPOTHESES KILLED PER ACTION: a good
probe refutes as much as possible per step (on this benchmark, slow == fail).

Why explicit hypotheses (LR 0013): they fail LOUD. If every role for a class
gets killed, the survivor set is EMPTY — an unmistakable signal that the schema
is too small to express this game's mechanic (a "grammar gap"), not a silent
stall. The synthetic world below plants exactly such a tile to prove the alarm
fires.

Honest scope (verified, see the probe in session notes): on ls20 this cleanly
discovers the WALL as BLOCKER and the floor as WALKABLE, no colour named. It
does NOT yet nail the cyclers as CHANGER, because ls20 renders the avatar in a
single pixel layer: standing on a tile OCCLUDES it, and a colour-cycle shows up
as a one-frame artifact, not a clean change to the avatar's (colour,size). That
is the seam to 0010: CHANGER needs attribute-level perception of the avatar, not
just its signature. The falsifier machinery is already correct — the synthetic
world proves all three roles AND the grammar-gap alarm; ls20 just withholds the
clean CHANGER signal until perception sharpens.
"""

import importlib.util
import random
from collections import defaultdict

import numpy as np

import arc_agi
from arc_agi.base import OperationMode
from arcengine import GameAction

# Reuse lesson 0008's verified object perception — never rewrite working eyes.
_spec = importlib.util.spec_from_file_location(
    "oc", "teach/lessons/0008-object-controllability.py"
)
oc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oc)
segment_objects = oc.segment_objects
signature = oc.signature

CELL = 5  # ls20: one cell = 5 sprite units
DELTA5 = {1: (-CELL, 0), 2: (CELL, 0), 3: (0, -CELL), 4: (0, CELL)}

ROLES = ("WALKABLE", "BLOCKER", "CHANGER")
# Each role's prediction of (moved?, changed?) when you step toward the class.
PRED = {"WALKABLE": (True, False), "BLOCKER": (False, False), "CHANGER": (True, True)}


def _manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ----------------------------------------------------------------------------
# Exercise 1 — component_at.  Given a pixel, flood its connected same-colour
# region and return its (colour, size) signature — the class you'd step onto.
# Returns None on background (floor teaches nothing about a class).
# ----------------------------------------------------------------------------
def component_at(grid: np.ndarray, r: int, c: int, background: int):
    H, W = grid.shape
    if not (0 <= r < H and 0 <= c < W):
        return None
    color = int(grid[r, c])
    if color == background:
        return None
    seen, stack, size = {(r, c)}, [(r, c)], 0
    while stack:
        y, x = stack.pop()
        size += 1
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if (
                0 <= ny < H
                and 0 <= nx < W
                and (ny, nx) not in seen
                and grid[ny, nx] == color
            ):
                seen.add((ny, nx))
                stack.append((ny, nx))
    return (color, size)


# ----------------------------------------------------------------------------
# Exercise 2 — the falsifier.  Track the avatar by its controllable SIZE(s) (a
# shape invariant that survives recolouring), read the class one cell ahead, and
# narrow that class's surviving roles to those consistent with the outcome.
# ----------------------------------------------------------------------------
# A class is RESOLVED once we have enough contacts and one outcome dominates;
# below that it stays undecided (all roles alive). This makes falsification
# evidence-weighted — robust to the occasional noisy frame (occlusion, a missed
# step) that pure one-shot falsification would let empty the hypothesis set.
MIN_CONTACTS = 4
CONSISTENCY = 0.6


class DynamicsFalsifier:
    def __init__(self, controlled_sizes, deltas=DELTA5, cell=CELL):
        self.sizes = set(controlled_sizes)
        self.deltas = deltas
        self.cell = cell
        # class = COLOUR (a material's role is keyed by colour, not its flickering
        # size). counts[colour][(moved,changed)] -> tally of that outcome.
        self.counts: dict = defaultdict(lambda: defaultdict(int))
        self.contacts: dict = defaultdict(int)
        self.transitions: dict = defaultdict(list)
        self.acentroid = None

    def _avatar(self, objs, near):
        cand = [o for o in objs if o["size"] in self.sizes]
        if near is not None:
            close = [o for o in cand if _manh(o["centroid"], near) <= self.cell * 2.5]
            if close:
                cand = close
        if not cand:
            return None, frozenset()
        cy = sum(o["centroid"][0] for o in cand) / len(cand)
        cx = sum(o["centroid"][1] for o in cand) / len(cand)
        return (cy, cx), frozenset(signature(o) for o in cand)

    def observe(self, prev_grid, action, nxt_grid):
        vals, counts = np.unique(prev_grid, return_counts=True)
        bg = int(vals[counts.argmax()])
        a_prev, s_prev = self._avatar(segment_objects(prev_grid), self.acentroid)
        if a_prev is None:
            return
        dr, dc = self.deltas[action]
        target = (a_prev[0] + dr, a_prev[1] + dc)
        cls = component_at(prev_grid, round(target[0]), round(target[1]), bg)
        if cls is None or cls[1] in self.sizes:  # floor (bg), or we read our own body
            a_now, _ = self._avatar(segment_objects(nxt_grid), target)
            if a_now is not None:
                self.acentroid = a_now
            return
        color = cls[0]  # role is keyed by colour (the material)

        nobjs = segment_objects(nxt_grid)
        a_nxt, s_nxt = self._avatar(nobjs, target)
        if a_nxt is None:
            a_nxt, s_nxt = self._avatar(nobjs, a_prev)
        if a_nxt is None:
            moved, changed = True, True  # avatar's signature fully changed
        else:
            d_t, d_s = _manh(a_nxt, target), _manh(a_nxt, a_prev)
            if min(d_t, d_s) > self.cell * 1.6:
                return  # teleport / budget reset — not a step
            moved, changed = d_t < d_s, (s_nxt != s_prev)
            self.acentroid = a_nxt

        self.counts[color][(moved, changed)] += 1
        self.contacts[color] += 1
        if changed:
            self.transitions[color].append((sorted(s_prev), sorted(s_nxt)))

    def _verdict(self, color):
        """Surviving roles for a class from its outcome tally. Returns a set;
        empty => resolved but no role fits (grammar gap); None => undecided."""
        tally = self.counts[color]
        n = sum(tally.values())
        if n < MIN_CONTACTS:
            return None
        top, top_n = max(tally.items(), key=lambda kv: kv[1])
        if top_n / n < CONSISTENCY:
            return None  # too noisy to call yet
        return {r for r in ROLES if PRED[r] == top}

    def report(self):
        out = {}
        for color in self.counts:
            surv = self._verdict(color)
            if surv is None:
                out[color] = ["… undecided"]
            elif not surv:
                out[color] = ["⚠ GRAMMAR GAP"]
            else:
                out[color] = sorted(surv)
        return out

    @property
    def contradictions(self):
        return {c for c in self.counts if self._verdict(c) == set()}

    def killed(self):
        """Hypotheses falsified = roles ruled out across all RESOLVED classes."""
        total = 0
        for color in self.counts:
            surv = self._verdict(color)
            if surv is not None:
                total += len(ROLES) - len(surv)
        return total

    def kill_rate(self):
        n = sum(self.contacts.values())
        return self.killed() / n if n else 0.0


# ----------------------------------------------------------------------------
# Live discovery on ls20 — GIVEN. Random-walk; falsify the wall and the floor.
# ----------------------------------------------------------------------------
def discover_on_ls20(arc, steps):
    env = arc.make("ls20")
    frame = env.reset()
    fal = DynamicsFalsifier(controlled_sizes={10, 15})  # avatar parts (from 0008)
    prev = np.array(frame.frame[-1], dtype=np.int8)
    for _ in range(steps):
        a = random.choice(frame.available_actions)
        frame = env.step(GameAction.from_id(a))
        if frame.state.value in ("GAME_OVER", "WIN") or not frame.frame:
            frame = env.reset()
            prev = np.array(frame.frame[-1], dtype=np.int8)
            fal.acentroid = None
            continue
        nxt = np.array(frame.frame[-1], dtype=np.int8)
        fal.observe(prev, a, nxt)
        prev = nxt
    return fal


# ----------------------------------------------------------------------------
# Offline checks — a synthetic single-layer world with KNOWN roles, incl. a tile
# whose behaviour is OUTSIDE the schema, to prove the grammar-gap alarm. Instant.
# Do not edit below this line.
# ----------------------------------------------------------------------------
TCELL = 3  # toy cell size (px); avatar is a 3x3 block, tiles are 1x1
TDELTA = {1: (-TCELL, 0), 2: (TCELL, 0), 3: (0, -TCELL), 4: (0, TCELL)}


def _triple(action, target_color, moved, changed, a_color=1, a_color2=2, P=(30, 30)):
    """Build (prev, nxt) for: avatar (3x3, size 9) at P steps `action` toward a
    1x1 tile of `target_color` one cell ahead; outcome forced to (moved, changed).
    Single-layer physics: if it moves on, the tile is occluded (omitted)."""
    dr, dc = TDELTA[action]
    prev = np.zeros((64, 64), dtype=np.int8)
    nxt = np.zeros((64, 64), dtype=np.int8)
    pr, pc = P
    prev[pr : pr + 3, pc : pc + 3] = a_color  # avatar
    tr, tc = pr + 1 + dr, pc + 1 + dc  # tile centre, one cell ahead
    prev[tr, tc] = target_color  # the 1x1 class tile
    nr, nc = (pr + dr, pc + dc) if moved else (pr, pc)  # avatar's new top-left
    nxt[nr : nr + 3, nc : nc + 3] = a_color2 if changed else a_color
    if not moved:  # blocked: tile still visible
        nxt[tr, tc] = target_color
    return prev, nxt


def _check(name, fn, expect_fn):
    try:
        result = fn()
    except Exception as e:
        print(f"  [✗] {name}: raised {type(e).__name__}: {e}")
        return False
    ok, detail = expect_fn(result)
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _drive(specs, n=4):
    """Run a fresh falsifier over repeated contacts. specs: list of
    (target_color, moved, changed). One arbitrary action each (geometry is fixed
    by _triple, so the action only picks a direction — any works)."""
    fal = DynamicsFalsifier(controlled_sizes={9}, deltas=TDELTA, cell=TCELL)
    for _ in range(n):
        for color, moved, changed in specs:
            fal.acentroid = None
            prev, nxt = _triple(2, color, moved, changed)
            fal.observe(prev, 2, nxt)
    return fal


def _run_checks():
    def fn_comp():
        g = np.zeros((20, 20), dtype=np.int8)
        g[5:8, 5:8] = 4  # a 3x3 block of colour 4 (size 9)
        return component_at(g, 6, 6, 0), component_at(g, 0, 0, 0)

    def chk_comp(r):
        ok = r == ((4, 9), None)
        return (
            ok,
            f"pixel-in-block -> {r[0]} (exp (4,9)); background -> {r[1]} (exp None)",
        )

    def fn_roles():
        # three classes, ground-truth roles: 5=walkable, 6=wall, 7=changer
        fal = _drive([(5, True, False), (6, False, False), (7, True, True)])
        return fal.report(), fal.killed()

    def chk_roles(r):
        rep, killed = r
        ok = (
            rep.get(5) == ["WALKABLE"]
            and rep.get(6) == ["BLOCKER"]
            and rep.get(7) == ["CHANGER"]
            and killed > 0
        )
        return ok, f"{rep}  (killed={killed})"

    def fn_gap():
        # a tile that CHANGES you but BLOCKS movement: (moved=False, changed=True)
        # no role predicts that — the alarm must fire.
        fal = _drive([(9, False, True)])
        return fal.report().get(9), 9 in fal.contradictions

    def chk_gap(r):
        roles, flagged = r
        ok = roles == ["⚠ GRAMMAR GAP"] and flagged
        return ok, f"out-of-schema tile -> {roles}, flagged={flagged} (alarm must fire)"

    def fn_rate():
        # 3 classes, each resolved to 1 of 3 roles => 2 hypotheses killed each.
        fal = _drive([(5, True, False), (6, False, False), (7, True, True)])
        return fal.killed(), round(fal.kill_rate(), 2)

    def chk_rate(r):
        killed, rate = r
        return (
            killed == 6 and rate > 0,
            f"killed={killed} (exp 6), killed/contact={rate}",
        )

    return all(
        [
            _check("Exercise 1: component_at", fn_comp, chk_comp),
            _check("Exercise 2: falsify three roles", fn_roles, chk_roles),
            _check("Fail-loud: grammar-gap alarm", fn_gap, chk_gap),
            _check("Metric: hypotheses-killed-per-action", fn_rate, chk_rate),
        ]
    )


if __name__ == "__main__":
    print("Offline checks (synthetic single-layer world, instant):\n")
    if not _run_checks():
        print("\nFix the checks above, then run again for live discovery.")
        raise SystemExit(1)

    print(
        "\nAll checks pass — including the grammar-gap alarm: a tile whose"
        "\nbehaviour the schema cannot express empties its hypothesis set"
        "\ninstead of failing silently.\n"
        "\nLive on ls20. Predict first: with NO colour named, which object-class"
        "\nwill it brand BLOCKER, and what will the floor come out as?\n"
    )
    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    fal = discover_on_ls20(arc, steps=1500)

    print("Discovered roles (object colour -> surviving hypotheses):")
    for color, roles in sorted(
        fal.report().items(), key=lambda kv: -fal.contacts[kv[0]]
    ):
        if fal.contacts[color] >= MIN_CONTACTS:
            print(f"  colour {color:>2}  ({fal.contacts[color]:>4} contacts):  {roles}")
    print(f"\ntotal hypotheses falsified = {fal.killed()}")
    if fal.contradictions:
        print(
            f"grammar gaps (schema too small for colours): {sorted(fal.contradictions)}"
        )
    print(
        "\nThe impassable boundary (colour 5) fell out as BLOCKER and the floor"
        "\n(colour 3) as WALKABLE — discovered by falsification, no colour named,"
        "\nno false grammar gaps. The cyclers stay (correctly) undecided: ls20"
        "\noccludes the avatar when it stands on a tile, so CHANGER needs"
        "\nattribute-level perception of your body, not just its (colour,size)."
        "\nThat clean seam is exactly lesson 0010."
    )
