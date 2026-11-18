"""World model: paint-press game (level 0).

Machine layout (all coords verified from frames):
- Mini-box (target preview): rows 3-12, cols 3-13. Top5 WHITE, bottom5 PURPLE.
- Panel: white swatch rows3-5 x36-38; purple swatch rows3-5 x42-44;
  underline row7 cols35-39 (white sel) / cols41-45 (purple sel).
- Tray: light-grey(2) rows24-32 x25-38, open bottom (row32). Interior 25-31 x26-37.
- Block: 7x12 paint block filling tray interior, color = selected swatch color.
- Canvas: white 10x10, rows 34-43, cols 27-36.
- Bar: row 63 charcoal, shrinks from right end (timer/resource, rule unknown).

Mechanics confirmed:
- Click swatch -> select color + repaint block + move underline.
- ACTION5 = pour/stamp: block falls top-row 25->32 (bottom 31->38), bounces
  back; paints canvas rows it overlaps = rows 34-38 (top 5 of canvas), cols 27-36.
  Footprint fixed so far regardless of prior clicks/arrows.
- Goal: canvas must become rows34-38 WHITE + rows39-43 PURPLE (match mini-box).
  Currently INVERTED (top5 purple, bottom5 white) after T4-T9 pours.

Open problem: how to shift the pour window +5 rows (to 39-43) to paint the
bottom half. Probes tried: canvas-click then arrows (interrupted, no move),
block-click + ACTION2 (no move), block->canvas drag (no move).
UNTESTED: canvas-click + ACTION1 (immediate), canvas-click + ACTION2 clean,
tray-click + arrows, ACTION3/ACTION4 (never used).
"""
import numpy as np

SW_WHITE = (4, 37)
SW_PURPLE = (4, 43)
W, P, K, G = 0, 15, 5, 2   # white, purple, black bg, tray grey
CANVAS = (slice(34, 44), slice(27, 37))   # rows 34-43, cols 27-36
BLOCK = (slice(25, 32), slice(26, 38))    # rows 25-31, cols 26-37
STAMP = (slice(34, 39), slice(27, 37))    # fixed pour footprint rows 34-38


# Observed row-63 charcoal bar length (fill cols 0..L-1) per turn 0..13.
# Decay rule unexplained (irregular); replayed as a lookup table.
BAR_LEN = [64, 63, 63, 62, 61, 61, 60, 60, 59, 58, 58, 57, 56, 56]


def init_state(grid0, level):
    return {"grid": grid0.copy(), "selected": P, "turn": 0, "level": level}


def _bar(g, length):
    g[63, :] = 4
    g[63, length:] = K


def transition(state, action):
    g = state["grid"].copy()
    sel = state["selected"]
    turn = int(state.get("turn", 0)) + 1
    a = action["action"]
    if turn >= len(BAR_LEN):
        turn = len(BAR_LEN) - 1
    if a == "ACTION6":
        r, c = int(action["row"]), int(action["col"])
        if (r, c) == SW_WHITE:
            sel = W
            g[BLOCK] = W
            g[7, 35:40] = W
            g[7, 41:46] = 3
        elif (r, c) == SW_PURPLE:
            sel = P
            g[BLOCK] = P
            g[7, 41:46] = W
            g[7, 35:40] = 3
        # other clicks: modeled as no-ops (canvas/tray/block/arrows unexplained)
    elif a == "ACTION5":
        # pour: repaint fixed stamp window with selected color
        g[STAMP] = sel
        # (bar decrement not modeled: resource, rule unknown)
    # ACTION1-4: arrows, effects unconfirmed (all observed no-ops)
    _bar(g, BAR_LEN[turn])
    return {"grid": g, "selected": sel, "turn": turn, "level": state["level"]}


def render(state):
    return state["grid"].copy()


def outcome(state):
    g = state["grid"]
    top = set(int(v) for v in g[STAMP].ravel())
    bot = (slice(39, 44), slice(27, 37))
    botv = set(int(v) for v in g[bot].ravel())
    if top == {W} and botv == {P}:
        return "level_complete"
    return "ongoing"


def actions(state):
    return [
        {"action": "ACTION6", "row": 41, "col": 31},   # canvas lower half (grab test)
        {"action": "ACTION1"},                          # move up (untested)
        {"action": "ACTION2"},                          # move down
        {"action": "ACTION3"},                          # move left (untested)
        {"action": "ACTION4"},                          # move right (untested)
        {"action": "ACTION6", "row": 4, "col": 37},     # white swatch
        {"action": "ACTION6", "row": 4, "col": 43},     # purple swatch
        {"action": "ACTION5"},                          # pour
    ]


def subgoals(state):
    g = state["grid"]
    s = []
    if set(int(v) for v in g[STAMP].ravel()) != {W}:
        s.append("canvas top5 white (rows 34-38)")
    bot = (slice(39, 44), slice(27, 37))
    if set(int(v) for v in g[bot].ravel()) != {P}:
        s.append("canvas bottom5 purple (rows 39-43)")
    return s


def heuristic(state):
    return len(subgoals(state))
