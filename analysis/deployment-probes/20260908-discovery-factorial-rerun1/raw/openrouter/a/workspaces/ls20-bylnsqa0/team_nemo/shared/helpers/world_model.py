"""World model for the 64x64 grid game (provisional — dynamics unverified).

Z_SCHEMA
--------
player      : list[[r,c]]  sprite cells (colors 0 and 1), global coords
anchor      : [r,c]        top-left of the 3x3 sprite bbox
pattern     : list[[r_off,c_off,color]]  sprite pattern relative to anchor
items12     : list[[r,c]]  color-12 cells (light-green block)
items9      : list[[r,c]]  color-9 cells (light-blue blocks, incl. boxes)
play        : {r0,c0,rows}  subgrid rows r0..r0+26, cols c0..c0+40 as hex strings
level       : int (carried from previous z; encode cannot see it)

predict: 1-cell step of the sprite in the action direction (sub-hypothesis M1,
UNVERIFIED). A step is allowed iff every target cell of the sprite footprint is
walkable. Walkable := current color in {3,4} (red ground / dark void) — UNVERIFIED.
If blocked -> no_impact (player unchanged).
"""
import numpy as np

PLAY_R0, PLAY_C0, PLAY_H, PLAY_W = 24, 14, 27, 41
WALKABLE = {3, 4}          # UNVERIFIED; red ground + dark void
SOLID = {5, 8, 9, 11, 12}  # borders, bars, items

DIRECTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


def _hexify(arr):
    return ["".join(format(int(v), "x") for v in row) for row in arr]


def encode(grid):
    g = np.asarray(grid)
    z = {
        "player": [],
        "anchor": None,
        "pattern": [],
        "items12": [],
        "items9": [],
        "play": None,
        "level": 0,
    }
    ys, xs = np.where((g == 0) | (g == 1))
    if len(ys):
        r0, c0 = int(ys.min()), int(xs.min())
        z["player"] = [[int(r), int(c)] for r, c in zip(ys, xs)]
        z["anchor"] = [r0, c0]
        z["pattern"] = [
            [int(r - r0), int(c - c0), int(g[r, c])]
            for r, c in zip(ys, xs)
        ]
    for col, key in [(12, "items12"), (9, "items9")]:
        yy, xx = np.where(g == col)
        z[key] = [[int(r), int(c)] for r, c in zip(yy, xx)]
    z["play"] = {
        "r0": PLAY_R0,
        "c0": PLAY_C0,
        "rows": _hexify(g[PLAY_R0:PLAY_R0 + PLAY_H, PLAY_C0:PLAY_C0 + PLAY_W]),
    }
    return z


def _decode_play(z):
    p = z["play"]
    arr = np.array(
        [[int(ch, 16) for ch in row] for row in p["rows"]], dtype=np.int64
    )
    return arr, p["r0"], p["c0"]


def _clear_sprite(arr, r0, c0, z, ground=3):
    """Erase sprite footprint in the play subgrid, restoring ground color."""
    for r, c in z["player"]:
        rr, cc = r - r0, c - c0
        if 0 <= rr < arr.shape[0] and 0 <= cc < arr.shape[1]:
            arr[rr, cc] = ground


def _place_sprite(arr, r0, c0, anchor, pattern):
    for dr, dc, col in pattern:
        r, c = anchor[0] + dr, anchor[1] + dc
        rr, cc = r - r0, c - c0
        if 0 <= rr < arr.shape[0] and 0 <= cc < arr.shape[1]:
            arr[rr, cc] = col
        else:
            return False  # sprite would leave the play subgrid
    return True


def is_blocked(z, anchor_new):
    """True if moving sprite to anchor_new collides with a non-walkable cell."""
    arr, r0, c0 = _decode_play(z)
    for dr, dc, _ in z["pattern"]:
        r, c = anchor_new[0] + dr, anchor_new[1] + dc
        if not (0 <= r < 64 and 0 <= c < 64):
            return True
        rr, cc = r - r0, c - c0
        if not (0 <= rr < arr.shape[0] and 0 <= cc < arr.shape[1]):
            return True  # outside play subgrid -> unknown, treat as blocked
        if arr[rr, cc] not in WALKABLE:
            return True
    return False


def predict(z, action):
    z = dict(z)  # shallow copy; we rebuild fields we touch
    if action not in DIRECTIONS:
        return z  # RESET etc.: unknown, leave as is
    dr, dc = DIRECTIONS[action]
    a = [z["anchor"][0] + dr, z["anchor"][1] + dc]
    if is_blocked(z, a):
        return z  # no_impact: unchanged
    # move: shift sprite footprint by (dr,dc)
    new_player = [[r + dr, c + dc] for r, c in z["player"]]
    arr, r0, c0 = _decode_play(z)
    _clear_sprite(arr, r0, c0, z)
    ok = _place_sprite(arr, r0, c0, a, z["pattern"])
    if not ok:
        return z
    z["player"] = new_player
    z["anchor"] = a
    z["play"] = {"r0": r0, "c0": c0, "rows": _hexify(arr)}
    return z


def expected_class(z, action):
    """Class to put in predicted_path: 'move' if predict says the sprite moves."""
    if action not in DIRECTIONS:
        return "no_impact"
    dr, dc = DIRECTIONS[action]
    a = [z["anchor"][0] + dr, z["anchor"][1] + dc]
    return "no_impact" if is_blocked(z, a) else "move"
