
"""World model for the click-grid game. encode(grid)->z, predict(z, action)->z'."""
import numpy as np

Z_SCHEMA = {
    "player": "(x,y) center of the 11 (bbb) cluster",
    "goal_center": "(cx,cy) center of the 9-diamond",
    "goal_radius": "r of the 9-diamond (|dx|+|dy|<=r)",
    "cells": "dict (r,c)->color, all non-5 cells (grid background)",
    "status": "NOT_FINISHED / WIN / GAME_OVER",
    "levels_completed": int,
    "step": int,
}

def _cells(g, bg=5):
    m = g != bg
    d = {}
    for r, c in zip(*np.where(m)):
        d[(int(r), int(c))] = int(g[r, c])
    return d

def _cluster_center(d, color=11):
    pts = [k for k, v in d.items() if v == color]
    if not pts:
        return None
    rows = [p[0] for p in pts]; cols = [p[1] for p in pts]
    return (float(np.mean(cols)), float(np.mean(rows)))

def _diamond(d, color=9):
    # ignore the status band at row 1 (wide horizontal run)
    pts = [k for k, v in d.items() if v == color and k[0] > 3]
    if not pts:
        return None, None
    rows = [p[0] for p in pts]; cols = [p[1] for p in pts]
    cy = (min(rows) + max(rows)) / 2.0
    cx = (min(cols) + max(cols)) / 2.0
    # radius = max manhattan distance from center
    r = max(abs(p[0] - cy) + abs(p[1] - cx) for p in pts)
    return (cx, cy), r

def encode(grid_rows, status="NOT_FINISHED", levels_completed=0, step=0):
    g = np.array([[int(ch, 16) for ch in row] for row in grid_rows], dtype=np.int64)
    d = _cells(g)
    player = _cluster_center(d, 11)
    gcenter, gr = _diamond(d, 9)
    return {"cells": d, "player": player, "goal_center": gcenter,
            "goal_radius": gr, "status": status,
            "levels_completed": levels_completed, "step": step}

# ---- mechanics ----
# Clicks act on cell (x=col, y=row):
#   - if cell == player color 11 (within player cluster): player +1 row (DOWN)
#   - if cell == 9: the diamond grows by 1 layer (radius +1), 9 overwrites hit cells
#   - else: no-impact
# Player wins by entering the goal diamond? (untested: what happens at touch)
# Clicking the 5-border/edges: unknown.
MOVE_COST = 1  # one CLICK per +1 row

def z_to_cells(z):
    return dict(z["cells"])

def predict(z, action):
    """action = ('CLICK', x, y). Returns (z', expect_label)."""
    import copy
    z2 = copy.deepcopy(z)
    kind = action[0]
    if kind == "RESET":
        return None, "level_up"  # unknown; flag for re-encoding
    if kind != "CLICK":
        return z2, "no_impact"
    x, y = action[1], action[2]
    d = z_to_cells(z2)
    player = z2["player"]
    gcenter = z2["goal_center"]
    gr = z2["goal_radius"]
    # player cluster cells: all cells colored 11
    pcells = {k for k, v in d.items() if v == 11}
    pc = pcells if player is None else pcells
    if (y, x) in pc:
        # move player down by 1
        new = {}
        for (r, c), v in d.items():
            if v == 11:
                new[(r + 1, c)] = 11
            else:
                new[(r, c)] = v
        # leave old spot: if not covered, it becomes 0 (observed: 16,32 went 11->0)
        z2["cells"] = new
        z2["player"] = (player[0], player[1] + 1)
        # did we reach the goal diamond?
        if gcenter is not None:
            # check player center manhattan distance
            if abs(z2["player"][1] - gcenter[1]) + abs(z2["player"][0] - gcenter[0]) <= gr:
                return z2, "level_up"  # guess; needs testing
        return z2, "move"
    if d.get((y, x)) == 9:
        # diamond grows by 1 layer
        cy, cx = gcenter
        for dy in range(-gr - 1, gr + 2):
            for dx in range(-gr - 1, gr + 2):
                if abs(dx) + abs(dy) == gr + 1:
                    r, c = int(cy + dy), int(cx + dx)
                    if 0 <= r < 64 and 0 <= c < 64:
                        z2["cells"][(r, c)] = 9
        z2["goal_radius"] = gr + 1
        return z2, "move"
    return z2, "no_impact"

def plan_to_goal(z):
    """Greedy: click own player cell (player_x, player_y) repeatedly until
    player row reaches goal row. Returns list of actions."""
    px, py = z["player"]
    gy = z["goal_center"][1]
    acts = []
    y = py
    while y < gy - 1:  # stop 1 above center row; touch check at predict
        acts.append(("CLICK", int(px), int(y)))
        y += 1
    return acts
