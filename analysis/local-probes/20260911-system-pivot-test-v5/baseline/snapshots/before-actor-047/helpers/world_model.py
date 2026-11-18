
"""World model for the grid game.

Facts (steps 1-7):
- agent: 5x3 solid color-9 block (rows a..a+2, cols c..c+4). UP/DOWN: a -=/+2.
  LEFT/RIGHT: c -=/+5.
- trail: color 12, 2 rows x 5 cols, set from background (4) on entry, cleared
  back to background on exit. Observed placement (a-3, a-2) x [c, c+4] for
  vertical moves.
- bar: color 11, rows 61-62, shrinking from the LEFT end by 2 cells per move.
- All other cells static.

Horizontal-move trail placement is NOT yet confirmed; the model currently
mirrors the vertical rule (2 rows x 5 cols at the new position) and is being
tested by the LEFT probe.
"""
import numpy as np

C_BG = 4
C_BAR = 11
C_AGENT = 9
C_TRAIL = 12


def encode(grid):
    g = grid
    z = {"grid": g.tolist()}
    r, cl = np.where(g == C_TRAIL)
    if len(r):
        z["trail"] = [int(r.min()), int(r.max()), int(cl.min()), int(cl.max())]
    else:
        z["trail"] = None
    r9, c9 = np.where(g == C_AGENT)
    agent = None
    for rr, cc in zip(r9, c9):
        a0, c0 = int(rr), int(cc)
        if a0 + 3 > 64 or c0 + 5 > 64:
            continue
        if (g[a0:a0+3, c0:c0+5] == C_AGENT).all():
            agent = [a0, c0]
            break
    z["agent"] = agent
    r, cl = np.where(g == C_BAR)
    z["bar"] = [int(r.min()), int(r.max()), int(cl.min()), int(cl.max())] if len(r) else None
    return z


def _set(z, r0, r1, c0, c1, val):
    for i in range(r0, r1):
        for j in range(c0, c1):
            if 0 <= i < 64 and 0 <= j < 64:
                z["grid"][i][j] = val


def _clear_trail_cells(z):
    t = z.get("trail")
    if t:
        tr0, tr1, tc0, tc1 = t
        for i in range(tr0, tr1 + 1):
            for j in range(tc0, tc1 + 1):
                if z["grid"][i][j] == C_TRAIL:
                    z["grid"][i][j] = C_BG


def predict(z, action):
    z = {"grid": [list(r) for r in z["grid"]], "bar": z["bar"], "agent": z["agent"]}
    z["trail"] = z.get("trail")
    agent = z["agent"]
    if agent is None:
        return z
    a, c = agent
    horizontal = False
    if action == "UP":
        na, nc = a - 2, c
    elif action == "DOWN":
        na, nc = a + 2, c
    elif action == "LEFT":
        na, nc = a, c - 5
        horizontal = True
    elif action == "RIGHT":
        na, nc = a, c + 5
        horizontal = True
    else:
        return z
    _clear_trail_cells(z)
    _set(z, a, a + 3, c, c + 5, C_BG)
    _set(z, na, na + 3, nc, nc + 5, C_AGENT)
    z["agent"] = [na, nc]
    # trail is ALWAYS at (na-2, na-1) x (nc, nc+4) — 2 rows above block top
    tr0 = na - 2
    if tr0 >= 0 and tr0 + 1 < 64:
        _set(z, tr0, tr0 + 2, nc, nc + 5, C_TRAIL)
        z["trail"] = [tr0, tr0 + 1, nc, nc + 4]
    else:
        z["trail"] = None
    if z["bar"] is not None:
        r0, r1, c0, c1 = z["bar"]
        for j in range(c0, c0 + 2):
            for i in range(r0, r1 + 1):
                z["grid"][i][j] = C_BG
        z["bar"] = [r0, r1, c0 + 2, c1] if c0 + 2 <= c1 else None
    return z
