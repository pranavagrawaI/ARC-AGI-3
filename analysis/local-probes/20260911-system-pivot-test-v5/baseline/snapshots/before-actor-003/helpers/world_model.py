
"""World model for the grid game.

Facts (steps 1-7):
- agent: 5x3 solid color-9 block (rows a..a+2, cols c..c+4). UP/DOWN: a -=/+2.
  LEFT/RIGHT: c -=/+5.
- trail: color 12, 2 rows x 5 cols, at rows a-3..a-2, cols c..c+4. Set from
  background (4) on entry; cleared back to background on exit.
- bar: color 11, rows 61-62, shrinking from the LEFT end by 2 cells per move.
- All other cells static.
"""
import numpy as np

C_BG = 4
C_BAR = 11
C_AGENT = 9
C_TRAIL = 12


def _agent_block(z):
    a, c = z["agent"]
    sub = z["grid"][a:a+3, c:c+5]
    if sub.size and not (sub == C_AGENT).all():
        return None
    return (a, c)


def encode(grid):
    z = {"grid": grid.tolist()}
    a, c = z["agent"]
    trail_rows = (a-3, a-2) if a >= 2 else None
    z["trail"] = list(trail_rows) + [c, c+5] if trail_rows else None
    r, cl = np.where(grid == C_BAR)
    z["bar"] = [int(r.min()), int(r.max()), int(cl.min()), int(cl.max())] if len(r) else None
    return z


def _set(z, rr, cc, val):
    for i in range(rr[0], rr[1]):
        for j in range(cc[0], cc[1]):
            if 0 <= i < 64 and 0 <= j < 64:
                z["grid"][i][j] = val


def _rect(z, rr, cc):
    return [z["grid"][i][j] for i in range(rr[0], rr[1]) for j in range(cc[0], cc[1])]


def predict(z, action):
    z = {"grid": [list(r) for r in z["grid"]], "bar": z["bar"]}
    agent = _agent_block(z)
    a, c = agent
    if action == "UP":
        na, nc = a-2, c
    elif action == "DOWN":
        na, nc = a+2, c
    elif action == "LEFT":
        na, nc = a, c-5
    elif action == "RIGHT":
        na, nc = a, c+5
    else:
        return z
    # trail vacate
    old_trail = (slice(a-3, a-1), slice(c, c+5))
    for i in range(a-3, a-1):
        for j in range(c, c+5):
            if z["grid"][i][j] == C_TRAIL:
                z["grid"][i][j] = C_BG
    # agent move
    _set(z, (slice(a, a+3), slice(c, c+5)), C_BG)
    _set(z, (slice(na, na+3), slice(nc, nc+5)), C_AGENT)
    # new trail
    if na >= 2:
        _set(z, (slice(na-3, na-1), slice(nc, nc+5)), C_TRAIL)
    # bar shrink from left
    if z["bar"] is not None:
        r0, r1, c0, c1 = z["bar"]
        for j in range(c0, c0+2):
            for i in range(r0, r1+1):
                z["grid"][i][j] = C_BG
        z["bar"] = [r0, r1, c0+2, c1] if c0+2 <= c1 else None
    return z
