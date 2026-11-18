
"""Slot-machine world model.

Layout (64x64, hex colors 0-f):
- bg = 4
- Frame: rows 24..35, cols 16..46; top/bottom bars color 8 (rows 25, 34), corners color 0.
- 3 levers: 2x2 blocks of color 2 at cols (28,29), (34,35), (40,41).
  Initially at rows 29-30. Static under SPACEBAR.
- Top palette panel rows 1-6: 4 squares 9/e/b/f at cols 24,32,40,48 (with 5 border).
- Bottom targets rows 57-60: 4 squares e/f/9/b at cols 28,36,44,52.
- Status band row 53: mostly color 2; last cell (col 63) is a COUNTER.
  Initially 2. After 1x SPACEBAR it becomes 3. So it increments: counter = 2 + presses?

Hypothesis: SPACEBAR presses a button that increments a counter (col 63 row 53).
  Levers are static. The goal is NOT about lever alignment.
  Maybe SPACEBAR cycles a value and the goal is to match something.
"""
import numpy as np

BG = 4
LEVER = 0x2
LEVER_COLS = [28, 34, 40]
ARM_TOP, ARM_BOT = 26, 32

Z_SCHEMA = {
    "levers": "dict: col -> top row of its 2x2 lever block (static under SPACEBAR)",
    "counter": "int: value at grid[53, 63] — SPACEBAR counter",
    "status": "color of status band row 53 (excluding col 63)",
}

def parse(rows):
    return np.array([[int(ch, 16) for ch in r] for r in rows], dtype=np.int64)

def encode(grid):
    g = grid if isinstance(grid, np.ndarray) else parse(grid)
    z = {}
    z["levers"] = {}
    for c in LEVER_COLS:
        colpair = g[:, c:c+2]
        mask = (colpair == LEVER).all(axis=1)
        rows = np.where(mask)[0]
        rows = rows[(rows >= ARM_TOP) & (rows <= ARM_BOT)]
        z["levers"][c] = int(rows[0]) if len(rows) else None
    z["counter"] = int(g[53, 63])
    z["status"] = int(g[53, 0])  # dominant color of status band
    return z

def predict(z, action):
    nz = dict(z)
    if action == "SPACEBAR":
        nz["counter"] = z["counter"] + 1
    # levers static under SPACEBAR, UNDO, RESET (for now)
    return nz

def aligned(z):
    rows = [r for r in z["levers"].values() if r is not None]
    return len(rows) == 3 and len(set(rows)) == 1
