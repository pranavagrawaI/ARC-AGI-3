"""World model: 64x64 grid of 32x32 2x2 tiles.

Colors (hex digit):
  0 = black      (dark / target)
  2 = green      (light / fillable)
  4 = blue
  5 = white      (wall / empty board)
  8 = gray
  9 = red
  12 (0xc) = orange (status bar)

Grid layout: 32 rows x 32 cols of 2x2 tiles.
  Tile row 31 / grid row 63: status bar (color 12 on top half).
  Board rows 0..30, board cols 0..30.
  Tiles at board positions (0,0),(0,1),(1,0),(1,1): always 5 (empty) — corner gaps.

Hypothesized mechanic:
  Clicking a 4-connected run of same-color tiles (excluding 5 and status bar)
  changes them to the "opposite" color, where 0<->2 and 8<->9 are swap pairs.
  Goal: make all target colors disappear (all tiles become 5? or some count = 0).
"""
import numpy as np

Z_SCHEMA = {
    "board": "32x32x32 uint4 array — tile colors (board rows/cols 0..30 plus corner gaps)",
    "status": "integer count shown in the status bar (unknown unit; see notes)",
    "level": "int — current level",
}

SWAP = {0: 2, 2: 0, 8: 9, 9: 8, 4: 4, 5: 5}

def tile_grid(arr):
    """arr: 64x64 int array -> 32x32 tile array (each tile 2x2 uniform, take top-left)."""
    return arr[0::2, 0::2]

def encode(grid_rows):
    """grid_rows (list of 64 hex strings) -> z dict."""
    arr = _parse(grid_rows)
    return {"board": tile_grid(arr).copy(), "rows": grid_rows}

def _parse(grid_rows):
    a = np.empty((64, 64), dtype=np.uint8)
    for r, line in enumerate(grid_rows):
        for c, ch in enumerate(line):
            a[r, c] = int(ch, 16)
    return a

def flood(board, r, c, color):
    """4-connected component of `color` containing (r,c) in the 32x32 board."""
    if not (0 <= r < 32 and 0 <= c < 32):
        return set()
    if board[r, c] != color:
        return set()
    seen = set()
    stack = [(r, c)]
    while stack:
        rr, cc = stack.pop()
        if (rr, cc) in seen:
            continue
        seen.add((rr, cc))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < 32 and 0 <= nc < 32 and (nr, nc) not in seen and board[nr, nc] == color:
                stack.append((nr, nc))
    return seen

def runs(board):
    """All 4-connected same-color runs (color != 5)."""
    board = np.asarray(board)
    seen = np.zeros(board.shape, bool)
    out = []
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            col = board[r, c]
            if col == 5 or seen[r, c]:
                continue
            comp = flood(board, r, c, int(col))
            for rr, cc in comp:
                seen[rr, cc] = True
            out.append((int(col), comp))
    return out

def predict(z, action):
    """z (from encode) + 'CLICK x y' (grid coords, col x row y) -> z' (or None if no change).

    Hypothesis: 4-connected flood fill of the clicked tile's color flips
    0<->2 and 8<->9; if the run is 4, nothing happens; 5 is a wall.
    """
    board = np.array(z["board"])
    if action in ("", "RESET"):
        return None
    if action.startswith("CLICK"):
        try:
            x, y = (int(v) for v in action.split()[1:3])
        except Exception:
            return None
    else:
        return None
    tr, tc = y // 2, x // 2
    if not (0 <= tr < 32 and 0 <= tc < 32):
        return None
    col = int(board[tr, tc])
    if col == 5:
        return None
    comp = flood(board, tr, tc, col)
    if not comp:
        return None
    new = board.copy()
    if col == 4:
        # hypothesis: 4 runs are inert
        return {"board": new, "changed": []}
    new[comp] = SWAP.get(col, col)
    changed = [(r, c) for r, c in comp]
    return {"board": new, "changed": changed, "comp": comp}
