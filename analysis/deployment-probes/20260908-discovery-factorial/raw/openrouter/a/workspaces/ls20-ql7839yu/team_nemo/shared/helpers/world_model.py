
import numpy as np

Z_SCHEMA = {
    "grid": "list[64] of str, each 64 hex chars - full 64x64 grid",
    "player": "list of [row, col, color(int)] for the 5 plus-cells",
    "center": "[row, col] center of the plus",
}

DIRS = {'UP': (-1,0), 'DOWN': (1,0), 'LEFT': (0,-1), 'RIGHT': (0,1)}
WALKABLE = {'3','4'}  # provisional: 3 floor, 4 outside-floor

def _to_rows(g):
    return [''.join(format(int(v),'x') for v in row) for row in g]

def _find_player(g):
    mask = (g==0)|(g==1)
    ys, xs = np.where(mask)
    cells = {(int(r),int(c)): int(g[r,c]) for r,c in zip(ys,xs)}
    center = None
    for (r,c) in cells:
        if (r-1,c) in cells and (r+1,c) in cells and (r,c-1) in cells and (r,c+1) in cells:
            center = (r,c); break
    return center, cells

def encode(grid):
    g = np.asarray(grid)
    center, cells = _find_player(g)
    player = [[r,c,col] for (r,c),col in sorted(cells.items())]
    return {"grid": _to_rows(g), "player": player,
            "center": [int(center[0]), int(center[1])] if center else None}

def predict(z, action):
    if action not in DIRS:
        return z  # RESET/unknown: provisional no-change
    dr, dc = DIRS[action]
    center = tuple(z['center'])
    offcol = {}
    for (r,c,col) in z['player']:
        offcol[(r-center[0], c-center[1])] = col
    nc = (center[0]+dr, center[1]+dc)
    cur = {(center[0]+dr_, center[1]+dc_) for (dr_,dc_) in offcol}
    tgt = {(nc[0]+dr_, nc[1]+dc_) for (dr_,dc_) in offcol}
    g = [list(row) for row in z['grid']]
    for (r,c) in tgt:
        if not (0<=r<64 and 0<=c<64):
            return z
        if g[r][c] not in WALKABLE and (r,c) not in cur:
            return z  # blocked by wall/item
    for (r,c) in cur:
        if (r,c) not in tgt:
            g[r][c] = '3'  # provisional: vacated cell reverts to floor 3
    for (dr_,dc_),col in offcol.items():
        g[nc[0]+dr_][nc[1]+dc_] = format(col,'x')
    return {"grid": [''.join(row) for row in g],
            "player": [[nc[0]+dr_, nc[1]+dc_, col] for (dr_,dc_),col in offcol.items()],
            "center": [int(nc[0]), int(nc[1])] }
