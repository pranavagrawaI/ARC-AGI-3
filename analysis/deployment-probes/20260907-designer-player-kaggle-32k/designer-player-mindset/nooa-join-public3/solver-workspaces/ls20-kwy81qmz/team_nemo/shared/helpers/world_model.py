import numpy as np

VERSION = "v1-plus-moves-need-4-free"

# Sprite relative offsets (center at +2). Colors: N/E 0, W/S 1, center 0.
_OFFSETS = [(-1, 0, 0), (0, 1, 0), (0, -1, 1), (1, 0, 1), (0, 0, 0)]
_DIR = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
WALL = 3

Z_SCHEMA = {
    "grid": "list[list[int]] 64x64 copy of the full grid (safe cells carried verbatim)",
    "player": "[row, col] sprite center",
    "facing": "UP|DOWN|LEFT|RIGHT (provisional; initial 'RIGHT' unconfirmed)",
}


def encode(grid):
    g = np.asarray(grid, dtype=int)
    mask = (g == 0) | (g == 1)
    rows, cols = np.where(mask)
    cr, cc = int(32), int(21)
    if len(rows):
        # player = cell of color 0 that is NOT a plus arm: prefer the plus center
        # (the color-0 cell whose N,S,E,W includes a 1 pair) - fallback centroid.
        for r in rows:
            for c in cols:
                if g[r, c] == 0 and g[r - 1, c] in (0, 1) and g[r + 1, c] in (0, 1) \
                   and g[r, c - 1] in (0, 1) and g[r, c + 1] in (0, 1):
                    cr, cc = int(r), int(c)
    return {"grid": g.tolist(), "player": [cr, cc], "facing": "RIGHT"}


def _sprite_cells(g, player):
    """List of (r, c) the plus occupies; all must be 0/1 cells or center."""
    cr, cc = player
    return [(cr + dr, cc + dc) for dr, dc, _ in _OFFSETS]


def predict(z, action):
    g = np.array(z["grid"], dtype=int)
    player = list(z["player"])
    facing = z.get("facing", "RIGHT")
    zout = {"grid": g.tolist(), "player": player, "facing": facing}
    if action not in _DIR:
        return zout  # RESET etc.: unknown, carry state
    dr, dc = _DIR[action]
    nplayer = [player[0] + dr, player[1] + dc]
    blocked = False
    for (r, c) in _sprite_cells(g, nplayer):
        if not (0 <= r < 64 and 0 <= c < 64) or g[r, c] == WALL:
            blocked = True
            break
    if not blocked:
        # clear old sprite cells, stamp new one
        for (r, c) in _sprite_cells(g, player):
            g[r, c] = 4
        for dr2, dc2, col in _OFFSETS:
            g[nplayer[0] + dr2, nplayer[1] + dc2] = col
        player = nplayer
        if action in _DIR:
            facing = action
    return {"grid": g.tolist(), "player": player, "facing": facing}
