
"""World model for the grid game: player sprite on floor color 3."""
import numpy as np
from collections import deque

Z_SCHEMA = {
    "grid": "list[64 x list[64 int]] — full settled grid snapshot (latent; lossless)",
    "player": "list[[row,col] x5] — sprite cells",
}
FLOOR = 3
VOID = 4

_TMPL = [[31, 21], [32, 20], [32, 21], [32, 22], [33, 21]]

def _anchor(player):
    return min(player, key=lambda p: (p[0], p[1]))

def _drift(player, dr, dc):
    ar, ac = _anchor(player)
    return [[r + dr, c + dc] for r, c in player if not (r == ar and c == ac)]

def _find_sprite(g):
    m = (g == 0) | (g == 1)
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    seen = set()
    best = []
    for y0, x0 in zip(ys, xs):
        y0, x0 = int(y0), int(x0)
        if (y0, x0) in seen:
            continue
        q = deque([(y0, x0)])
        seen.add((y0, x0))
        comp = []
        while q:
            r, c = q.popleft()
            comp.append([r, c])
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < 64 and 0 <= c2 < 64 and (r2, c2) not in seen and m[r2, c2]:
                    seen.add((r2, c2))
                    q.append((r2, c2))
        if len(comp) > len(best):
            best = comp
    if not best:
        return None
    if len(best) != len(_TMPL):
        return sorted(best)
    tr, tc = _anchor(best)
    rel = sorted((r - tr, c - tc) for r, c in _TMPL)
    got = sorted((r - tr, c - tc) for r, c in best)
    if rel != got:
        return sorted(best)
    return sorted([[tr + dr, tc + dc] for dr, dc in rel])

def encode(grid):
    """grid: 64x64 numpy array (or list) of ints -> latent dict."""
    g = np.asarray(grid)
    found = _find_sprite(g)
    return {"grid": g.tolist(),
            "player": found if found else [[r, c] for r, c in _TMPL]}

_D = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

def predict(z, action):
    g = [row[:] for row in z["grid"]]
    A = np.array(g)
    if action == "RESET" or action not in _D:
        return dict(z)
    dr, dc = _D[action]
    player = [list(p) for p in z["player"]]
    ar, ac = _anchor(player)
    nrow, ncol = ar + dr, ac + dc
    if not (0 <= nrow < 64 and 0 <= ncol < 64):
        return dict(z)
    ok = A[nrow, ncol] == FLOOR
    if ok:
        for r, c in _drift(player, dr, dc):
            if not (0 <= r < 64 and 0 <= c < 64) or A[r, c] != FLOOR:
                ok = False
                break
    if not ok:
        return dict(z)
    for r, c in player:
        g[r][c] = FLOOR
    new_player = [[r + dr, c + dc] for r, c in player]
    return {"grid": g, "player": sorted(new_player)}
