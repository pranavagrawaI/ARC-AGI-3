# world_model.py — v1
# Confirmed: 5x5 block (top 2 rows color c/12, bottom 3 rows color 9) slides exactly 5 cells
# (one 5-tile) in the action direction on the '3' arena, leaving '3' behind.
# Every action also erodes one column (rows 61-62) from the LEFT of the b(11) strip; erased -> '3'.
Z_SCHEMA = {
    "rows": "full 64x64 grid hex strings (block drawn in place)",
    "br": "block top-left row (int)",
    "bc": "block top-left col (int)",
    "L":  "b-strip left-edge col (starts 13, +1 per action)",
}
DIRS = {"UP": (-5, 0), "DOWN": (5, 0), "LEFT": (0, -5), "RIGHT": (0, 5)}
B_ROWS = (61, 62)
B_RIGHT = 54
ARENA = "3"
def _hex(c):
    return "c" if c == 12 else "9"
def _block_cells(br, bc):
    return [(br+dr, bc+dc) for dr in range(5) for dc in range(5)]
def encode(grid):
    rows = ["".join(f"{int(v):x}" for v in row) for row in grid.tolist()]
    g = [list(r) for r in rows]
    br, bc = 45, 34
    for r in range(0, 60):
        for c in range(0, 60):
            if g[r][c] == "c":
                ok = True
                for dr in range(5):
                    for dc in range(5):
                        if g[r+dr][c+dc] != ("c" if dr < 2 else "9"):
                            ok = False; break
                    if not ok: break
                if ok:
                    br, bc = r, c
    L = 64
    for c in range(64):
        if g[61][c] == "b":
            L = c; break
    return {"rows": rows, "br": int(br), "bc": int(bc), "L": int(L)}
def predict(z, action):
    g = [list(r) for r in z["rows"]]
    br, bc = z["br"], z["bc"]
    L = z["L"]
    if action in DIRS:
        dr, dc = DIRS[action]
        nbr, nbc = br+dr, bc+dc
        ok = True
        for (rr, cc) in _block_cells(nbr, nbc):
            if not (0 <= rr < 64 and 0 <= cc < 64) or g[rr][cc] != ARENA:
                ok = False; break
        if ok:
            for (rr, cc) in _block_cells(br, bc):
                g[rr][cc] = ARENA
            for drr in range(5):
                for dcc in range(5):
                    g[nbr+drr][nbc+dcc] = "c" if drr < 2 else "9"
            br, bc = nbr, nbc
    L2 = min(L+1, B_RIGHT+1)
    for rr in B_ROWS:
        for cc in range(L, L2):
            if 0 <= cc < 64 and g[rr][cc] == "b":
                g[rr][cc] = ARENA
    return {"rows": ["".join(r) for r in g], "br": int(br), "bc": int(bc), "L": int(L2)}
