
"""World model for the CLICK grid game (64x64, bg=5, HUD row 63).

Colors:
  5 bg, 3 walls, 2 box border, 4 box interior,
  b(11) solid/ring item A, e(14) ring item B, d(13) marker dots

MECHANIC (confirmed turns 2-3):
  - CLICK r c (row first, col second).
  - Click on a ring center: ring dissolves (e-ring -> bg, b-ring -> solid b block),
    HUD cell (63, 63-k) 3 -> 4 for the next k. A d-marker appears at the ring's
    left-center cell (for e-rings; b-rings become solid, no marker).
  - Goal: fill HUD to 64 fours (then what? level up).
  - Walls (3) block clicks? (untested)
  - Boxes (2/4): containers; rings live inside them.

Z_SCHEMA:
  level: int
  cells: dict color -> sorted [(r,c)]
  player: None (no avatar found; the d at (9,52) cluster is decorative? or a ring?)
  hud: tuple 16 ints (per-color counts row 63)
  hud_fill: int — number of 4s in HUD
  rings: list of dict(color, r0, r1, c0, c1, center_clicked: bool, in_box: int)
  blocks: list of dict(color, r0,r1,c0,c1) — solid blocks
"""
import numpy as np
from collections import deque

BG = 5
HUD_ROW = 63
WALL = 3
BORDER = 2
INTERIOR = 4

def _components(mask):
    comps = []
    lab = -np.ones(mask.shape, int)
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c] and lab[r, c] == -1:
                q = deque([(r, c)]); lab[r, c] = 1; pts = []
                while q:
                    y, x = q.popleft(); pts.append((int(y), int(x)))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny,nx] and lab[ny,nx] == -1:
                            lab[ny,nx] = 1; q.append((ny,nx))
                comps.append(pts)
    return comps

def encode(g):
    g = np.asarray(g)
    hud = [0]*16
    for v in g[HUD_ROW].tolist():
        hud[int(v)] += 1
    field = g.copy(); field[HUD_ROW] = BG
    cells = {}
    for v in np.unique(field).tolist():
        if v == BG: continue
        ys, xs = np.nonzero(field == v)
        cells[int(v)] = sorted(zip(ys.tolist(), xs.tolist()))
    # identify rings (3x3 with center != ring color)
    rings = []
    blocks = []
    for col in (11, 14):
        mask = field == col
        for comp in _components(mask):
            rs = [p[0] for p in comp]; cs = [p[1] for p in comp]
            r0,r1,c0,c1 = min(rs),max(rs),min(cs),max(cs)
            h, w = r1-r0+1, c1-c0+1
            if h==3 and w==3 and len(comp)==8:
                rings.append(dict(color=int(col), r0=r0,r1=r1,c0=c0,c1=c1,
                                  center=(r0+1,c0+1), center_val=int(field[r0+1,c0+1])))
            elif h==3 and w==3 and len(comp)==9:
                blocks.append(dict(color=int(col), r0=r0,r1=r1,c0=c0,c1=c1))
            else:
                blocks.append(dict(color=int(col), r0=r0,r1=r1,c0=c0,c1=c1, n=len(comp)))
    boxes = []
    for comp in _components(field == BORDER):
        rs = [p[0] for p in comp]; cs = [p[1] for p in comp]
        boxes.append(dict(r0=min(rs),r1=max(rs),c0=min(cs),c1=max(cs)))
    rings.sort(key=lambda d:(d["r0"],d["c0"]))
    boxes.sort(key=lambda d:(d["r0"],d["c0"]))
    return dict(level=0, cells=cells, hud=tuple(hud), hud_fill=hud[4],
                rings=rings, blocks=blocks, boxes=boxes,
                walls=cells.get(3, []))

def predict(z, action):
    """action = ('CLICK', r, c). Returns z'.

    Rule: if (r,c) is a ring center -> dissolve ring (e->bg+marker, b->solid),
    HUD +1. Else no_impact (wall/interior/empty).
    """
    z2 = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in z.items()}
    if action and action[0] == "CLICK":
        _, r, c = action
        for ring in z2["rings"]:
            if ring["center"] == (r, c):
                z2["rings"].remove(ring)
                if ring["color"] == 14:  # e-ring: dissolve, marker at (r, c0)
                    z2["hud_fill"] += 1
                    z2["hud"] = tuple(hud_upd(z["hud"]))
                    z2["cells"] = dict(z["cells"])
                    c13 = list(z["cells"].get(13, []))
                    c13.append((r, ring["c0"]))
                    c13.sort()
                    z2["cells"][13] = c13
                elif ring["color"] == 11:  # b-ring: solidify
                    z2["hud_fill"] += 1
                    z2["hud"] = tuple(hud_upd(z["hud"]))
                    z2["blocks"].append(dict(color=11, r0=ring["r0"],r1=ring["r1"],c0=ring["c0"],c1=ring["c1"]))
                break
    return z2

def hud_upd(hud):
    hud = list(hud)
    if hud[3] > 0:
        hud[3] -= 1; hud[4] += 1
    return tuple(hud)

def grid_from_z(z):
    g = np.full((64,64), BG, int)
    for v, pts in z["cells"].items():
        for (r,c) in pts: g[r,c] = v
    for v in range(16):
        g[HUD_ROW, :z["hud"][v]] = v
    return g
