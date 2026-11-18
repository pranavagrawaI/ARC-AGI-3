
"""World model for 'the game' — 64x64 grid, level 0.

Palette: 0=black 2=green 3=olive 4=navy 5=purple 15=white.

Verified level-0 map (turn 0) — everything else bg=5(purple):
  top-left r0-17 c0-17: A=black 10x5 r3-7 c3-12; B=white 10x5 r8-12 c3-12;
      navy wall c16-17 r0-15; navy base r16-17 c0-17
  top-right r0-8 c18-63: solid olive(3); C=navy frame w/ black 3x3 r2-6 c35-39;
      D=navy frame w/ white 3x3 r2-6 c41-45; E=black strip 5x1 r7 c41-45
  middle: F=green frame box r24-32 c25-38, white fill r25-31 c26-37, OPEN bottom
      (r32 has only corners c25,c38 = green)
  bottom: G=black 10x10 r34-43 c27-36
  bottom row 63 = navy(4) across c0-62; cell (63,63) is SPECIAL:
      4 at step0, 5 after 1 action (turn 1). -> step-counter OR invisible-player marker.
  No visible player sprite at step 0.
"""
import numpy as np

ELEMENTS = {
    "A":  dict(r0=3,  r1=7,  c0=3,  c1=12),
    "B":  dict(r0=8,  r1=12, c0=3,  c1=12),
    "C":  dict(r0=2,  r1=6, c0=35, c1=39),
    "D":  dict(r0=2,  r1=6, c0=41, c1=45),
    "E":  dict(r0=7,  r1=7,  c0=41, c1=45),
    "F":  dict(r0=24, r1=32, c0=25, c1=38),
    "G":  dict(r0=34, r1=43, c0=27, c1=36),
}

def hud_counter(grid):
    """Bottom-right corner cell (63,63) value. 4@step0, 5@step1."""
    return int(grid[63, 63])

def encode(grid):
    z = {"el": {}}
    for name, d in ELEMENTS.items():
        z["el"][name] = grid[d["r0"]:d["r1"]+1, d["c0"]:d["c1"]+1].tobytes()
    z["row63_other"] = grid[63, :63].tobytes()
    z["counter"] = hud_counter(grid)
    return z

def playfield(grid):
    """Grid with the special (63,63) cell normalized to border 4."""
    g = grid.copy()
    g[63, 63] = 4
    return g

def wm_changed(g1, g2):
    return np.argwhere(g1 != g2).tolist()

def diff_playfield(g1, g2):
    """Changed pixels ignoring the special corner cell."""
    return wm_changed(playfield(g1), playfield(g2))

def diff_regions(g1, g2):
    z1, z2 = encode(g1), encode(g2)
    changed = [k for k in z1["el"] if z1["el"][k] != z2["el"][k]]
    if z1["row63_other"] != z2["row63_other"]:
        changed.append("row63_other")
    if z1["counter"] != z2["counter"]:
        changed.append("counter")
    return changed

def predict(z, action):
    """Dynamics placeholder. Each action so far only changes the counter +1;
    no playfield object moves. Refine as probes land."""
    z2 = dict(z); z2["el"] = dict(z["el"])
    z2["counter"] = z["counter"] + 1
    return z2

def expected_level0():
    g = np.full((64, 64), 5, dtype=int)
    g[0:9, 18:64] = 3
    g[0:16, 16:18] = 4
    g[16:18, 0:18] = 4
    g[3:8, 3:13] = 0
    g[8:13, 3:13] = 15
    g[2:7, 35:40] = 4; g[3:6, 36:39] = 0
    g[2:7, 41:46] = 4; g[3:6, 42:45] = 15
    g[7, 41:46] = 0
    g[24:33, 25:39] = 5
    g[24, 25:39] = 2; g[25:32, 25] = 2; g[25:32, 38] = 2
    g[32, 25] = 2; g[32, 38] = 2
    g[25:32, 26:38] = 15
    g[34:44, 27:37] = 0
    g[63, :] = 4
    return g
