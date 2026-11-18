
import numpy as np

# World model, v2 (measured against ALL 7 frames of the step 0-6 history).
#
# CONFIRMED (every transition 0->6):
#   * Agent = 5x5 solid block of color 12.
#   * HUD band: 2 rows (61,62) of color 11; LEFT edge creeps +1 col per
#     executed step (13,14,15,16,17,18,19); RIGHT edge fixed at 54.
#   * 8-widget rows 61-62 cols 56-63: fixed decoration.
#
# PROVISIONAL (unresolved; do NOT trust for batching):
#   * Recorded motion vs executed action: step 1 UP -> agent went RIGHT;
#     steps 4/5 UP -> no move; step 6 LEFT -> no move. The recorded grid
#     motion may be a separate (possibly animated) actor, or input may be
#     lagged/absorbed. Until step 7 disambiguates, predict() is a pure
#     identity on the agent fields (only the band is predicted to move).

Z_SCHEMA = {
    "agent_top_row": "int, top row of the 5x5 agent (12 block)",
    "agent_col": "int, left col of the agent",
    "band_left": "int, leftmost col of the 11-band (= 13 + steps)",
    "band_right": "int, rightmost col of the 11-band (fixed 54 so far)",
    "steps": "int, steps executed this level",
}

_A, _B = 12, 11

def encode(grid):
    g = grid
    m = (g == _A)
    if m.sum() == 0:
        return None
    rs, cs = np.where(m)
    top, col = int(rs.min()), int(cs.min())
    m11 = (g == _B)
    bl = int(np.where(m11)[1].min()) if m11.sum() > 0 else 13
    br = int(np.where(m11)[1].max()) if m11.sum() > 0 else 54
    return {"agent_top_row": top, "agent_col": col,
            "band_left": bl, "band_right": br,
            "steps": max(0, bl - 13),
            "agent_w": int(cs.max()-cs.min()+1), "agent_h": int(rs.max()-rs.min()+1)}

def predict(z, action):
    """Provisional dynamics: agent stays put (motion unexplained), band creeps."""
    z2 = dict(z)
    # agent fields unchanged - step 6 LEFT showed zero agent motion
    z2["band_left"] = z["band_left"] + 1
    z2["band_right"] = z["band_right"]
    z2["steps"] = z.get("steps", 0) + 1
    return z2
