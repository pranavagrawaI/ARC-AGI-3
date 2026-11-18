
import numpy as np

# Playfield is rows 0..51 (64 cols). HUD rows 52..63.
PF_ROWS = (0, 52)
HUD_ROWS = (52, 64)

def parse(g):
    """g: 64x64 int array. -> dict with playfield facts + hud."""
    pf = g[:52]
    out = {"pf_rows": pf.tolist()}
    ys, xs = np.where(pf == 9)
    out["entity9"] = (int(ys.mean()), int(xs.mean())) if len(ys) else None
    ys, xs = np.where(pf == 12)
    out["goal12"] = (int(ys.mean()), int(xs.mean())) if len(ys) else None
    # goal color = color of 5x5 block under/overlapped by 12; here 1
    ys, xs = np.where(pf == 1)
    out["goal1"] = (int(ys.mean()), int(xs.mean())) if len(ys) else None
    # player 0 marker
    ys, xs = np.where(pf == 0)
    out["player0"] = (int(ys.mean()), int(xs.mean())) if len(ys) else None
    # 1-color patch
    out["hud"] = g[52:64].tolist()
    return out

def encode(g):
    return parse(g)

def diff(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return int((a != b).sum())
