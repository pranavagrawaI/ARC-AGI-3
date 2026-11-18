
import numpy as np

def encode(grid):
    g = np.asarray(grid, dtype=int)
    assert g.shape == (64, 64)
    row = g[63, :]
    yel = np.where(row == 4)[0]
    non3 = np.where(row != 3)[0]
    return {"counter_cells": [int(c) for c in yel],
            "row63_non3": [int(c) for c in non3]}

def predict(z, action):
    z = dict(z)
    if action.startswith("CLICK"):
        non3 = sorted(set(z["row63_non3"]))
        yel = sorted(set(z["counter_cells"]))
        if non3:
            left = min(non3)
            if left > 0:
                non3.append(left - 1)
                yel.append(left - 1)
        else:
            yel.append(63); non3.append(63)
        z["row63_non3"] = sorted(non3)
        z["counter_cells"] = sorted(yel)
        return z
    if action == "RESET":
        z["counter_cells"] = []
        z["row63_non3"] = []
        return z
    return z
