
import numpy as np

Z_SCHEMA = {
  "player": "list of 5 [row,col] of the dark blob (3x '0' + 2x '1'), moves with input",
  "patches": "dict key (color) -> list of [row,col] for the two shifting patches 'c' and '9'; shift with input too (same dx,dy as player?)",
  "step": "int",
}

def encode(grid):
    g = np.array(grid, dtype=int)
    z = {}
    z["player"] = np.argwhere((g == 0) | (g == 1)).tolist()
    z["patches"] = {str(c): np.argwhere(g == c).tolist() for c in (12, 9)}
    return z

def predict(z, action):
    import copy
    z2 = copy.deepcopy(z)
    dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}.get(action, (0, 0))
    if action in ("UP", "DOWN", "LEFT", "RIGHT"):
        for k in list(z2["patches"].keys()):
            z2["patches"][k] = [[r + dr, c + dc] for r, c in z2["patches"][k]]
        z2["player"] = [[r + dr, c + dc] for r, c in z2["player"]]
    return z2
