import numpy as np
def encode(grid):
    return {"x": int(np.where(grid == 12)[1].min()), "right": int(np.where(grid == 11)[1].max())}
def predict(z, action):
    return {"x": z["x"] + 1, "right": z["right"] - 2}
