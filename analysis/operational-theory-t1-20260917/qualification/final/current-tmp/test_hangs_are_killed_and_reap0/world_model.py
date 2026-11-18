while True: pass
def encode(grid): return {"x": int(grid[0,0])}
def predict(z, action):
    z["x"] += 1 if action == "RIGHT" else 10
    return z
