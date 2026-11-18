
def encode(grid):
    """Encode floor run state. grid: 64x64 int array."""
    row = [int(x) for x in grid[63]]
    run_start = 63
    for i in range(63, -1, -1):
        if row[i] == 4:
            run_start = i
        else:
            break
    run_len = 63 - run_start + 1 if row[run_start] == 4 else 0
    # count c13 pixels
    c13 = sum(1 for r in range(64) for c in range(64) if grid[r,c] == 13)
    return {"floor_run_start": int(run_start), "floor_run_len": int(run_len), "c13_count": int(c13)}

def predict(z, action):
    """Predict next state. Floor grows leftward by 1 or 2 cells (irregular)."""
    import copy
    z2 = dict(z)
    # We can't predict exact growth amount, so mark as uncertain
    # The run grows by at least 1
    z2["floor_run_len"] = z["floor_run_len"] + 1  # conservative
    z2["floor_run_start"] = z["floor_run_start"] - 1
    return z2
