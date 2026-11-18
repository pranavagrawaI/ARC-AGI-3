
BR_MIN, BR_MAX = 40, 45
BC_MIN, BC_MAX = 34, 39
BR_INIT, BC_INIT, BAR_INIT = 45, 34, 13
STEP = 5

def encode(grid):
    a = grid if isinstance(grid, list) else grid.tolist()
    br = None
    for r in range(BR_MIN, BR_MAX + 1):
        if sum(1 for c in range(BC_MIN, BC_MAX + 1) if int(a[r][c]) == 12) >= 5:
            br = r
            break
    bc = None
    for c in range(BC_MIN, BC_MAX + 1):
        if sum(1 for r in range(BR_MIN, BR_MAX + 2) if int(a[r][c]) == 12) >= 2:
            bc = c
            break
    bar = None
    for c in range(64):
        if int(a[61][c]) == 11 and int(a[62][c]) == 11:
            bar = c
            break
    return {"br": br, "bc": bc, "bar": bar}

def predict(z, action):
    a = str(action).upper()
    br, bc, bar = z["br"], z["bc"], z["bar"]
    if br is None: br = BR_INIT
    if bc is None: bc = BC_INIT
    if bar is None: bar = BAR_INIT
    if a == "UP":    br = max(BR_MIN, br - STEP)
    elif a == "DOWN": br = min(BR_MAX, br + STEP)
    elif a == "LEFT": bc = max(BC_MIN, bc - STEP)
    elif a == "RIGHT": bc = min(BC_MAX, bc + STEP)
    elif a == "RESET":
        return {"br": BR_INIT, "bc": BC_INIT, "bar": BAR_INIT}
    else:
        return {"br": br, "bc": bc, "bar": bar}
    bar = bar + 1
    return {"br": br, "bc": bc, "bar": bar}
