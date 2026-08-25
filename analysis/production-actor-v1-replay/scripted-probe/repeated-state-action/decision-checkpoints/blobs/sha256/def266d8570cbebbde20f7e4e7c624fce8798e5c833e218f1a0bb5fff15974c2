"""world_model.py — deterministic Moore machine for level 0."""
from dataclasses import dataclass
import numpy as np
import wmlib
from scipy.ndimage import label

UNKNOWN = wmlib.UNKNOWN
TRANSITION_RETURNS_NEW_STATE = True

@dataclass
class State:
    orig_a_r: int
    orig_a_c: int
    orig_b_r: int
    orig_b_c: int
    move_count: int
    shape_a_pat: np.ndarray
    shape_b_pat: np.ndarray
    static_grid: np.ndarray
    init_a_r: int
    init_a_c: int
    init_b_r: int
    init_b_c: int
    history: list

def init_state(grid0: np.ndarray, level: int) -> State:
    grid = grid0.copy()
    # mask UI borders
    mask_a = ((grid == 5) | (grid == 0))
    mask_a[63, :] = False
    mask_a[:, 63] = False
    labeled, n = label(mask_a)
    best_i = 1
    best_sz = 0
    for i in range(1, n + 1):
        sz = np.sum(labeled == i)
        if sz > best_sz:
            best_sz = sz
            best_i = i
    ys, xs = np.where(labeled == best_i)
    r0, r1 = int(ys.min()), int(ys.max())
    c0, c1 = int(xs.min()), int(xs.max())
    h, w = r1 - r0 + 1, c1 - c0 + 1
    pat_a = np.full((h, w), -1, dtype=np.int16)
    pat_a[ys - r0, xs - c0] = grid[ys, xs]

    mask_b = (grid == 4)
    mask_b[63, :] = False
    mask_b[:, 63] = False
    labeled_b, nb = label(mask_b)
    best_i_b = 1
    best_sz_b = 0
    for i in range(1, nb + 1):
        sz = np.sum(labeled_b == i)
        if sz > best_sz_b:
            best_sz_b = sz
            best_i_b = i
    ys_b, xs_b = np.where(labeled_b == best_i_b)
    r0b, r1b = int(ys_b.min()), int(ys_b.max())
    c0b, c1b = int(xs_b.min()), int(xs_b.max())
    hb, wb = r1b - r0b + 1, c1b - c0b + 1
    pat_b = np.full((hb, wb), -1, dtype=np.int16)
    pat_b[ys_b - r0b, xs_b - c0b] = grid[ys_b, xs_b]

    static = grid0.copy()
    static[mask_a] = 9
    static[mask_b] = 9

    return State(
        orig_a_r=r0, orig_a_c=c0, orig_b_r=r0b, orig_b_c=c0b,
        move_count=0,
        shape_a_pat=pat_a, shape_b_pat=pat_b,
        static_grid=static,
        init_a_r=r0, init_a_c=c0, init_b_r=r0b, init_b_c=c0b,
        history=[]
    )

def _get_action(action):
    if isinstance(action, dict):
        return action.get("action")
    return action

def transition(state: State, action) -> State:
    s = wmlib.clone(state)
    act = _get_action(action)
    if act == "RESET":
        s.orig_a_r = s.init_a_r
        s.orig_a_c = s.init_a_c
        s.orig_b_r = s.init_b_r
        s.orig_b_c = s.init_b_c
        s.move_count = 0
        s.history = []
        return s
    if act == "UNDO":
        if s.history:
            prev_a_r, prev_a_c, prev_b_r, prev_b_c = s.history.pop()
            s.orig_a_r = prev_a_r
            s.orig_a_c = prev_a_c
            s.orig_b_r = prev_b_r
            s.orig_b_c = prev_b_c
        # UNDO does not increment move counter
        return s
    dy = 0
    dx = 0
    inc = False
    if act == "UP":
        dy = -3
        inc = True
    elif act == "DOWN":
        dy = 3
        inc = True
    elif act == "LEFT":
        dx = -3
        inc = True
    elif act == "RIGHT":
        dx = 3
        inc = True
    elif act == "SPACEBAR":
        inc = True  # counter increments, no movement
    else:
        # CLICK, etc. no effect
        return s
    if dy != 0 or dx != 0:
        # push current positions onto history before moving
        s.history.append((s.orig_a_r, s.orig_a_c, s.orig_b_r, s.orig_b_c))
        s.orig_a_r += dy
        s.orig_a_c += dx
        s.orig_b_r += dy
        s.orig_b_c -= dx
    if inc:
        s.move_count += 1
    return s

def render(state: State) -> np.ndarray:
    grid = state.static_grid.copy()
    # shape B first (under)
    pat_b = state.shape_b_pat
    r0, c0 = state.orig_b_r, state.orig_b_c
    rows, cols = np.where(pat_b != -1)
    for dr, dc in zip(rows, cols):
        r = r0 + int(dr)
        c = c0 + int(dc)
        if 0 <= r < 64 and 0 <= c < 64:
            if grid[r, c] != 10:  # do not overwrite wall
                grid[r, c] = int(pat_b[dr, dc])
    # shape A on top
    pat_a = state.shape_a_pat
    r0, c0 = state.orig_a_r, state.orig_a_c
    rows, cols = np.where(pat_a != -1)
    for dr, dc in zip(rows, cols):
        r = r0 + int(dr)
        c = c0 + int(dc)
        if 0 <= r < 64 and 0 <= c < 64:
            val = int(pat_a[dr, dc])
            if val == 5:
                grid[r, c] = 5
            elif val == 0:
                # zero cells are holes: paint 0 unless underlying is wall (10) or target (11)
                if grid[r, c] != 10 and grid[r, c] != 11:
                    grid[r, c] = 0
    # move counter on right border
    for i in range(state.move_count):
        if i < 63:
            grid[i, 63] = 5
    return grid

def observation_variants(state: State):
    return []

def outcome(state: State) -> str:
    return "ongoing"

def actions(state: State):
    return [
        {"action": "UP", "row": None, "col": None},
        {"action": "DOWN", "row": None, "col": None},
        {"action": "LEFT", "row": None, "col": None},
        {"action": "RIGHT", "row": None, "col": None},
        {"action": "SPACEBAR", "row": None, "col": None},
        {"action": "CLICK", "row": None, "col": None},
        {"action": "UNDO", "row": None, "col": None},
        {"action": "RESET", "row": None, "col": None},
    ]
