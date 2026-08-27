"""world_model.py — editable model of this game as a deterministic Moore machine.

L0 mechanics (from observed frames 0->5):
  * Board 64x64. Colors: 9=bg, 10=orange central wall (cols 30-32, full height),
    11=red right wall/floor + a static red block bottom-right, 5=green block,
    4=gray block, 0=holes inside the green block.
  * The green block (t0 bbox rows15-23, cols18-26) and the gray block (t0 bbox
    rows15-23, cols36-44) share a row offset; each block also has its own column
    offset.  Actions:
      RIGHT : green col_off += 3, gray col_off -= 3  (converge toward central axis)
      LEFT  : green col_off -= 3, gray col_off += 3  (diverge)
      DOWN  : row_off += 3
      UP    : row_off -= 3
  * Every turn-taking action (LEFT/RIGHT/UP/DOWN/SPACEBAR) also fills the next top-most
    cell of the right wall column (col 63) green top-down — a move/step counter. SPACEBAR
    advances the counter without moving the blocks.
"""
from dataclasses import dataclass, field
import numpy as np
import wmlib

UNKNOWN = wmlib.UNKNOWN
TRANSITION_RETURNS_NEW_STATE = True


@dataclass
class State:
    base: np.ndarray            # static board (blocks/holes cleared, no counter fill)
    green: np.ndarray           # 9x9 green block shape (9 = empty)
    gray: np.ndarray            # 9x9 gray block shape (9 = empty)
    row_off: int = 0
    gc_off: int = 0
    cc_off: int = 0
    steps: int = 0


def init_state(grid0: np.ndarray, level: int) -> State:
    base = grid0.copy()
    green = grid0[15:24, 18:27].copy()
    gray = grid0[15:24, 36:45].copy()
    # clear both block footprints (including 0-holes) from the base grid
    base[15:24, 18:27] = 9
    base[15:24, 36:45] = 9
    # strip the step counter fill that may already exist in grid0 (cols63 top) -> base keeps it
    return State(base=base, green=green, gray=gray, row_off=0, gc_off=0, cc_off=0, steps=0)


def transition(state: State, action: dict) -> State:
    a = action["action"]
    if a == "RESET":
        # RESET rewinds to the initial state (blocks home, step counter cleared).
        return wmlib.clone(state.__class__(
            base=state.base.copy(),
            green=state.green.copy(),
            gray=state.gray.copy(),
            row_off=0, gc_off=0, cc_off=0, steps=0))
    s = wmlib.clone(state)
    if a in ("LEFT", "RIGHT", "UP", "DOWN", "SPACEBAR"):
        # Any of these front a turn and advance the move/step counter (col63 fills top-down).
        s.steps += 1
        if a == "LEFT":
            s.gc_off -= 3; s.cc_off += 3
        elif a == "RIGHT":
            s.gc_off += 3; s.cc_off -= 3
        elif a == "DOWN":
            s.row_off += 3
        elif a == "UP":
            s.row_off -= 3
        # SPACEBAR increments the counter only; no block movement (observed turn 14).
    return s


def render(state: State) -> np.ndarray:
    g = state.base.copy()
    # paint green block (only cells != 9)
    ry, cx = 15 + state.row_off, 18 + state.gc_off
    sh = state.green
    mask = sh != 9
    g[ry:ry + sh.shape[0], cx:cx + sh.shape[1]][mask] = sh[mask]
    # paint gray block
    ry, cx = 15 + state.row_off, 36 + state.cc_off
    sh = state.gray
    mask = sh != 9
    g[ry:ry + sh.shape[0], cx:cx + sh.shape[1]][mask] = sh[mask]
    # step counter fill top-down on right wall col 63
    for i in range(min(state.steps, 63)):
        g[i, 63] = 5
    return g


def observation_variants(state: State) -> list:
    return []


def outcome(state: State) -> str:
    return "ongoing"


def actions(state: State) -> list:
    return [{"action": "UP", "row": None, "col": None},
            {"action": "DOWN", "row": None, "col": None},
            {"action": "LEFT", "row": None, "col": None},
            {"action": "RIGHT", "row": None, "col": None},
            {"action": "SPACEBAR", "row": None, "col": None},
            {"action": "CLICK", "row": None, "col": None},
            {"action": "UNDO", "row": None, "col": None}]


def subgoals(state: State) -> list:
    return [lambda s: outcome(s) == "level_complete"]


def heuristic(state: State) -> float:
    return 0