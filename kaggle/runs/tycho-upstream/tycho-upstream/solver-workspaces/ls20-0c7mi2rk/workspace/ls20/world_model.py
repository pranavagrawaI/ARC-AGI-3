"""Moore-style model for the 64x64 grid game (level 0 first).
Player = white(0)+lightgrey(1) glyph. Actions 1-4 = up/down/left/right (docs convention).
Floor/wall assignment refined after first probe (see notes/world_model.md).
"""
import numpy as np
from dataclasses import dataclass, field

TRANSITION_RETURNS_NEW_STATE = True
UNKNOWN = -1


@dataclass
class State:
    grid: np.ndarray = field(default_factory=lambda: np.zeros((64, 64), dtype=np.int16))
    level: int = 0
    turn: int = 0
    player: tuple = (31, 21)  # anchor row,col of white glyph (to be confirmed)
    status: str = "ongoing"


def init_state(grid0, level=0):
    g = np.array(grid0, dtype=np.int16)
    # locate player glyph: white cells
    ys, xs = np.argwhere((g == 0)).T
    anchor = (int(ys.min()), int(xs.min())) if len(ys) else (31, 21)
    return State(grid=g, level=level, turn=0, player=anchor)


def _clone(s):
    return State(grid=s.grid.copy(), level=s.level, turn=s.turn + 1,
                 player=s.player, status=s.status)


def transition(state, action):
    ns = _clone(state)
    a = action["action"]
    dr, dc = {"ACTION1": (-1, 0), "ACTION2": (1, 0),
              "ACTION3": (0, -1), "ACTION4": (0, 1)}.get(a, (0, 0))
    r, c = ns.player
    nr, nc = r + dr, c + dc
    # provisional walkable set: charcoal 4 and dark-grey 3 (TBD after probe)
    if 0 <= nr < 64 and 0 <= nc < 64 and ns.grid[nr, nc] in (3, 4):
        ns.player = (nr, nc)
    return ns


def render(state):
    return state.grid


def outcome(state):
    return state.status


def actions(state):
    return [{"action": a} for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")]


def subgoals(state):
    return []


def heuristic(state):
    return 0
