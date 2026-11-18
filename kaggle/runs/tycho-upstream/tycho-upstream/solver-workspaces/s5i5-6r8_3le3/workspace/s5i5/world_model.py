"""Moore-style model for the unknown click game (level 0).

DISCOVERED MECHANICS (fits all 15 observed clicks across 2 attempts, 16/16):
- 64x64 board, mostly 5 (bg); 7 static objects (no board cell other than
  row 63 has ever changed under any click).
- Row 63 is a 'track' of value 3; a fill of value 4 grows LEFT from col 63.
- Every committed ACTION6 click advances the fill by +1, EXCEPT when the
  PRE-CLICK bar length L satisfies L % 9 in {1, 6} (milestones
  1,6,10,15,19,24,28,33,37,42,46,51,55,60,64), which give +2 instead.
  In click-count form: +2 at clicks #2,6,9,13,16,20,23,27,30,34,37,41,44,
  48,51,55,58,62 (differences 4,3,4,3,... from #6 on).
  Target cell is IRRELEVANT (15 varied clicks, delta depends only on L).
- Goal not yet discovered; no terminal at L=9 or L=12. Top hypothesis:
  fill the bar to full (64) -> reached at click #50.
"""
import numpy as np

from dataclasses import dataclass, field

TRANSITION_RETURNS_NEW_STATE = True

# +2 when (pre-click bar length) % 9 in {1, 6}; else +1.
EXTRA_RESIDUES = (1, 6)
EXTRA_MOD = 9


def bar_increment(L):
    return 2 if L % EXTRA_MOD in EXTRA_RESIDUES else 1


@dataclass
class State:
    level: int = 0
    turn: int = 0
    base: np.ndarray = field(default=None, repr=False)  # board, track row set to 3
    bar_cols: set = field(default_factory=set, repr=False)  # filled cols in row 63
    clicks: int = 0  # number of committed ACTION6 clicks this attempt

    def __post_init__(self):
        if self.base is None:
            self.base = np.zeros((64, 64), dtype=np.int16)


def init_state(grid0, level=0):
    g = np.array(grid0, dtype=np.int16).copy()
    # strip any pre-existing fill so we can rebuild it from click count
    for c in np.argwhere(g[63, :] == 4):
        g[63, int(c)] = 3
    base = g.copy()
    base[63, :] = np.where(base[63, :] == 4, 3, base[63, :])
    return State(level=level, turn=0, base=base, bar_cols=set(), clicks=0)


def _extend_bar(s, n):
    # fill grows from the right edge leftwards over track cells
    for c in range(63, -1, -1):
        if n <= 0:
            break
        if c not in s.bar_cols and s.base[63, c] == 3:
            s.bar_cols.add(c)
            n -= 1


def transition(state, action):
    if TRANSITION_RETURNS_NEW_STATE:
        s = State(
            level=state.level,
            turn=state.turn + 1,
            base=state.base.copy(),
            bar_cols=set(state.bar_cols),
        )
    else:
        s = state
        s.turn += 1
    a = action["action"]
    if a == "ACTION6":
        L = len(s.bar_cols)
        inc = bar_increment(L)  # +2 at pre-click lengths with L % 9 in {1,6}
        _extend_bar(s, inc)
    # RESET / unknown: leave state
    return s


def render(state):
    g = state.base.copy()
    for c in state.bar_cols:
        g[63, c] = 4
    return g


def outcome(state):
    # Goal not yet discovered; no terminal / GAME_OVER observed.
    return "ongoing"


def actions(state):
    # Only ACTION6 matters and target does NOT affect the delta; pick a
    # fixed safe bg cell.
    return [{"action": "ACTION6", "row": 0, "col": 0}]


def subgoals(state):
    return []


def heuristic(state):
    return 64 - len(state.bar_cols)
