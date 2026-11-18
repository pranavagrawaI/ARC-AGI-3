from dataclasses import dataclass
import numpy as np
@dataclass
class State:
    grid: np.ndarray

def init_state(grid0, level): return State(grid0.copy())
def transition(state, action):
    value=int(state.grid[0,0]) + 2
    return State(np.full_like(state.grid, 14 if value >= 3 else value))
def render(state): return state.grid
def outcome(state): return 'level_complete' if int(state.grid[0,0]) == 14 else 'ongoing'
def actions(state): return [{'action':'UP','row':None,'col':None}]
def heuristic(state): return 0
