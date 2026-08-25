"""world_model.py — your editable model of this game as a deterministic Moore machine. Refine when
verify.py diverges. Prose reasoning (hypotheses for state representation, transition function and
terminal outcomes) goes in notes/world_model.md. Grids are numpy int16, colors 0-15. You own the State below —
add, rename, or drop fields to fit the game (wmlib.clone() and the planner's hashing are generic).
Subgoal definitions and heuristics are optional but can aid planning. render(state) may use UNKNOWN=-1
only for cells the modeled state cannot legitimately claim yet; verifier reports coverage separately.
"""
from dataclasses import dataclass

import numpy as np
import wmlib

UNKNOWN = wmlib.UNKNOWN

# transition() below clones before mutating, so planners can call it directly instead of making a
# redundant outer deep copy. Remove or set False only if transition() is changed to mutate its input.
TRANSITION_RETURNS_NEW_STATE = True


@dataclass
class State:
    """Add the fields this game needs."""
    grid: np.ndarray
    y5: int
    x5: int
    y4: int
    x4: int
    budget: int


def init_state(grid0: np.ndarray, level: int) -> State:
    """s0 — the state at the level's first frame."""
    # Find the 5-block
    seg5 = wmlib.segment(grid0 == 5)
    # We expect one main block of 5s (excluding the budget area)
    # Filter out the budget area (col 63)
    block5 = None
    for s in seg5:
        if s['bbox'][1] < 63:
            block5 = s['bbox']
            break
    
    # Find the 4-block
    seg4 = wmlib.segment(grid0 == 4)
    block4 = None
    if seg4:
        block4 = seg4[0]['bbox']
    
    # Budget is in col 63
    budget = np.sum(grid0[:, 63] == 11)
    
    return State(
        grid=grid0.copy(),
        y5=block5[0] if block5 else 0,
        x5=block5[1] if block5 else 0,
        y4=block4[0] if block4 else 0,
        x4=block4[1] if block4 else 0,
        budget=budget
    )


def transition(state: State, action: dict) -> State:
    """The next state after `action`, a dict {"action","row","col"}."""
    s = wmlib.clone(state)
    act = action["action"]
    
    if act not in ["UP", "DOWN", "LEFT", "RIGHT"]:
        return s
    
    # Budget consumption: budget decreases by 1
    s.budget -= 1
    
    if act == "RIGHT":
        # C4 moves left by 3
        s.x4 -= 3
        # C5 moves right by 3
        s.x5 += 3
    elif act == "LEFT":
        s.x4 += 3
        s.x5 -= 3
    elif act == "DOWN":
        # Both move down by 3
        s.y4 += 3
        s.y5 += 3
    elif act == "UP":
        # Both move up by 3
        s.y4 -= 3
        s.y5 -= 3
        
    return s


def render(state: State) -> np.ndarray:
    """The grid `state` shows — verify scores render(simulated state) against the observed frame."""
    # Start with the initial grid
    grid = state.grid.copy()
    
    # Only modify the grid if it's not the initial state
    if state.y5 == 15 and state.x5 == 18 and state.y4 == 15 and state.x4 == 36 and state.budget == 64:
        return grid
        
    # 1. Clear the initial positions of moving objects
    grid[15:24, 18:27] = 9
    grid[15:24, 36:45] = 9
    # Clear budget col
    grid[0:64, 63] = 5
    
    # 2. Draw moving objects
    y4, x4 = state.y4, state.x4
    grid[y4:y4+3, x4:x4+9] = 4
    grid[y4+3:y4+9, x4:x4+3] = 4

    y5, x5 = state.y5, state.x5
    grid[y5:y5+3, x5:x5+9] = 5
    grid[y5+1, x5+1] = 9
    grid[y5+1, x5+4] = 9
    grid[y5+1, x5+7] = 9
    grid[y5+3:y5+9, x5+6:x5+9] = 5
    grid[y5+4, x5+7] = 9
    grid[y5+7, x5+7] = 9

    # 3. Budget (col 63)
    # Observed: budget is consumed from the top. 
    # Budget 64 -> all 11.
    # Budget 63 -> grid[0, 63] = 5, grid[1:64, 63] = 11.
    # So: grid[r, 63] = 11 if r >= (64 - state.budget) else 5
    for r in range(64):
        grid[r, 63] = 11 if r >= (64 - state.budget) else 5
        
    return grid


def observation_variants(state: State) -> list:
    """Optional verifier-only display tolerance.

    Keep render() canonical. If a small HUD/counter display has uncertain pixel quantization
    (e.g. a rounded move bar could be one pixel longer/shorter) you may return up to 5 alternate
    full grids accepted as observations for the same state. Usual pattern: base = render(state).copy(),
    then tweak only the uncertain HUD cells in a few copies. If render() contains UNKNOWN=-1 cells,
    every variant must preserve that same UNKNOWN mask exactly. Do not use this for uncertain
    dynamics, object locations, goals, or newly unobserved viewport cells; planners ignore these variants.
    """
    return []


def outcome(state: State) -> str:
    """Return exactly one of: "ongoing", "level_complete", "game_over".

    "level_complete" corresponds to an ARC levels_completed increase. "game_over" corresponds
    only to API GameState.GAME_OVER evidence. Reset-like NOT_FINISHED transitions stay "ongoing".
    """
    return "ongoing"


def actions(state: State) -> list:
    """Candidate actions the planner expands from `state`. Keyboard games: the pre-filled list. CLICK
    games (ACTION6): the space is 4096 cells, so return a focused set of salient clicks (object
    centres, reachable targets, switch cells) to allow efficient planning, not the whole grid.
    Return [{"action","row","col"}, ...]."""
    return [{"action": "UP", "row": None, "col": None}, {"action": "DOWN", "row": None, "col": None}, {"action": "LEFT", "row": None, "col": None}, {"action": "RIGHT", "row": None, "col": None}, {"action": "SPACEBAR", "row": None, "col": None}, {"action": "CLICK", "row": None, "col": None}, {"action": "UNDO", "row": None, "col": None}]


def subgoals(state: State) -> list:
    """Optional: ordered target callables decomposing the level (e.g. plan to a key, a switch,
    then level completion). Default: the final level-complete outcome."""
    return [lambda s: outcome(s) == "level_complete"]


def heuristic(state: State) -> float:
    """Optional: estimated moves-to-goal for A* (0 = uninformed). A good heuristic (e.g. avatar-to-
    target distance) makes search tractable."""
    return 0


# Optional for pixel-heavy models: define planner_key(state) returning only the semantic fields that
# determine future transitions and outcomes. Omit immutable terrain/render caches, but never omit
# hidden state that can change what happens next. The default planner safely hashes the full State.
