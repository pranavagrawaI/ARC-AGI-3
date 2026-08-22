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
        # C5 does not move in RIGHT
    elif act == "LEFT":
        s.x4 += 3
        # C5 does not move in LEFT
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
    # Using a larger area to ensure we clear everything that might have been there
    # but let's be more precise.
    grid[15:24, 18:27] = 9
    grid[15:24, 36:45] = 9
    # Clear budget col
    grid[0:64, 63] = 5
    
    # 2. Draw moving objects
    # C4 Shape: 3x9 top, 6x3 bottom
    y4, x4 = state.y4, state.x4
    grid[y4:y4+3, x4:x4+9] = 4
    grid[y4+3:y4+9, x4:x4+3] = 4

    # C5 Shape: 
    # Row 0: cols 0-8 (relative to x5)
    # Row 1: cols 0, 2, 3, 4, 5, 6, 8
    # Row 2: cols 0-8
    # Row 3-8: cols 6-8 (relative to x5)
    # Wait, looking at the relative coordinates from run_python:
    # Shape 5 relative: {(0, 0): 5, (0, 1): 5, (0, 2): 5, (0, 3): 5, (0, 4): 5, (0, 5): 5, (0, 6): 5, (0, 7): 5, (0, 8): 5, 
    # (1, 0): 5, (1, 2): 5, (1, 3): 5, (1, 4): 5, (1, 5): 5, (1, 6): 5, (1, 8): 5, (C1,1 and C1,7 are 9)
    # (2, 0): 5, (2, 1): 5, (2, 2): 5, (2, 3): 5, (2, 4): 5, (2, 5): 5, (2, 6): 5, (2, 7): 5, (2, 8): 5,
    # (3, 6): 5, (3, 7): 5, (3, 8): 5, ... (up to 8, 8)
    # Wait, the run_python output for Shape 5 was relative to min_y5=15, min_x5=18.
    # Let's just use the set of relative coordinates.
    
    y5, x5 = state.y5, state.x5
    # Redraw based on the actual relative coords found
    rel_coords5 = {
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
        (1, 0), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 8),
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
        (3, 6), (3, 7), (3, 8),
        (4, 6), (4, 8),
        (5, 6), (5, 7), (5, 8),
        (6, 6), (6, 7), (6, 8),
        (7, 6), (7, 8),
        (8, 6), (8, 7), (8, 8)
    }
    # Wait, the relative x for Shape 5 was based on min_x5=18.
    # The cells were (0, 18), (0, 19)... (0, 26). 
    # So relative x is 0, 1, 2, 3, 4, 5, 6, 7, 8.
    # But the relative x coordinates I just listed are 0-8.
    # Let's use the exact relative offsets from the run_python output.
    # The relative coordinates in run_python:
    # (0, 0): 5, ..., (0, 8): 5  --- Wait, (0, 18) - 18 = 0.
    # (1, 0): 5, (1, 2): 5, ... (1, 8): 5
    # (2, 0): 5, ... (2, 8): 5
    # (3, 6): 5, (3, 7): 5, (3, 8): 5
    # (4, 6): 5, (4, 8): 5
    # (5, 6): 5, (5, 7): 5, (5, 8): 5
    # (6, 6): 5, (6, 7): 5, (6, 8): 5
    # (7, 6): 5, (7, 8): 5
    # (8, 6): 5, (8, 7): 5, (8, 8): 5
    
    # Let's reconstruct this carefully.
    for dy, dx in rel_coords5:
        grid[y5 + dy, x5 + dx] = 5

    # 3. Budget (col 63)
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
