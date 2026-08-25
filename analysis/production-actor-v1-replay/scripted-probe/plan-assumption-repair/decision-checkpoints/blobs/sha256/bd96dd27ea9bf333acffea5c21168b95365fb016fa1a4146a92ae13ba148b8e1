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
    """Add the fields this game needs (e.g. avatar_xy, steps_left, palette, parts). `grid` is a default
    for games that mostly move/recolor cells; for a richer model keep a compact state here and build
    the grid in render(). A move/life counter rendered to pixels is real state, so hold its value in a
    field and paint it in render(). Use UNKNOWN only for genuinely unobserved/uninferable cells, such
    as unseen viewport terrain; keep coverage high and do not use UNKNOWN for ordinary mistakes."""
    grid: np.ndarray


def init_state(grid0: np.ndarray, level: int) -> State:
    """s0 — the state at the level's first frame. Read what you can from grid0; set fields not visible
    in one frame to their start value and update them in transition() as you learn them."""
    return State(grid=grid0.copy())


def transition(state: State, action: dict) -> State:
    """The next state after `action`, a dict {"action","row","col"} (row/col = the ACTION6 click cell,
    else None) — read action["action"]. Copy, mutate, return."""
    s = wmlib.clone(state)
    grid = s.grid
    act = action.get("action")
    if act in ("UP", "DOWN", "LEFT", "RIGHT"):
        # Find movable block component consisting of colors 9 (blue) and 12 (orange)
        # Start from orange cells (12) and flood fill through 9/12 to isolate the block
        H, W = grid.shape
        visited = np.zeros((H, W), dtype=bool)
        # find all orange cells
        orange_coords = np.argwhere(grid == 12)
        component_mask = np.zeros((H, W), dtype=bool)
        from collections import deque
        for r0, c0 in orange_coords:
            if visited[r0, c0]:
                continue
            # BFS
            q = deque()
            q.append((r0, c0))
            visited[r0, c0] = True
            cells = []
            while q:
                r, c = q.popleft()
                if grid[r, c] not in (9, 12):
                    continue
                component_mask[r, c] = True
                cells.append((r, c))
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and grid[nr, nc] in (9,12):
                        visited[nr, nc] = True
                        q.append((nr, nc))
        if np.any(component_mask):
            rs, cs = np.where(component_mask)
            r_min, r_max = rs.min(), rs.max()
            c_min, c_max = cs.min(), cs.max()
            h = r_max - r_min + 1
            w = c_max - c_min + 1
            # extract block pattern
            block = grid[r_min:r_max+1, c_min:c_max+1].copy()
            new_r_min = r_min
            new_c_min = c_min
            if act == "UP":
                new_r_min = r_min - h
            elif act == "DOWN":
                new_r_min = r_min + h
            elif act == "LEFT":
                new_c_min = c_min - w
            elif act == "RIGHT":
                new_c_min = c_min + w
            if 0 <= new_r_min and new_r_min + h <= H and 0 <= new_c_min and new_c_min + w <= W:
                # clear old location to background 3
                grid[r_min:r_max+1, c_min:c_max+1] = 3
                # place block at new location
                grid[new_r_min:new_r_min+h, new_c_min:new_c_min+w] = block
        # Step counter / yellow bar decrement
        # Yellow bar is color 11 on rows 61-62, shrinking from left each action
        for r in (61, 62):
            cols = np.where(grid[r] == 11)[0]
            if cols.size > 0:
                leftmost = int(cols.min())
                grid[r, leftmost] = 3
    return s


def render(state: State) -> np.ndarray:
    """The grid `state` shows — verify scores render(simulated state) against the observed frame.
    Return state.grid if you keep the board directly; otherwise build the HxW int16 grid from your
    fields (e.g. wmlib.composite(state.parts, H, W), or paint an avatar/counter). Return colors 0-15
    for cells you claim. Return UNKNOWN (-1) only for cells not determined by the modeled state or
    observations yet; those cells are ignored for accepted accuracy but reduce prediction_coverage."""
    return state.grid


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
    return [{"action": "UP", "row": None, "col": None}, {"action": "DOWN", "row": None, "col": None}, {"action": "LEFT", "row": None, "col": None}, {"action": "RIGHT", "row": None, "col": None}]


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
