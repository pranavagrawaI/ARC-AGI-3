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
    move_count: int = 0


def init_state(grid0: np.ndarray, level: int) -> State:
    """s0 — the state at the level's first frame. Read what you can from grid0; set fields not visible
    in one frame to their start value and update them in transition() as you learn them."""
    # infer move counter from column 63: count leading 5s from top
    col = grid0[:, 63]
    move_count = 0
    for v in col:
        if v == 5:
            move_count += 1
        else:
            break
    return State(grid=grid0.copy(), move_count=move_count)


def transition(state: State, action: dict) -> State:
    """The next state after `action`, a dict {"action","row","col"} (row/col = the ACTION6 click cell,
    else None) — read action["action"]. Copy, mutate, return."""
    s = wmlib.clone(state)
    act = action["action"]
    # work on a copy of grid before moves
    grid_before = s.grid.copy()
    new_grid = grid_before.copy()

    # move movable pieces (color 4 and 5 components, excluding bottom row and column 63)
    if act in ("UP", "DOWN", "LEFT", "RIGHT"):
        dr = 0
        base_dc = 0
        if act == "UP":
            dr = -3
        elif act == "DOWN":
            dr = 3
        elif act == "LEFT":
            base_dc = -3
        elif act == "RIGHT":
            base_dc = 3

        segs = wmlib.segment(grid_before)
        for seg in segs:
            col = seg['color']
            r0, c0, r1, c1 = seg['bbox']
            # movable pieces are color 4 or 5, not bottom row, not touching column 63
            if col in (4, 5) and r1 != 63 and c1 < 63:
                h = r1 - r0 + 1
                w = c1 - c0 + 1
                # ignore tiny single-cell artifacts (e.g., move counter markers)
                if h < 2 or w < 2:
                    continue
                sub = grid_before[r0:r1+1, c0:c1+1]
                # clear old area to background (9)
                new_grid[r0:r1+1, c0:c1+1] = 9
                if act in ("UP", "DOWN"):
                    nr0 = r0 + dr
                    nc0 = c0
                else:  # LEFT/RIGHT with mirrored direction per side
                    centroid_col = (c0 + c1) / 2.0
                    # left side of barrier (~col < 32) moves with action direction, right side moves opposite
                    if centroid_col < 32:
                        dc = base_dc
                    else:
                        dc = -base_dc
                    nr0 = r0
                    nc0 = c0 + dc
                nr1 = nr0 + h - 1
                nc1 = nc0 + w - 1
                if 0 <= nr0 and 0 <= nc0 and nr1 < 64 and nc1 < 64:
                    new_grid[nr0:nr1+1, nc0:nc1+1] = sub
        s.grid = new_grid

    # move counter painted on column 63 (monotonic bar)
    s.move_count += 1
    row = s.move_count - 1
    if 0 <= row < 64:
        s.grid[row, 63] = 5

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
