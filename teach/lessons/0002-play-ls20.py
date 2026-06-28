"""Lesson 0002 companion: play ls20 with live scoring and recording.

Run from the repo root:  uv run python teach/lessons/0002-play-ls20.py

Differences from the lesson-0001 harness:
- Every completed level immediately prints YOUR score vs the human baseline.
- The full session is recorded to recordings/<scorecard_id>/*.jsonl —
  your own play becomes data an agent could learn from later.
"""

import sys

import numpy as np
import arc_agi
from arc_agi.base import OperationMode
from arc_agi.rendering import COLOR_MAP, hex_to_rgb
from arcengine import GameAction

BASELINES = [22, 123, 73, 84, 96, 192, 186]  # upper-median human, per level

RGB = {v: hex_to_rgb(c) for v, c in COLOR_MAP.items()}


def render_compact(steps, frame_data):
    """Draw the 64x64 grid in 64x32 terminal cells.

    Each character shows two vertically stacked pixels: '▀' colored with
    the top pixel as foreground and the bottom pixel as background.
    """
    grid = frame_data.frame[-1]
    height, width = grid.shape
    lines = []
    for y in range(0, height, 2):
        parts = []
        for x in range(width):
            tr, tg, tb = RGB.get(int(grid[y, x]), (0, 0, 0))
            br, bg, bb = RGB.get(int(grid[y + 1, x]), (0, 0, 0))
            parts.append(f"\033[38;2;{tr};{tg};{tb}m\033[48;2;{br};{bg};{bb}m▀")
        lines.append("".join(parts) + "\033[0m")
    sys.stdout.write("\033[H\033[2J" + "\n".join(lines) + "\n")
    sys.stdout.flush()


arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
env = arc.make("ls20", renderer=render_compact, save_recording=True)

frame = env.reset()
prev_grid = frame.frame[0].copy()
completed = frame.levels_completed
level_actions = 0
total_actions = 0
level_scores = []

print(f"\nGame: {frame.game_id}  |  levels to win: {frame.win_levels}")
print("Commands: 1 2 3 4 = actions, r = reset, q = quit")
print(f"Level 1 human baseline: {BASELINES[0]} actions — beat it!\n")

while True:
    cmd = input("action> ").strip().lower()
    if cmd == "q":
        break
    elif cmd == "r":
        frame = env.reset()
        level_actions += 1  # RESET is an action too (id 0) — it isn't free
        total_actions += 1
        print("-- reset (counted as an action) --")
    elif cmd in ("1", "2", "3", "4"):
        frame = env.step(GameAction.from_id(int(cmd)))
        level_actions += 1
        total_actions += 1
    else:
        print("?? type 1-4, r, or q")
        continue

    grid = frame.frame[0]
    changed = int(np.sum(grid != prev_grid))
    prev_grid = grid.copy()

    if frame.levels_completed > completed:
        baseline = BASELINES[completed]
        score = min((baseline / level_actions) ** 2, 1.15)
        level_scores.append(score)
        print(
            f"\n*** LEVEL {completed + 1} COMPLETE: {level_actions} actions"
            f" vs human {baseline}  →  level score {score:.3f} ***\n"
        )
        completed = frame.levels_completed
        level_actions = 0
        if completed < frame.win_levels:
            print(f"Level {completed + 1} baseline: {BASELINES[completed]} actions\n")

    print(
        f"state={frame.state.value}  level={min(frame.levels_completed + 1, frame.win_levels)}"
        f"/{frame.win_levels}  level_actions={level_actions}  pixels_changed={changed}"
    )
    if frame.state.value == "WIN":
        print("\n*** YOU WON THE GAME ***")
        break
    if frame.state.value == "GAME_OVER":
        print("\n*** GAME OVER — type r to reset (it costs an action) ***")

# Game score: weighted average of level scores, weight = 1-indexed level number.
# Unreached levels score 0 and still carry their (large) weights.
weights = list(range(1, frame.win_levels + 1))
padded = level_scores + [0.0] * (frame.win_levels - len(level_scores))
game_score = sum(w * s for w, s in zip(weights, padded)) / sum(weights)

print(f"\nLevels completed: {completed}/{frame.win_levels}  |  total actions: {total_actions}")
print(f"Per-level scores: {[round(s, 3) for s in padded]}")
print(f"GAME SCORE (level-weighted): {game_score:.1%}")
print("\nYour play was recorded as data — find it with:  ls recordings/*/")
print("(one JSON line per step: action, state, and the full 64x64 frame)")
