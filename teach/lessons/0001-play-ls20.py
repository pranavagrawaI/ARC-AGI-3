"""Lesson 0001 companion: play ls20 by hand and watch the agent loop.

Run from the repo root:  uv run python teach/lessons/0001-play-ls20.py

You type an action, the environment returns a frame. That loop —
observation in, action out — is the entire interface every ARC-AGI-3
agent has. No instructions, no reward signal, just frames.
"""

import numpy as np
import arc_agi
from arc_agi.base import OperationMode
from arcengine import GameAction

# OFFLINE = use only the locally downloaded game in environment_files/.
# No network, no API key needed, and steps run as fast as Python allows.
arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
env = arc.make("ls20", render_mode="terminal-fast")

frame = env.reset()
prev_grid = frame.frame[0].copy()
n_actions = 0

print(f"\nGame: {frame.game_id}  |  levels to win: {frame.win_levels}")
print(f"Available actions: {frame.available_actions}")
print("Commands: 1 2 3 4 = actions, r = reset, q = quit\n")

while True:
    cmd = input("action> ").strip().lower()
    if cmd == "q":
        break
    elif cmd == "r":
        frame = env.reset()
        n_actions = 0
        print("-- reset --")
    elif cmd in ("1", "2", "3", "4"):
        frame = env.step(GameAction.from_id(int(cmd)))
        n_actions += 1
    else:
        print("?? type 1-4, r, or q")
        continue

    grid = frame.frame[0]
    changed = int(np.sum(grid != prev_grid))
    prev_grid = grid.copy()
    print(
        f"state={frame.state.value}  level={frame.levels_completed + 1}/{frame.win_levels}"
        f"  actions={n_actions}  pixels_changed={changed}"
    )
    if frame.state.value == "WIN":
        print("\n*** YOU WON THE GAME ***")
        break
    if frame.state.value == "GAME_OVER":
        print("\n*** GAME OVER — type r to reset ***")

print("\nFinal scorecard:")
print(arc.get_scorecard())
