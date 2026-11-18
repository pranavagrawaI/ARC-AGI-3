"""World model: 64x64 sliding grid game.

Palette (from level 0):
  4,5,6,7  void/free space (6=inside the bottom strip)
  3        maze wall (TBC: passable or not)
  8,11,12,0,1  solid blocks (TBC)
  9        player/entity
"""
import numpy as np

WALL = 3
SOLID = (8, 11, 12, 0, 1)
VOID = (4, 5, 6, 7)
PLAYER = 9
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}


def parse_grid(grid_rows):
    arr = np.zeros((64, 64), dtype=np.int8)
    for r, row in enumerate(grid_rows):
        vals = [int(ch, 16) for ch in row]
        arr[r, :] = vals
    return arr


def encode(grid_rows):
    g = parse_grid(grid_rows)
    ys, xs = np.where(g == PLAYER)
    return {"player": (int(ys[0]), int(xs[0])), "n_players": int(len(ys))}


def walk(g, r, c, dr, dc, block=(3, 8, 11, 12, 0, 1)):
    """Slide through void cells; stop at anything else."""
    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < 64 and 0 <= nc < 64):
            break
        if g[nr, nc] not in block:
            r, c = nr, nc
            continue
        break
    return r, c


def predict(grid_rows, actions, block=(3, 8, 11, 12, 0, 1)):
    g = parse_grid(grid_rows)
    for a in actions:
        if a not in MOVES:
            break
        dr, dc = MOVES[a]
        pr, pc = np.argwhere(g == PLAYER)[0]
        g[pr, pc] = 4
        g[walk(g, pr, pc, dr, dc, block)] = PLAYER
    return g
