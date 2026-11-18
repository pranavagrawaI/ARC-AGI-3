
import numpy as np

# World model for the grid game (turn-1 scaffold, dynamics untested)
Z_SCHEMA = {
    "step": ("int", ">=0"),
    "level": ("int", ">=0"),
    "centers": ("list[(int,int)]", "cells of color 3; each has exactly two 5-neighbors"),
    "diamonds": ("list[(int,int)]", "centers of plus-shaped color-13 markers"),
    "fill14": ("bool", "True if 14 is used as machine fill"),
    "fill11": ("bool", "True if 11 is used as machine fill"),
}

def encode(grid):
    g = np.asarray(grid)
    centers = [(int(r), int(c)) for r, c in zip(*np.where(g == 3))]
    d13 = [(r, c) for r, c in zip(*np.where(g == 13))]
    diamonds = []
    for (r, c) in d13:
        if (g[r-1, c] == 13 and g[r+1, c] == 13
                and g[r, c-1] == 13 and g[r, c+1] == 13):
            diamonds.append((int(r), int(c)))
    centers.sort()
    diamonds.sort()
    return {
        "step": 0,
        "level": 0,
        "centers": centers,
        "diamonds": diamonds,
        "fill14": bool((g == 14).any()),
        "fill11": bool((g == 11).any()),
    }

def predict(z, action):
    # Untested. Working hypothesis: only CLICK is meaningful; clicking a
    # diamond (or its arms) advances ONE of the color-3 centers one cell
    # toward a 5-neighbor. Unknown which center, which direction, and what
    # completes the level.
    return dict(z)
