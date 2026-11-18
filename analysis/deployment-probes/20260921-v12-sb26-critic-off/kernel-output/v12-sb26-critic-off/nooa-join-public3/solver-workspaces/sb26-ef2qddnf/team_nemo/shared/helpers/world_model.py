
import json
import numpy as np

def encode(grid):
    return {"grid": grid.tolist()}

def predict(z, action):
    # No dynamic model yet — identity. But host compares full latent; so we need
    # to keep predict(z) == encode(grid) so that "unchanged" states predict correctly.
    # Return z as-is (identity).
    return dict(z)
