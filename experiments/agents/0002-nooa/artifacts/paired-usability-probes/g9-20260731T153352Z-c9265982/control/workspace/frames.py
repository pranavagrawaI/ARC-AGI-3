"""Load and compare public observations. Provided by the harness; edit freely.

    import sys; sys.path.insert(0, "."); import frames

    frames.steps()          -> [0, 1, 2, ...] every step on disk
    frames.load(3)          -> the whole observation dict for step 3
                               (every function also accepts a file path)
    frames.grid(3)          -> the current layer as a 2-D numpy array
    frames.diff(3, 4)       -> what changed between two steps
    frames.where(3, value)  -> [(y, x), ...] cells holding a value

Coordinates are (y, x) = (row, column) throughout, matching the x and y an action
takes. grid() returns pixels[-1]: the layer count varies between steps and the
last layer is the current state.
"""

import json
import glob

import numpy as np

OBSERVATIONS = "observations"
MAX_LISTED_CELLS = 200


def steps():
    """Every step number with an observation on disk, ascending."""

    found = []
    for path in glob.glob(OBSERVATIONS + "/step-*.json"):
        try:
            found.append(int(path.split("step-")[1].split(".")[0]))
        except (IndexError, ValueError):
            continue
    return sorted(found)


def path(step):
    return "%s/step-%05d.json" % (OBSERVATIONS, step)


def resolve(step_or_path):
    """Accept a step number or a path to an observation file.

    The first live use of this helper called diff() with two file paths and failed;
    the model then retried with step numbers. Taking either costs one line and
    removes the failure, which is worth more than being strict about it.
    """

    if isinstance(step_or_path, str):
        return step_or_path
    return path(step_or_path)


def load(step_or_path):
    """The whole observation dict for a step number or a path."""

    with open(resolve(step_or_path)) as handle:
        return json.load(handle)


def grid(step_or_observation):
    """The current layer as a 2-D array.

    The number of layers changes between steps and the current state is always the
    last one, so this returns pixels[-1]. Indexing pixels[0] is the single most
    common way recorded runs read a stale grid without any error being raised.
    """

    observation = (
        step_or_observation
        if isinstance(step_or_observation, dict)
        else load(step_or_observation)
    )
    return np.array(observation["pixels"][-1])


def where(step, value):
    """Every (y, x) holding a value."""

    return [tuple(int(n) for n in cell) for cell in np.argwhere(grid(step) == value)]


def __getattr__(name):
    """Name what exists when something else is asked for.

    A live model guessed frames.load_observation() and got Python's bare
    AttributeError, which says what is missing but not what is available. Listing
    the API in the error costs nothing and turns a dead end into a correction.
    """

    raise AttributeError(
        "frames has no %r. Available: steps(), load(step_or_path), "
        "grid(step_or_path), diff(before, after), where(step, value)." % (name,)
    )


def diff(before, after, max_cells=MAX_LISTED_CELLS):
    """What changed between two steps.

    Returns counts, the bounding box of the change, how many cells made each
    value transition, and the changed cells themselves. The cell list is capped
    and says so rather than truncating silently.
    """

    a, b = grid(before), grid(after)
    if a.shape != b.shape:
        return {
            "comparable": False,
            "reason": "grids have different shapes",
            "before_shape": list(a.shape),
            "after_shape": list(b.shape),
        }
    mask = a != b
    ys, xs = np.nonzero(mask)
    transitions = {}
    for y, x in zip(ys, xs):
        key = "%d->%d" % (a[y, x], b[y, x])
        transitions[key] = transitions.get(key, 0) + 1
    cells = [
        {"y": int(y), "x": int(x), "from": int(a[y, x]), "to": int(b[y, x])}
        for y, x in list(zip(ys, xs))[:max_cells]
    ]
    return {
        "comparable": True,
        "changed": int(mask.sum()),
        "listed": len(cells),
        "truncated": int(mask.sum()) > len(cells),
        "bbox": None
        if not len(ys)
        else {
            "y0": int(ys.min()),
            "x0": int(xs.min()),
            "y1": int(ys.max()),
            "x1": int(xs.max()),
        },
        "transitions": transitions,
        "cells": cells,
    }
