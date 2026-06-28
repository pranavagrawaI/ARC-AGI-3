"""Lesson 0004 companion: give your agent eyes — perception from frame diffs.

Run from the repo root:  uv run python teach/lessons/0004-perception.py

Your dataset is YOUR OWN winning playthrough of ls20 (406 steps, recorded).
Implement the three TODO functions. The check harness gives instant
feedback against ground-truth values verified from the recording.
No environment needed — perception is pure numpy over recorded frames.
"""

import glob
import json
import os

import numpy as np

MOVE_ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")


def load_triples(path=None):
    """Load (prev_frame, action, next_frame, same_level) tuples. GIVEN."""
    if path is None:  # default: the biggest recording you have
        path = max(glob.glob("recordings/*/*.jsonl"), key=os.path.getsize)
    records = [json.loads(line)["data"] for line in open(path)]
    frames = [np.array(r["frame"][-1], dtype=np.int8) for r in records]
    triples = []
    for i in range(1, len(records)):
        triples.append(
            (
                frames[i - 1],
                records[i]["action_input"]["id"],
                frames[i],
                records[i]["levels_completed"] == records[i - 1]["levels_completed"],
            )
        )
    return triples


# ----------------------------------------------------------------------------
# Exercise 1 — change detection.
# Return a boolean (64, 64) mask: True where the pixel differs between frames.
# ----------------------------------------------------------------------------
def change_mask(prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    return np.not_equal(prev, nxt)  # TODO (it's a one-liner)


# ----------------------------------------------------------------------------
# Exercise 2 — find the player.
# In ls20 the player carries a unique orange marker (color 12).
# Return its centroid as (row, col) floats.
# General principle (later lessons): the controllable object is the one whose
# motion CORRELATES with your actions — color is just ls20's cheap shortcut.


# NOTE: This is insufficent for ls20, the player is a square box with a blue bottom and a orange top. the same frame color appears in various places throughout the game. To reliably find the player, you need to find the orange pixels that are directly above blue pixels. This is a common technique in computer vision called "color segmentation" combined with "spatial relationships".
# ----------------------------------------------------------------------------
def find_player(frame: np.ndarray) -> tuple[float, float]:
    orange_mask = frame == 12
    blue_mask = frame == 9
    player_mask = np.logical_and(orange_mask, np.roll(blue_mask, shift=-1, axis=0))
    if not np.any(player_mask):
        raise ValueError("Player not found in the frame")
    rows, cols = np.where(player_mask)

    # find centroid of the player mask
    y = rows.mean()
    x = cols.mean()

    return y, x  # TODO (use orange+blue masks to find player)


# ----------------------------------------------------------------------------
# Exercise 3 — learn action semantics from data.
# For each movement action, compute the player's median (d_row, d_col) across
# all within-level steps. Your agent thereby DISCOVERS what each action does —
# the thing you discovered by hand in lesson 0001.
# Return e.g. {"ACTION1": (-4.0, 0.0), ...}
# ----------------------------------------------------------------------------
def action_semantics(triples) -> dict:
    # TODO (use find_player on prev and next)
    semantics = {}
    for action in MOVE_ACTIONS:
        deltas = []
        for prev, a, nxt, same_level in triples:
            if a == action and same_level:
                y_prev, x_prev = find_player(prev)
                y_nxt, x_nxt = find_player(nxt)
                deltas.append((y_nxt - y_prev, x_nxt - x_prev))
        if deltas:
            semantics[action] = np.median(deltas, axis=0)
    return semantics


# ----------------------------------------------------------------------------
# Check harness — instant feedback. Do not edit below this line.
# ----------------------------------------------------------------------------
def _check(name, fn, expect_fn):
    try:
        result = fn()
    except NotImplementedError:
        print(f"  [ ] {name}: not implemented yet")
        return
    except Exception as e:
        print(f"  [✗] {name}: raised {type(e).__name__}: {e}")
        return
    ok, detail = expect_fn(result)
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")


if __name__ == "__main__":
    triples = load_triples()
    print(f"Loaded {len(triples)} steps from your winning recording.\n")

    def exp_mask(med):
        return med == 88, f"median changed pixels within-level = {med} (expected 88)"

    def exp_player(pos):
        y, x = pos
        ok = abs(y - 45.5) < 1.5 and abs(x - 36.0) < 1.5
        return (
            ok,
            f"player at ({y:.1f}, {x:.1f}) in step-1 frame (expected ≈ (45.5, 36.0))",
        )

    def exp_semantics(sem):
        signs = {a: (np.sign(d[0]), np.sign(d[1])) for a, d in sem.items()}
        expected = {
            "ACTION1": (-1, 0),
            "ACTION2": (1, 0),
            "ACTION3": (0, -1),
            "ACTION4": (0, 1),
        }
        ok = all(signs.get(a) == s for a, s in expected.items())
        names = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}
        learned = {a: names.get(s, "??") for a, s in signs.items()}
        return ok, f"learned {learned}" + (
            "" if ok else " (expected up/down/left/right)"
        )

    _check(
        "Exercise 1: change_mask",
        lambda: int(
            np.median([change_mask(p, n).sum() for p, a, n, same in triples if same])
        ),
        exp_mask,
    )
    _check("Exercise 2: find_player", lambda: find_player(triples[0][2]), exp_player)
    _check(
        "Exercise 3: action_semantics", lambda: action_semantics(triples), exp_semantics
    )

    print(
        "\nWhen all three pass, your agent has eyes. Bring the printed"
        "\nsemantics table to your teacher and we wire perception into"
        "\nan agent that never walks into a wall twice."
    )
