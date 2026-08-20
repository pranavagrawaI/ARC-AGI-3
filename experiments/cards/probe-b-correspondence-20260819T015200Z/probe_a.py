"""Probe A: was the agent's dynamics model solve-ready, with the goal as the only gap?

Nothing in world_model.py is modified. The goal is supplied externally through
plan_bfs's `target` parameter, which is exactly the seam the agent left empty by
leaving outcome() returning "ongoing".

Part 1 replays the hypothesised solution through the agent's own transition().
Part 2 asks whether BFS would have found it, given only a goal predicate.
"""

import re
import sys
import time

import numpy as np

import wmlib
import world_model as M

HEX = {"0": 0, "4": 4, "5": 5, "9": 9, "a": 10, "b": 11}


def load_grid(path):
    rows = []
    for line in open(path):
        m = re.match(r"y(\d\d): (.*)", line)
        if m:
            rows.append([HEX[c] for c in m.group(2).replace("|", "").split()])
    return np.array(rows, dtype=np.int16)


grid0 = load_grid("level_0/turn_000.txt")
start = M.init_state(grid0, 0)

# The target region, read off the opening frame: 11-valued cells excluding the
# col-63 move counter.
tr, tc = np.where(grid0 == 11)
keep = tc != 63
TARGET_CELLS = (tr[keep], tc[keep])
print(f"target cells: {len(TARGET_CELLS[0])}")


def b_covers_target(state):
    """The goal the agent never wrote: Shape B fully covers the target region."""
    g = M.render(state)
    return bool(np.all(g[TARGET_CELLS] == 4))


print(f"goal already satisfied at start? {b_covers_target(start)}")
print(f"agent's own outcome(start): {M.outcome(start)!r}")

# ---------------------------------------------------------------- part 1
# Replay the hypothesised solution through the agent's unmodified transition().
plan = ["DOWN"] * 10 + ["LEFT"] * 5
s = start
for i, a in enumerate(plan, 1):
    s = M.transition(s, a)
print(f"\npart 1: replay {len(plan)} presses (10 DOWN, 5 LEFT) through agent's transition()")
print(f"  goal reached: {b_covers_target(s)}")
print(f"  move_count after: {s.move_count}")

g = M.render(s)
vals, counts = np.unique(g[TARGET_CELLS], return_counts=True)
print(f"  values on target cells: {dict(zip(vals.tolist(), counts.tolist()))}")

# What the agent actually pursued: drive Shape A onto the target instead.
sA = start
for a in ["DOWN"] * 10 + ["RIGHT"] * 11:
    sA = M.transition(sA, a)
gA = M.render(sA)
valsA, countsA = np.unique(gA[TARGET_CELLS], return_counts=True)
print(f"\n  counterfactual — the agent's actual approach (10 DOWN, 11 RIGHT on Shape A):")
print(f"    values on target cells: {dict(zip(valsA.tolist(), countsA.tolist()))}")
print(f"    goal reached: {b_covers_target(sA)}")

# ---------------------------------------------------------------- part 2
# Would BFS have found it, given only the goal predicate?
print("\npart 2: plan_bfs against the agent's own model, goal supplied externally")
for max_nodes in (20000, 200000):
    t0 = time.time()
    diag = wmlib.plan_bfs_with_diagnostics(
        M, start, target=b_covers_target, max_nodes=max_nodes
    )
    dt = time.time() - t0
    found = diag["status"] == "found"
    print(
        f"  max_nodes={max_nodes:>7}: status={diag['status']:<22} "
        f"expanded={diag['expanded_nodes']:<7} seen={diag['seen_states']:<7} "
        f"plan_len={len(diag['plan']) if found else None} {dt:.1f}s"
    )
    if diag["errors"]:
        print(f"    errors: {diag['errors'][:2]}")
    if found:
        print(f"    plan: {[a['action'] if isinstance(a, dict) else a for a in diag['plan']]}")
        break
