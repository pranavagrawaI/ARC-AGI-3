"""Probe A, part 2b: why did BFS fail on a 15-move solution?

Hypothesis: State.history is an unbounded list appended on every move, so two
states with identical shape positions hash differently and `seen` never dedups.
wmlib documents the remedy the model never used — an optional planner_key(state)
hook projecting only the fields that distinguish future dynamics and outcomes.

Test: attach that hook without touching transition(), render(), or any dynamics.
"""

import re
import time

import numpy as np

import wmlib
import world_model as M

HEX = {"0": 0, "4": 4, "5": 5, "9": 9, "a": 10, "b": 11}
rows = []
for line in open("level_0/turn_000.txt"):
    m = re.match(r"y(\d\d): (.*)", line)
    if m:
        rows.append([HEX[c] for c in m.group(2).replace("|", "").split()])
grid0 = np.array(rows, dtype=np.int16)
start = M.init_state(grid0, 0)

tr, tc = np.where(grid0 == 11)
keep = tc != 63
TARGET_CELLS = (tr[keep], tc[keep])


def b_covers_target(state):
    return bool(np.all(M.render(state)[TARGET_CELLS] == 4))


# Confirm the history list is what makes states unique.
s1 = M.transition(M.transition(start, "DOWN"), "UP")
print("after DOWN then UP:")
print(f"  shape positions identical to start? "
      f"{(s1.orig_a_r, s1.orig_a_c, s1.orig_b_r, s1.orig_b_c) == (start.orig_a_r, start.orig_a_c, start.orig_b_r, start.orig_b_c)}")
print(f"  len(history): start={len(start.history)} after={len(s1.history)}")
print(f"  wmlib planner keys equal? {wmlib._planner_key(M, s1) == wmlib._planner_key(M, start)}")

# The projection the model never defined. Positions determine future dynamics
# and the goal; move_count only paints the col-63 counter, which is not a
# target cell, and history only affects UNDO.
M.planner_key = lambda s: (s.orig_a_r, s.orig_a_c, s.orig_b_r, s.orig_b_c)
print(f"  with planner_key hook, keys equal? {wmlib._planner_key(M, s1) == wmlib._planner_key(M, start)}")

print("\nplan_bfs with planner_key attached (dynamics untouched):")
t0 = time.time()
diag = wmlib.plan_bfs_with_diagnostics(M, start, target=b_covers_target, max_nodes=200000)
dt = time.time() - t0
print(f"  status={diag['status']} expanded={diag['expanded_nodes']} "
      f"seen={diag['seen_states']} {dt:.1f}s")
if diag["status"] == "found":
    plan = [a["action"] if isinstance(a, dict) else a for a in diag["plan"]]
    print(f"  plan length {len(plan)}: {plan}")
    s = start
    for a in plan:
        s = M.transition(s, a)
    print(f"  replayed through agent's transition() -> goal reached: {b_covers_target(s)}")
