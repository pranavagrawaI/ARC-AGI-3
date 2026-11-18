"""Actions to clear each (game, level) across independent full runs of the KF-family harness.
Right-censored when the run ended on that level. usage: python3 -B levelcosts.py"""
import json, glob, os, statistics as st
from pathlib import Path
RUNS = Path.home() / "projects/arc-agi-3-reference/duck-qwen38/runs/flash-next"
ARMS = ["kf2", "c1", "r1", "g1", "g2", "p1", "p2", "k2"]
cost = {}  # (game, level) -> list of (actions, cleared)
for arm in ARMS:
    for f in glob.glob(str(RUNS / f"out-{arm}/artifacts/*_events.jsonl")):
        g = os.path.basename(f)[:4]; start = 0; lvl = 1; n = 0
        for l in open(f):
            e = json.loads(l)
            if e["type"] != "action": continue
            n = int(e["action_num"])
            if str(e["level_completed"]) == "True":
                cost.setdefault((g, lvl), []).append((n - start, True, arm)); start = n; lvl += 1
        if str(e.get("state")) != "WIN": cost.setdefault((g, lvl), []).append((n - start, False, arm))
json.dump({f"{g}:{l}": v for (g, l), v in cost.items()}, open("levelcosts.json", "w"))
print("level  cleared/runs  actions-to-clear (sorted; * = censored, not cleared)")
for (g, l), v in sorted(cost.items()):
    c = sorted(a for a, ok, _ in v if ok)
    if len(c) < 2: continue
    cens = sorted(a for a, ok, _ in v if not ok)
    print(f"{g} L{l}  {len(c)}/{len(v)}  {c}  cens={cens}  max/min={c[-1]/max(1,c[0]):.1f}")
