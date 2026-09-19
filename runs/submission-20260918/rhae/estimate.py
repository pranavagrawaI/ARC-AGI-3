"""RHAE estimate of the commit run (25 public games) against the human baselines
shipped in the competition's environment_files/*/metadata.json.

level_score = min((human/agent)**2, 1.15); game = sum(i * level_i) / sum(1..n); total = mean over games.
Baselines were pulled from Kaggle competition data on 2026-09-20 (only metadata.json, not the game code).
"""
import glob, json, os, sys

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "..", "kaggle-output", "artifacts")
base = {}
for f in glob.glob(os.path.join(here, "environment_files", "*", "*", "metadata.json")):
    m = json.load(open(f))
    base[m["game_id"][:4]] = m["baseline_actions"]

rows = []
for f in sorted(glob.glob(os.path.join(out, "*events.jsonl"))):
    g = os.path.basename(f).split("-")[0]
    n = 0; hits = []; last = 0
    for line in open(f):
        e = json.loads(line)
        if e.get("type") != "action":
            continue
        n += 1
        if e.get("level_completed"):
            hits.append(n - last); last = n
    hb = base[g]; L = len(hb); W = L * (L + 1) / 2
    score = sum((i + 1) * min((hb[i] / a) ** 2, 1.15) for i, a in enumerate(hits)) / W * 100
    ceiling = sum(i + 1 for i in range(len(hits))) / W * 100
    rows.append((g, len(hits), L, hits, hb[: len(hits)], score, ceiling))

print(f"{'game':5} {'lv/n':>5} {'agent actions/level':22} {'human baseline':22} {'game%':>6} {'ceiling%':>8}")
for g, k, L, hits, hb, s, c in rows:
    print(f"{g:5} {k:>2}/{L:<2} {str(hits):22} {str(hb):22} {s:6.2f} {c:8.2f}")
n = len(rows)
print(f"\nestimated RHAE, 25 public games: {sum(r[5] for r in rows)/n:.2f}%")
print(f"same levels at human efficiency:  {sum(r[6] for r in rows)/n:.2f}%")
print("Kaggle public LB (55 hidden-public games), ref 56355127: 0.67%")
