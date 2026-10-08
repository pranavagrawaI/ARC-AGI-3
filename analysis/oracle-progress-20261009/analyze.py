"""Turn oracle distances (rows.jsonl from oracle.py score) into candidate hill-climbing metrics and test them.

  python3 -B analyze.py

Per run x level row (exact graphs only): d[i] = actions still needed under perfect play after step i (0 = solved).
  progress   = (d0 - min d) / d0           partial credit, defined on stuck levels too
  opt_rate   = share of moves with d -> d-1  (decision quality per action, thinking time excluded)
  back_rate  = share of moves that increase d
  prog_per_min, prog_per_ktok = oracle steps gained per minute / per 1k generated tokens in the level
"""
import json, math, statistics, sys
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "metric-validation-20261008"))
from validate import HIDDEN, logfit, spearman  # noqa: E402


def level_metrics(r):
    d = r["d"]; d0 = r["d0"]
    known = [x for x in d if x is not None]
    if not d0 or not known: return None
    best = min(known)
    moves = [(a, b) for a, b in zip(d, d[1:]) if a is not None and b is not None]
    mins = max((r["t"][-1] - r["t"][0]) / 60, 0.5); ktok = max(sum(r["tok"]) / 1000, 0.5)
    gain = d0 - best
    return dict(progress=gain / d0, opt_rate=sum(b == a - 1 for a, b in moves) / len(moves) if moves else float("nan"),
                back_rate=sum(b > a for a, b in moves) / len(moves) if moves else float("nan"),
                prog_per_min=gain / mins, prog_per_ktok=gain / ktok, solved=r["solved"], covered=r["covered"],
                oracle_eff=(d0 / (len(d) - 1)) if r["solved"] and len(d) > 1 else float("nan"))


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(HERE / "rows.jsonl")]
    rows = [r for r in rows if r["d0"]]          # exact graphs + partial graphs where a shortest path was found (distances are upper bounds)
    M = {}
    for r in rows:
        m = level_metrics(r)
        if m: M[(r["run"], r["game"], r["level"])] = m
    runs = sorted({k[0] for k in M}); keys = ["progress", "opt_rate", "back_rate", "prog_per_min", "prog_per_ktok", "oracle_eff", "covered"]
    agg = {}
    print("## Per run (exact-graph levels only)\n")
    print("| run | level rows | stuck rows | " + " | ".join(keys) + " | stuck progress |"); print("|---" * (len(keys) + 4) + "|")
    for n in runs:
        v = [m for k, m in M.items() if k[0] == n]
        mean = lambda k: statistics.mean([m[k] for m in v if not math.isnan(m[k])] or [float("nan")])
        stuck = [m["progress"] for m in v if not m["solved"]]
        agg[n] = {k: mean(k) for k in keys}
        print(f"| {n} | {len(v)} | {len(stuck)} | " + " | ".join(f"{agg[n][k]:.3f}" for k in keys)
              + f" | {statistics.mean(stuck) if stuck else float('nan'):.2f} |")
    # common level set for the hidden test: (game, level) present for every hidden system
    hs = [n for n in HIDDEN if n in runs]
    common = set.intersection(*({(k[1], k[2]) for k in M if k[0] == n} for n in hs))
    print(f"\n## vs hidden ({len(hs)} systems, {len(common)} common game-levels)\n")
    print("| metric | " + " | ".join(hs) + " | spearman | resid sd |"); print("|---" * (len(hs) + 3) + "|")
    for k in keys[:-1]:
        x = [statistics.mean([M[(n, g, L)][k] for g, L in common if not math.isnan(M[(n, g, L)][k])] or [float('nan')]) for n in hs]
        y = [HIDDEN[n] for n in hs]
        fit = logfit(x, y)[1] if all(v > 0 and not math.isnan(v) for v in x) else float("nan")
        print(f"| {k} | " + " | ".join(f"{v:.3f}" for v in x) + f" | {spearman(x, y):.2f} | {fit:.2f} |")
    print("\n## noise vs signal, paired per game-level (B - A), INCLUDING stuck levels\n")
    print("| A | B | metric | n | mean diff | t | sd |"); print("|---|---|---|---|---|---|---|")
    for a, b in (("dfz-control", "dfz-r2"), ("kf2", "c1"), ("dfz-control", "dfz-s1"), ("kf2", "p1"), ("kf2", "dfz-control"), ("duck36-sub", "q38-commit")):
        for k in ("progress", "opt_rate", "prog_per_min"):
            d = [M[(b, g, L)][k] - M[(a, g, L)][k] for (_, g, L) in [x for x in M if x[0] == a] if (b, g, L) in M
                 and not math.isnan(M[(b, g, L)][k]) and not math.isnan(M[(a, g, L)][k])]
            if len(d) < 3: continue
            sd = statistics.stdev(d); mu = statistics.mean(d)
            print(f"| {a} | {b} | {k} | {len(d)} | {mu:+.3f} | {mu / (sd / math.sqrt(len(d))) if sd else float('nan'):.2f} | {sd:.3f} |")
