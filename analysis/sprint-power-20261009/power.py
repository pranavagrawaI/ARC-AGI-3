"""Offline power analysis for checkpoint sprints, from the three byte-identical dfranzen runs.

  python3 -B power.py

A sprint = resume at a start state, run T minutes, score 1 if the level is solved. Paired A/B on the same N starts.
Under the null (A = B) the difference of means has variance 2*E[p(1-p)]/N, where p is the per-start solve probability.
Level-entry states are replicated across runs, so k-of-n gives an unbiased E[p(1-p)] = k(n-k)/(n(n-1)).
Mid-level starts are unreplicated: q(1-q) of the pooled rate is an upper bound.
Compared with a full 25-game pass, whose identical-run total-levels diff has sd ~11 on ~105 levels (memory: incumbent-dfranzen-m2).
"""
import math, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "metric-validation-20261008"))
from validate import RUNS
from stuck import levels

NAMES = ["dfz-control", "dfz-r2", "dfz-s1"]   # s1 is an arm, near-identical (inconclusive); try both with and without it
STREAMS, PASS_GPU_H, PASS_REL_SD = 10, 2.2, 11 / 105 * math.sqrt(1)   # full-pass pair: sd of diff ~11 levels


def outcome(l, m, T):
    """1/0 if open at minute m and (not) solved by m+T; None if not open at m or censored."""
    if l["sec"] <= m * 60 and l["done"]: return None
    if l["sec"] <= m * 60: return None
    if l["done"]: return int(l["sec"] <= (m + T) * 60)
    return 0 if l["sec"] >= (m + T) * 60 else None


def main(names):
    Lv = {n: levels(RUNS[n]) for n in names}
    keys = sorted(set().union(*Lv.values()))
    print(f"\nruns: {names}")
    print("start | T min | starts | mean p | E[p(1-p)] | rel SE of A/B diff @N=100 | N for 10%% rel SE | GPU-h (2 arms, %d streams) | full-pass-pair equiv" % STREAMS)
    for m in (0, 10, 30):
        for T in (5, 10, 20, 30):
            ps, pq, n_starts = [], [], 0
            for k in keys:
                ys = [y for n in names if (l := Lv[n].get(k)) and (y := outcome(l, m, T)) is not None]
                if not ys: continue
                n_starts += 1
                ps.append(statistics.mean(ys))
                if len(ys) >= 2:
                    s = sum(ys); pq.append(s * (len(ys) - s) / (len(ys) * (len(ys) - 1)))
            if not ps: continue
            mp, v = statistics.mean(ps), statistics.mean(pq) if pq else float("nan")
            if not mp or not v == v: continue
            rel100 = math.sqrt(2 * v / 100) / mp
            N10 = 2 * v / (0.10 * mp) ** 2
            gpu_h = 2 * N10 * T / 60 / STREAMS
            print(f"  m={m:2d} | {T:3d} | {n_starts:4d} ({len(pq)} repl) | {mp:.2f} | {v:.3f} | {rel100:.0%} | {N10:5.0f} | {gpu_h:5.1f} | {gpu_h / (2 * PASS_GPU_H):.2f}x")


if __name__ == "__main__":
    main(NAMES); main(NAMES[:2])
