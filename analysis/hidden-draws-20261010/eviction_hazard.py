"""Does losing a level's start from context lower its solve hazard? Natural experiment, offline.

  python3 -B eviction_hazard.py

When a level's opening steps leave the prompt depends on how full the context was at level entry
(dfranzen trims 118K -> ~59K), which is roughly independent of the level's difficulty. At each
actions-in-level bin, compare P(solve within the next W actions) for attempts whose entry is already
evicted against attempts whose entry is still visible. Levels still open at the bin start are at risk.
"""
import glob
from pathlib import Path
from redraw_bound import ROOT, RUNS
from eviction import LOGGED, visibility, segments

W = 20
BINS = [(0, 20), (20, 40), (40, 60), (60, 90), (90, 140), (140, 400)]


def timelines():
    rows = []
    for run in LOGGED:
        S = segments(run)
        for f in glob.glob(str(ROOT / RUNS[run] / "*_requests.jsonl")):
            game = Path(f).name[:4]; vis = visibility(f)
            for (g, L), (solved, seq, entry) in S.items():
                if g != game or not seq:
                    continue
                last = seq[-1][2]
                ev = next((d for d, v in vis if d >= entry and v > entry), None)  # first decision step with entry gone
                rows.append(dict(run=run, game=g, L=L, solved=solved, n=len(seq), ev=None if ev is None else ev - entry))
    return rows


def main():
    rows = timelines()
    print(f"attempts {len(rows)}; ever evicted {sum(r['ev'] is not None for r in rows)}")
    print(f"{'bin':>9} | {'visible: at risk  solve%':>26} | {'evicted: at risk  solve%':>26}")
    tot = {True: [0, 0], False: [0, 0]}
    for a, b in BINS:
        cell = {True: [0, 0], False: [0, 0]}
        for r in rows:
            if r["n"] <= a and r["solved"]:
                continue  # solved before this bin
            if r["n"] <= a and not r["solved"]:
                continue  # ended unsolved before this bin
            evicted = r["ev"] is not None and r["ev"] <= a
            hit = r["solved"] and r["n"] <= a + W
            cell[evicted][0] += 1; cell[evicted][1] += hit
            tot[evicted][0] += 1; tot[evicted][1] += hit
        fmt = lambda c: f"{c[0]:>8}  {100 * c[1] / c[0]:6.1f}%" if c[0] else f"{0:>8}     -  "
        print(f"{a:>4}-{b:<4} | {fmt(cell[False]):>26} | {fmt(cell[True]):>26}")
    print("pooled  ", {('evicted' if k else 'visible'): f"{v[1]}/{v[0]} = {100 * v[1] / max(1, v[0]):.1f}%" for k, v in tot.items()})


if __name__ == "__main__":
    main()
