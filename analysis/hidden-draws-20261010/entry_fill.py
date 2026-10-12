"""Instrument: context fill at level entry. A level entered with a near-full window loses its start sooner.

  python3 -B entry_fill.py

Fill = total characters of the prompt messages at the first request of the level, as a share of the
largest prompt seen in that game (a proxy for the 118K trim point). Compare outcomes by fill tercile.
The fill is set by earlier levels, so it should be close to independent of this level's difficulty.
"""
import glob, json, re, statistics
from pathlib import Path
from redraw_bound import ROOT, RUNS
from eviction import LOGGED, segments, text_of, STEP


def prompts(path):
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            txt = [text_of(m.get("content")) for m in r["messages"]]
            steps = [int(m.group(1)) for t in txt for m in STEP.finditer(t)]
            if steps:
                out.append((max(steps), sum(map(len, txt))))
    return out


def main():
    rows = []
    for run in LOGGED:
        S = segments(run)
        for f in glob.glob(str(ROOT / RUNS[run] / "*_requests.jsonl")):
            game = Path(f).name[:4]; P = prompts(f)
            if not P:
                continue
            cap = max(c for _, c in P)
            for (g, L), (solved, seq, entry) in S.items():
                if g != game or not seq or L == 0:
                    continue  # L0 starts with an empty window for everyone
                first = next((c for d, c in P if d >= entry), None)
                if first is None:
                    continue
                rows.append(dict(game=g, L=L, solved=solved, fill=first / cap, n=len(seq)))
    rows.sort(key=lambda r: r["fill"])
    k = len(rows) // 3
    for name, part in (("low fill", rows[:k]), ("mid fill", rows[k:2 * k]), ("high fill", rows[2 * k:])):
        print(f"{name:9} n={len(part):3} fill {part[0]['fill']:.2f}-{part[-1]['fill']:.2f}  solved {sum(r['solved'] for r in part) / len(part):5.1%}"
              f"  median actions {statistics.median(r['n'] for r in part):5.0f}")
    # within-game: pair each game's above-median-fill levels with its below-median ones
    diffs = []
    for g in sorted({r["game"] for r in rows}):
        gr = [r for r in rows if r["game"] == g]
        if len(gr) < 4:
            continue
        m = statistics.median(r["fill"] for r in gr)
        hi = [r["solved"] for r in gr if r["fill"] > m]; lo = [r["solved"] for r in gr if r["fill"] <= m]
        if hi and lo:
            diffs.append(sum(hi) / len(hi) - sum(lo) / len(lo))
    print(f"within-game (high - low fill) solve rate: mean {statistics.mean(diffs):+.3f} over {len(diffs)} games; "
          f"negative in {sum(d < 0 for d in diffs)}, positive in {sum(d > 0 for d in diffs)}")


if __name__ == "__main__":
    main()


def same_level():
    """Same (game, level) across runs: when outcomes differ, did the solved attempt enter with lower fill?"""
    by = {}
    for run in LOGGED:
        S = segments(run)
        for f in glob.glob(str(ROOT / RUNS[run] / "*_requests.jsonl")):
            game = Path(f).name[:4]; P = prompts(f)
            if not P:
                continue
            cap = max(c for _, c in P)
            for (g, L), (solved, seq, entry) in S.items():
                if g != game or not seq or L == 0:
                    continue
                first = next((c for d, c in P if d >= entry), None)
                if first is not None:
                    by.setdefault((g, L), []).append((solved, first / cap, run))
    lower = higher = 0
    for key, xs in sorted(by.items()):
        for s1, f1, r1 in xs:
            for s2, f2, r2 in xs:
                if s1 and not s2:
                    lower += f1 < f2; higher += f1 > f2
                    print(f"  {key[0]} L{key[1]}: solved {r1} fill {f1:.2f}  vs stuck {r2} fill {f2:.2f}")
    print(f"discordant pairs: solved attempt had LOWER fill in {lower}, higher in {higher}")


if __name__ == "__main__" and True:
    same_level()
