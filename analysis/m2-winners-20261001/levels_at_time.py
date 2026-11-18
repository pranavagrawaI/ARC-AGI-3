"""Levels per game reached by minute T of each game's own clock, from transcript turn headers + action events.
   python3 levels_at_time.py T_MIN run_dir [run_dir ...]"""
import json, re, sys, datetime
from pathlib import Path
def at(out, tmin):
    res = {}
    for t in sorted(Path(out).glob("transcripts/*.txt")):
        g = t.name[:4]
        hdr = re.findall(r"^--- analysis_step=\d+ \| action=(\d+) \| (\d\d:\d\d:\d\d)", t.read_text(errors="replace"), re.M)
        if not hdr: res[g] = 0; continue
        ts = [datetime.datetime.strptime(h[1], "%H:%M:%S") for h in hdr]
        base = ts[0]; days = 0; secs = []
        for i, x in enumerate(ts):
            if i and x < ts[i-1]: days += 1
            secs.append((x - base).total_seconds() + 86400 * days)
        # last turn started before the cutoff -> action counter at that point (actions executed = action-1)
        after = [int(a) - 1 for (a, _), s in zip(hdr, secs) if s > tmin * 60]
        n = min(after) if after else 10**9   # actions executed before the first turn that starts after the cutoff
        ev = next(Path(out).glob(f"artifacts/{g}*_events.jsonl"))
        sc = 0
        for l in ev.open():
            e = json.loads(l)
            if e["type"] == "action" and e["action_num"] <= n: sc = max(sc, int(e.get("score") or 0))
        res[g] = sc
    return res
if __name__ == "__main__":
    T = float(sys.argv[1]); runs = {Path(r).name: at(r, T) for r in sys.argv[2:]}
    games = sorted(set().union(*runs.values()))
    print("game " + " ".join(f"{n[-10:]:>10s}" for n in runs))
    for g in games: print(f"{g} " + " ".join(f"{runs[n].get(g, 0):10d}" for n in runs))
    print("SUM  " + " ".join(f"{sum(r.values()):10d}" for r in runs.values()))
