"""Per run, per game: the minute (of the game's own clock) at which each level was completed.

Uses transcript turn headers (action counter + wall time) and the events stream (score after each action),
the same evidence as analysis/m2-winners-20261001/levels_at_time.py, computed once per game.
"""
import datetime, json, re, sys
from pathlib import Path

HDR = re.compile(r"^--- analysis_step=\d+ \| action=(\d+) \| (\d\d:\d\d:\d\d)", re.M)


def level_minutes(out, game):
    t = next(Path(out).glob(f"transcripts/{game}*.txt"))
    hdr = HDR.findall(t.read_text(errors="replace"))
    if not hdr:
        return []
    ts = [datetime.datetime.strptime(h[1], "%H:%M:%S") for h in hdr]
    days, secs = 0, []
    for i, x in enumerate(ts):
        if i and x < ts[i - 1]:
            days += 1
        secs.append((x - ts[0]).total_seconds() + 86400 * days)
    # turn k starts at secs[k] with actions_executed = action-1; an action n was taken during the
    # last turn whose start counter <= n-1, so it happened no later than the next turn start.
    starts = [(int(a) - 1, s) for (a, _), s in zip(hdr, secs)]
    ev = next(Path(out).glob(f"artifacts/{game}*_events.jsonl"))
    best, out_min = 0, []
    for line in ev.open():
        e = json.loads(line)
        if e["type"] != "action":
            continue
        sc = int(e.get("score") or 0)
        if sc > best:
            n = e["action_num"]
            later = [s for a, s in starts if a >= n]
            when = later[0] if later else secs[-1]
            out_min += [when / 60] * (sc - best)
            best = sc
    return out_min


def run(out):
    games = sorted(p.name.split("_p0")[0] for p in Path(out).glob("transcripts/*.txt"))
    return {g[:4]: level_minutes(out, g[:4]) for g in games}


if __name__ == "__main__":
    spec = json.loads(sys.argv[1])
    print(json.dumps({k: run(v) for k, v in spec.items()}))
