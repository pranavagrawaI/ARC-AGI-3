"""Per-game replay workload from the 2026-10-01 dfranzen control run.

python3 -B workload.py [run_dir] [out.json]

For each game: the request/response pairs in order with token usage, the
run-clock time of the analysis step that issued them, whether the request's
message list extends the previous one (a break is a trim, where dfranzen's
_PriorityGate hands the slot over), and the minute-40 checkpoint index.
Message bodies stay in the source logs; this file holds only what the
simulator and the replay driver need to index them.
"""
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else
           "../m2-winners-20261001/runs/out-control")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "workload.json")
CHECKPOINT_MIN = 40.0
HDR = re.compile(r"^--- analysis_step=(\d+) \| action=\d+ \| (\d\d:\d\d:\d\d)", re.M)


def message_hashes(messages):
    return [hashlib.sha1(json.dumps(m, sort_keys=True).encode()).hexdigest()[:16]
            for m in messages]


def step_times(transcript):
    """analysis_step -> seconds on the absolute clock (date from serve.log start)."""
    times, last, day = {}, None, 0
    for step, hms in HDR.findall(transcript.read_text(errors="replace")):
        t = datetime.datetime.strptime(hms, "%H:%M:%S")
        s = t.hour * 3600 + t.minute * 60 + t.second
        if last is not None and s + day * 86400 < last:
            day += 1
        last = s + day * 86400
        times.setdefault(int(step), last)
    return times


def load_game(path, times):
    pairs, pending = [], None
    for line in path.open():
        r = json.loads(line)
        if r["event"] == "request":
            pending = r
            continue
        if pending is None:
            raise ValueError(f"{path.name}: response without request")
        u = r["usage"]
        step = pending["analysis_step"]
        pairs.append(dict(
            line_index=len(pairs),
            analysis_step=step,
            request_index_within_turn=pending["request_index_within_turn"],
            t_step=times.get(step),
            prompt_tokens=u["prompt_tokens"],
            cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
            completion_tokens=u["completion_tokens"],
            finish_reason=r.get("finish_reason"),
            msg_hashes=message_hashes(pending["messages"]),
            resp_hashes=message_hashes(r["messages"]),
        ))
        pending = None
    prev = None
    for p in pairs:
        p["extends_previous"] = prev is not None and \
            p["msg_hashes"][:len(prev["msg_hashes"])] == prev["msg_hashes"]
        prev = p
    return pairs


def main():
    games = {}
    for req in sorted(RUN.glob("*_p0_requests.jsonl")):
        g = req.name[:4]
        times = step_times(next(RUN.glob(f"transcripts/{g}*.txt")))
        games[g] = load_game(req, times)
    t0 = min(p["t_step"] for ps in games.values() for p in ps if p["t_step"] is not None)
    out = {"run_dir": str(RUN), "checkpoint_min": CHECKPOINT_MIN, "games": {}}
    for g, ps in games.items():
        for p in ps:
            p["t_min"] = None if p["t_step"] is None else (p["t_step"] - t0) / 60
            del p["t_step"]
        ck = next((i for i, p in enumerate(ps)
                   if p["t_min"] is not None and p["t_min"] >= CHECKPOINT_MIN), None)
        out["games"][g] = {"checkpoint": ck, "requests": ps}
    OUT.write_text(json.dumps(out))
    live = sorted(g for g, v in out["games"].items() if v["checkpoint"] is not None)
    print(f"{len(games)} games; {len(live)} with a request at >= {CHECKPOINT_MIN} min: {' '.join(live)}")
    for g in sorted(games):
        ps = games[g]
        ck = out["games"][g]["checkpoint"]
        trims = sum(not p["extends_previous"] for p in ps[1:])
        tail = ps[ck:] if ck is not None else []
        print(f"{g}: {len(ps):4d} req, first t={ps[0]['t_min']:.1f} min, trims {trims:3d}, "
              f"checkpoint {ck}, after it {len(tail)} req / "
              f"{sum(p['completion_tokens'] for p in tail)} completion tok")


if __name__ == "__main__":
    main()
