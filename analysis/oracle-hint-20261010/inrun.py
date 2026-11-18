"""In-run stuck segments of the 40 hard levels, for asking: after many minutes in a full game, is the model still
wrong about the rules (as in the 1,200 s sprints, FAILURES.md), or does it fail some other way?

  python3 -B inrun.py table                         one row per (stuck run, level): minutes, actions, tokens
  python3 -B inrun.py read <game> <level> [run] [per] [n]
                                                    hint (if any), then n thinking blocks sampled across the segment
                                                    and the last 3, each truncated to per chars

Segment = everything the run did after entering level L (the run never completed L, so it runs to the end of the game).
Time and tokens come from benchmark.json history (wallclock_seconds is cumulative from game start).
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
M2 = HERE.parent / "m2-winners-20261001" / "runs"
RUNS = {"dfz-control": M2 / "out-control", "dfz-r2": M2 / "out-control-r2", "dfz-s1": M2 / "out-s1"}
ST = json.loads((HERE / "starts.json").read_text())
HINTS = json.loads((HERE / "hints.json").read_text())


def P(x):
    return json.loads(x) if isinstance(x, str) else x


def bench(run):
    b = json.loads((RUNS[run] / "benchmark.json").read_text())
    return {r["game_id"][:4]: r for r in b["game_runs"]}


def thinking_blocks(run, game, level):
    t = next((RUNS[run] / "transcripts").glob(f"{game}-*_p0.txt")).read_text(errors="replace")
    heads = [(m.start(), int(m.group(1))) for m in re.finditer(r"Current state: step \d+, level (\d+)\.", t)]
    mine = [p for p, k in heads if k == level + 1]
    nxt = [p for p, k in heads if k > level + 1]
    chunk = t[(mine[0] if mine else 0):(nxt[0] if nxt else len(t))]
    return [re.sub(r"\s+", " ", m.group(1)).strip()
            for m in re.finditer(r"\[THINKING\]\n(.*?)(?=\n\[TOOL CALL|\n\[USER PROMPT\]|\n\[MODEL RESPONSE|\Z)", chunk, re.S)]


def segment(run, game, level):
    r = bench(run)[game]
    h, apl = P(r["history"]), P(r["actions_per_level"])
    k = sum(apl[:level])
    seg = h[k:]
    t0 = h[k - 1]["wallclock_seconds"] if k else 0.0
    return dict(minutes=(h[-1]["wallclock_seconds"] - t0) / 60, game_minutes=h[-1]["wallclock_seconds"] / 60,
                entry_minute=t0 / 60, actions=len(seg), tokens=sum(x.get("generated_tokens", 0) for x in seg),
                resets=sum(x["action"]["id"] == "RESET" for x in seg), state=r["state"])


def cmd_table():
    rows = []
    for s in ST:
        for run in s["stuck_runs"]:
            seg = segment(run, s["game"], s["level"])
            tb = thinking_blocks(run, s["game"], s["level"])
            rows.append(dict(key=f"{s['game']}-L{s['level']}", klass=s["klass"], run=run, **seg,
                             think_blocks=len(tb), think_kchars=sum(map(len, tb)) // 1000))
    print(f"{'level':9s} {'class':7s} {'run':12s} entry  min  acts resets  ktok  blocks kchars state")
    for r in rows:
        print(f"{r['key']:9s} {r['klass']:7s} {r['run']:12s} {r['entry_minute']:5.0f} {r['minutes']:4.0f} {r['actions']:5d} "
              f"{r['resets']:5d} {r['tokens'] // 1000:5d} {r['think_blocks']:6d} {r['think_kchars']:6d} {r['state']}")
    (HERE / "evidence" / "inrun-table.json").write_text(json.dumps(rows, indent=1))


def cmd_read(game, level, run=None, per=500, n=6):
    level, per, n = int(level), int(per), int(n)
    s = next(x for x in ST if x["game"] == game and x["level"] == level)
    h = HINTS.get(f"{game}-L{level}")
    print(f"=== {game} L{level} ({s['klass']}); stuck {s['stuck_runs']} solved {s['solved_runs']}")
    if h:
        print(f"HINT: {h['goal']}\nROUTE: {len(h['route'])} actions")
    for r in ([run] if run else s["stuck_runs"]):
        seg = segment(r, game, level)
        tb = thinking_blocks(r, game, level)
        print(f"\n--- {r}: {seg['minutes']:.0f} min on level from minute {seg['entry_minute']:.0f}, {seg['actions']} actions, "
              f"{seg['resets']} resets, {seg['tokens'] // 1000}K tokens, {len(tb)} thinking blocks, end state {seg['state']}")
        idx = sorted({int(i * (len(tb) - 3) / max(n, 1)) for i in range(n)} | set(range(max(len(tb) - 3, 0), len(tb))))
        for i in idx:
            b = tb[i]
            print(f"[{i}/{len(tb)}] {b[:per]}{' ...' if len(b) > per else ''}")


if __name__ == "__main__":
    {"table": cmd_table, "read": cmd_read}[sys.argv[1]](*sys.argv[2:])
