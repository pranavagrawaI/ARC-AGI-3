"""Per-turn digest of every stall (>MIN minutes between level clears) that ended in a level.
usage: python3 -B stalldigest.py [MIN]   -> digests/<arm>-<game>-L<level>.txt and stalls.tsv"""
import json, re, sys, os
from pathlib import Path
MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 20
HERE = Path(__file__).parent
RUNS = Path.home() / "projects/arc-agi-3-reference/duck-qwen38/runs/flash-next"
A = json.load(open(HERE.parent / "arm-p-full-20260930/assessment.json"))
out = HERE / "digests"; out.mkdir(exist_ok=True)

def turns(txt):
    for blk in re.split(r"\n(?=--- analysis_step=)", txt):
        m = re.match(r"--- analysis_step=(\d+) \| action=(\d+) \| (\d\d:\d\d):\d\d", blk)
        if not m: continue
        goal = re.search(r"\n- Goal model: (.*)", blk)
        wm = re.search(r"\n- World model: (.*)", blk)
        thinks = re.findall(r"\[THINKING\]\n(.*?)(?=\n\[[A-Z][A-Z :_a-z]+\]\n)", blk, re.S)
        said = re.findall(r"\[ASSISTANT\]\n(.*?)(?=\n\[[A-Z][A-Z :_a-z]+\]\n)", blk, re.S)
        rc = sum(int(x) for x in re.findall(r"reasoning_chars: (\d+)", blk))
        yield dict(step=int(m[1]), action=int(m[2]), clock=m[3],
                   status="act" if "step_executed: True" in blk else ("YIELD" if "turn_time_budget" in blk else "ERR"),
                   calls=blk.count("[MODEL RESPONSE META]"), rc=rc,
                   goal=(goal[1] if goal else ""), wm=(wm[1] if wm else ""),
                   think=[" ".join(t.split()) for t in thinks], said=[" ".join(s.split()) for s in said])

rows = []
for arm, folder in [("P1", "out-p1"), ("P2", "out-p2")]:
    for game, v in A["arms"][arm]["per_game"].items():
        tr = RUNS / folder / "transcripts" / f"{game}_p0.txt"
        T = list(turns(tr.read_text()))
        ev = [json.loads(l) for l in open(RUNS / folder / "artifacts" / f"{game}_p0_events.jsonl")]
        acts = [e for e in ev if e["type"] == "action"]
        prev_s, prev_a = 0.0, 0
        for c in sorted(v["clears"], key=lambda c: c["seconds"]):
            gap = (c["seconds"] - prev_s) / 60
            if gap > MIN:
                win = [t for t in T if prev_a <= t["action"] <= c["action"]]
                post = next((t for t in T if t["action"] > c["action"]), None)
                L = [f"# {arm} {game} level {c['levels']}: stall {prev_s/60:.0f}->{c['seconds']/60:.0f} min, "
                     f"actions {prev_a}->{c['action']} ({c['action']-prev_a}), turns {len(win)}, "
                     f"yields {sum(t['status']=='YIELD' for t in win)}"]
                for t in win:
                    a = [e for e in acts if t["action"] < int(e["action_num"]) <= t["action"] + 60
                         and int(e["analysis_step"]) == t["step"]] if t["status"] == "act" else []
                    L.append(f"\n[{t['clock']} a{t['action']} {t['status']} calls={t['calls']} reason={t['rc']//1000}k]")
                    if t["goal"]: L.append("  GOAL(carried): " + t["goal"][:300])
                    for s in t["said"][:2]: L.append("  SAID: " + s[:400])
                    if t["think"]: L.append("  THINK0: " + t["think"][0][:450])
                    if len(t["think"]) > 1: L.append("  THINK-LAST: " + t["think"][-1][:450])
                    if a: L.append("  ACTIONS: " + " ".join(e["action_display"] + ("[LEVEL+]" if e["level_completed"] == "True" else "") + ("[GAMEOVER]" if e["game_over"] == "True" else "") for e in a))
                if post and post["think"]:
                    L.append(f"\n[POST {post['clock']} a{post['action']}]\n  THINK0: " + post["think"][0][:1500])
                (out / f"{arm}-{game[:4]}-L{c['levels']}.txt").write_text("\n".join(L))
                rows.append((arm, game[:4], c["levels"], round(prev_s/60), round(c["seconds"]/60), round(gap),
                             c["action"] - prev_a, len(win), sum(t["status"] == "YIELD" for t in win)))
            prev_s, prev_a = c["seconds"], c["action"]
with open(HERE / "stalls.tsv", "w") as f:
    f.write("arm\tgame\tlevel\tfrom_min\tto_min\tgap_min\tactions\tturns\tyields\n")
    for r in rows: f.write("\t".join(map(str, r)) + "\n")
for r in sorted(rows, key=lambda r: -r[5]): print(*r, sep="\t")
