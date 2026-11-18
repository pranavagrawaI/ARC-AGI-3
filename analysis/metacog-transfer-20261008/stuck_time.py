"""Per game: wall minutes of the last level-up vs the end, and share of the game's wall spent on the final (never-solved) level.
   python3 stuck_time.py run_dir"""
import json, re, sys, datetime
from pathlib import Path
out = Path(sys.argv[1]); rows = []
for t in sorted(out.glob("transcripts/*.txt")):
    g = t.name[:4]
    hdr = re.findall(r"^--- analysis_step=\d+ \| action=(\d+) \| (\d\d:\d\d:\d\d)", t.read_text(errors="replace"), re.M)
    if not hdr: continue
    ts = [datetime.datetime.strptime(h[1], "%H:%M:%S") for h in hdr]; days = 0; secs = []
    for i, x in enumerate(ts):
        if i and x < ts[i-1]: days += 1
        secs.append((x - ts[0]).total_seconds() + 86400 * days)
    acts = [int(a) for a, _ in hdr]
    ev = next(out.glob(f"artifacts/{g}*_events.jsonl")); sc = 0; lvl_at = []
    for l in ev.open():
        e = json.loads(l)
        if e["type"] == "action":
            s = int(e.get("score") or 0)
            if s > sc: sc = s; lvl_at.append(e["action_num"])
    def t_of(a):  # first turn header whose action counter exceeds a
        for x, s in zip(acts, secs):
            if x > a: return s
        return secs[-1]
    end = secs[-1]; last = t_of(lvl_at[-1]) if lvl_at else 0.0
    rows.append((g, sc, round(last/60), round(end/60), acts[-1] - (lvl_at[-1] if lvl_at else 0), (end-last)/end if end else 0))
tot_end = sum(r[3] for r in rows); tot_stuck = sum(r[3]-r[2] for r in rows)
print("game lv last_lvl_min end_min actions_on_final_level stuck_share")
for r in sorted(rows, key=lambda r: -r[5]): print(f"{r[0]} {r[1]:2d} {r[2]:4d} {r[3]:4d} {r[4]:5d} {r[5]:.2f}")
print(f"wall on final unsolved level: {tot_stuck}/{tot_end} min = {tot_stuck/tot_end:.0%}")
