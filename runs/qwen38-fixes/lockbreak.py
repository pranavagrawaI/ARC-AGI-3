"""Lock-break rate: does the actor reach for something new right after a question?
At each point (Q: a delivered question; control: every 2nd analyzer turn after the 2nd turn of a
level, the questioner's own cadence), look at the next WINDOW actions in the same level. A *break*
= one of them touches a target never used earlier in that level. Target = arrow/SPACE name, or the
8x8 board cell of a MOUSE click. Also reports a level completed within the window.
usage: lockbreak.py <arm dir> [--control]"""
import glob, json, os, re, sys
WINDOW = 10

def target(a):
    m = re.match(r"MOUSE\(row=(\d+), col=(\d+)\)", a)
    return f"M{int(m.group(1))//8},{int(m.group(2))//8}" if m else a

def episodes(d):
    for f in sorted(glob.glob(f"{d}/artifacts/*events.jsonl")):
        stem = os.path.basename(f).replace("_events.jsonl", "")
        yield stem, [json.loads(l) for l in open(f)]

def points_q(d, stem):
    tr = f"{d}/transcripts/{stem}_critic.jsonl"
    if not os.path.exists(tr): return []
    return [json.loads(l)["action_num"] - 1 for l in open(tr) if json.loads(l).get("question")]

def points_control(ev):
    pts, level, turns = [], None, 0; n = 0
    for e in ev:
        if e["type"] == "action": n += 1; continue
        if e["type"] != "analysis": continue
        lv = int(e.get("level") or 1)
        if lv != level: level, turns = lv, 0
        turns += 1
        if turns > 2 and turns % 2 == 1: pts.append(n)
    return pts

def score(d, control):
    hits = pts_total = lvl = 0
    for stem, ev in episodes(d):
        acts = [e for e in ev if e["type"] == "action"]
        pts = points_control(ev) if control else points_q(d, stem)
        for p in pts:
            after = acts[p:p + WINDOW]
            if not after: continue
            level = int(after[0].get("level") or 1)
            before = {target(a["action_display"]) for a in acts[:p] if int(a.get("level") or 1) == level}
            after = [a for a in after if int(a.get("level") or 1) == level] or after[:1]
            pts_total += 1
            hits += any(target(a["action_display"]) not in before for a in after)
            lvl += any(str(a.get("level_completed")) == "True" for a in after)
    return pts_total, hits, lvl

d = sys.argv[1]; control = "--control" in sys.argv
n, h, l = score(d, control)
print(f"{d} [{'control points' if control else 'question points'}]: n={n}, break rate {100*h/max(n,1):.0f}%, level within {WINDOW} actions {100*l/max(n,1):.1f}%")
