"""Levels reached by our runs within the same per-game action count a winner used (action-matched quality)."""
import json, sys
from pathlib import Path
OURS = Path.home()/"projects/arc-agi-3-reference/duck-qwen38/runs/flash-next"
W = Path(__file__).parent

def curve(events):
    """level reached after k actions -> list indexed by action count."""
    lv=[0]; 
    for line in events.open():
        e=json.loads(line)
        if e["type"]!="action": continue
        lvl=int(e.get("score") or 0)
        lv.append(max(lv[-1], lvl))
    return lv

def runs(out):
    return {p.name[:4]: curve(p) for p in (out/"artifacts").glob("*_p0_events.jsonl")}

for winner in sys.argv[1:]:
    w = runs(W/winner/"out")
    print(f"\n== {winner}: game  winner(levels/actions) | ours levels at same action count")
    arms = {a: runs(OURS/a) for a in ("out-kf2","out-c1","out-p1","out-p2")}
    tot={"w":0, **{a:0 for a in arms}}; totfull={a:0 for a in arms}
    for g,c in sorted(w.items()):
        n=len(c)-1
        if n==0: continue
        row=f"{g}  {c[-1]}/{n:4d} |"
        tot["w"]+=c[-1]
        for a,r in arms.items():
            oc=r.get(g)
            if oc is None: row+=f" {a}: -   "; continue
            v=oc[min(n,len(oc)-1)]; tot[a]+=v; totfull[a]+=oc[-1]
            row+=f" {a[4:]}:{v}(full {oc[-1]}/{len(oc)-1})"
        print(row)
    print("TOTAL matched:", tot, " ours full-run:", totfull)
