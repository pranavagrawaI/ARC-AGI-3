"""Game overs per game and what followed. usage: gameovers.py <artifacts dir>"""
import json, glob, sys, os
tot_go = tot_after = 0
for f in sorted(glob.glob(sys.argv[1] + "/*events.jsonl")):
    acts = [json.loads(l) for l in open(f)]; acts = [e for e in acts if e["type"] == "action"]
    go = [i for i, a in enumerate(acts) if str(a.get("game_over")) == "True"]
    lv = [i for i, a in enumerate(acts) if str(a.get("level_completed")) == "True"]
    if go:
        after = [i for i in lv if i > go[0]]
        tot_go += 1; tot_after += bool(after)
        print(f"{os.path.basename(f)[:4]}: game_overs at actions {[i+1 for i in go]}, levels at {[i+1 for i in lv]}, levels after first GO: {len(after)}; actions after first GO: {len(acts)-go[0]-1}")
print(f"games with a game over: {tot_go}; of them, a level after it: {tot_after}")
