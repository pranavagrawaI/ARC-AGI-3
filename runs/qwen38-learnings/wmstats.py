"""How often does the carried world model change? usage: wmstats.py <artifacts dir>"""
import json, glob, sys, os
T = N = C = E = 0
for f in sorted(glob.glob(sys.argv[1] + "/*events.jsonl")):
    prev = None; n = ch = empty = 0
    for l in open(f):
        e = json.loads(l)
        if e["type"] != "analysis": continue
        t = e["transcript"]; i = t.find("Working world model carried from earlier turns:")
        wm = t[i:t.find("- Revise any item above", i)] if i >= 0 else ""
        n += 1; empty += (i < 0); ch += (prev is not None and wm != prev); prev = wm
    T += 1; N += n; C += ch; E += empty
    print(f"{os.path.basename(f)[:4]} turns {n:3} WM changes {ch:3} turns with no WM {empty:3}")
print(f"ALL: {N} turns, WM changed on {C} ({100*C/N:.0f}%), no WM on {E} ({100*E/N:.0f}%)")
