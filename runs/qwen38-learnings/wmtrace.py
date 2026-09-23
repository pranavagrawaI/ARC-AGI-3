"""Per-game belief trace: the world model carried into each analyzer turn, deduplicated.
usage: python3 wmtrace.py <artifacts dir> <out dir>"""
import json, glob, re, sys, os
art, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for f in sorted(glob.glob(art + "/*events.jsonl")):
    g = os.path.basename(f)[:4]
    lines, prev, acts = [], None, []
    for l in open(f):
        e = json.loads(l)
        if e["type"] == "action":
            acts.append(e)
            if str(e.get("level_completed")) == "True":
                lines.append(f"  >>> LEVEL COMPLETED at action {e['action_num']}")
            continue
        if e["type"] != "analysis":
            continue
        t = e["transcript"]
        i = t.find("Working world model carried from earlier turns:")
        wm = t[i + 48 : t.find("- Revise any item above", i)].strip() if i >= 0 else "(none)"
        tail = t[t.rfind("[ANALYZER STATUS]"):]
        st = "act" if "step_executed: True" in tail else ("YIELD" if "turn_time_budget" in tail else "ERR")
        m = re.search(r"\| (\d\d:\d\d):\d\d \|", t[:200])
        hdr = f"[{m.group(1) if m else '?'} a{e['action_num']} L{e['level']} {st}]"
        if wm != prev:
            lines.append(hdr + "\n" + "\n".join("    " + x for x in wm.splitlines()))
            prev = wm
        else:
            lines.append(hdr + " (same WM)")
    last = " ".join(a["action_display"] for a in acts[-12:])
    lines.append(f"  last 12 actions: {last}")
    open(f"{out}/{g}.txt", "w").write("\n".join(lines))
    print(g, sum(len(x) for x in lines))
