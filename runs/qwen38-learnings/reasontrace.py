"""Per-turn reasoning digest: assistant text + tail of thinking + executed actions.
usage: reasontrace.py <artifacts dir> <out dir> [think_tail_chars]"""
import json, glob, re, sys, os
art, out = sys.argv[1], sys.argv[2]; K = int(sys.argv[3]) if len(sys.argv) > 3 else 700
os.makedirs(out, exist_ok=True)
def section(t, name):
    i = t.find(f"[{name}]")
    if i < 0: return ""
    j = t.find("\n[", i + len(name) + 2)
    return t[i + len(name) + 2 : j if j > 0 else None].strip()
for f in sorted(glob.glob(art + "/*events.jsonl")):
    g = os.path.basename(f)[:4]; L = []; pend = []
    for l in open(f):
        e = json.loads(l)
        if e["type"] == "action":
            pend.append(e["action_display"] + (" [LEVEL+]" if str(e.get("level_completed")) == "True" else "") + (" [GAMEOVER]" if str(e.get("game_over")) == "True" else ""))
            continue
        if e["type"] != "analysis": continue
        if pend: L.append("   actions: " + " ".join(pend)); pend = []
        t = e["transcript"]; tail = t[t.rfind("[ANALYZER STATUS]"):]
        st = "act" if "step_executed: True" in tail else ("YIELD" if "turn_time_budget" in tail else "ERR")
        th = section(t, "THINKING"); asst = section(t, "ASSISTANT")
        m = re.search(r"\| (\d\d:\d\d):\d\d \|", t[:200])
        L.append(f"[{m.group(1) if m else '?'} a{e['action_num']} L{e['level']} {st} think={len(th)}c]")
        if asst: L.append("   said: " + " ".join(asst.split())[:500])
        if th: L.append("   think-tail: " + " ".join(th.split())[-K:])
    if pend: L.append("   actions: " + " ".join(pend))
    open(f"{out}/{g}.txt", "w").write("\n".join(L))
