"""Per-attempt digest of a Duck analyzer transcript: time, thinking, code, executed actions, outcomes.
usage: digest.py <transcript.txt> [--think N] [--code N] [--from STEP] [--to STEP]
"""
import argparse, re, sys

ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--think", type=int, default=350)
ap.add_argument("--code", type=int, default=500); ap.add_argument("--from", dest="lo", type=int, default=0); ap.add_argument("--to", dest="hi", type=int, default=10**9)
a = ap.parse_args()
txt = open(a.path, encoding="utf-8").read()
att = re.split(r"(?m)^--- analysis_step=(\d+) \| action=(\d+) \| ([0-9:]+) \| tool-agent ---$", txt)
for i in range(1, len(att) - 3, 4):
    step, act, clock, body = int(att[i]), att[i + 1], att[i + 2], att[i + 3]
    if not (a.lo <= step <= a.hi):
        continue
    parts = re.split(r"(?m)^\[([A-Z :a-z]+)\]$", body)
    secs = [(parts[j], parts[j + 1].strip()) for j in range(1, len(parts) - 1, 2)]
    fu = next((c for l, c in secs if l == "USER PROMPT"), "")
    head = " ".join(l for l in fu.splitlines()[:6] if l.strip())[:260]
    el = re.findall(r"run_elapsed_seconds: ([0-9.]+)", body)
    print(f"\n=== step {step} | action {act} | {clock} | elapsed {float(el[-1]):.0f}s" if el else f"\n=== step {step} | action {act} | {clock}")
    print("  PROMPT:", head)
    for l, c in secs:
        if l == "THINKING":
            print("  THINK:", " ".join(c.split())[: a.think])
        elif l == "ASSISTANT":
            print("  TEXT:", " ".join(c.split())[: a.think])
        elif l.startswith("TOOL CALL"):
            m = re.search(r"<parameter=code>\n?(.*?)</parameter>", c, flags=re.S)
            code = (m.group(1) if m else c).strip()
            print("  CODE:", " | ".join(x.strip() for x in code.splitlines() if x.strip())[: a.code])
        elif l.startswith("TOOL RESULT"):
            ex = re.findall(r"action_display: (.*)", c); lv = re.findall(r"^level: (\d+)", c, flags=re.M)
            ch = re.findall(r"board_changed: (\w+)", c); lc = "LEVEL+" if "level_completed: true" in c else ("GAME_OVER" if "game_over: true" in c else "")
            out = c[:220].replace("\n", " | ") if not ex else f"actions={ex} changed={ch} level={lv[-1] if lv else '?'} {lc}"
            if "Traceback" in c or "Error" in c[:300]:
                out += " || ERR: " + " ".join(c.split())[:200]
            print("  RESULT:", out)
        elif l == "ANALYZER STATUS":
            m = re.search(r"message: (.*)", c); print("  STATUS:", m.group(1)[:120] if m else "")
