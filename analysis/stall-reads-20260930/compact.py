"""Compact view of a stall digest: one line per turn, goal on change, full final-turn thinking.
usage: python3 -B compact.py <digest file> [final_chars]"""
import re, sys
txt = open(sys.argv[1]).read(); K = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
post = txt.split("\n[POST ")[1] if "\n[POST " in txt else ""
txt = txt.split("\n[POST ")[0]
blocks = re.split(r"\n\n(?=\[)", txt)
print(blocks[0]); last_goal = None
for b in blocks[1:]:
    head = b.splitlines()[0]
    f = {m[0]: m[1] for m in re.findall(r"\n  (GOAL\(carried\)|SAID|THINK0|THINK-LAST|ACTIONS): (.*)", b)}
    g = f.get("GOAL(carried)")
    if g and g != last_goal: print("   >> GOAL now: " + g[:220]); last_goal = g
    print(head, "|", (f.get("SAID") or f.get("THINK0", ""))[:230], ("|| ACT " + f["ACTIONS"][:150]) if "ACTIONS" in f else "")
print("\n--- FINAL TURN ---\n" + blocks[-1][:K])
print("\n--- POST-CLEAR ---\n[" + post[:K])
