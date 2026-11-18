"""Generated tokens on solved levels vs the final unsolved level, per game, from benchmark.json history.
   python3 stuck_tokens.py benchmark.json"""
import json, sys
b = json.load(open(sys.argv[1])); S = U = 0; rows = []
for g in b["game_runs"]:
    h = g["history"]; apl = g["actions_per_level"]; n = g["levels_completed"]
    solved_actions = sum(apl[:n]); toks = [x["generated_tokens"] for x in h]
    s = sum(toks[:solved_actions]); u = sum(toks[solved_actions:])
    S += s; U += u; rows.append((g["game_id"][:4], n, s // 1000, u // 1000, len(h) - solved_actions))
print("game lv solvedK finalK final_actions")
for r in sorted(rows, key=lambda r: -r[3]): print(*r)
print(f"tokens on solved levels {S/1e6:.1f}M, on final unsolved level {U/1e6:.1f}M ({U/(S+U):.0%}); levels/M on solved {sum(r[1] for r in rows)/(S/1e6):.1f}")
