"""Freeze questioner inputs at recorded stall points of the 3.8 commit run.

For each chosen game, every `stride`-th analyzer turn from `start` on becomes one packet:
the actor's own words for the preceding turns (said + thinking, as tool_agent now records
them) and the recent action outcomes. Packets are graded with an empty ledger (independent
questions) and, for one game, as a sequence (ledger carried) to exercise repeat-avoidance.
usage: python3 build_packets.py <artifacts dir> > packets.json
"""
import json, glob, os, re, sys

GAMES = {  # game: (why it is here, start turn, stride)
    "sp80": ("B worldview lock: alignment permutations; 4-row gap never asked", 6, 3),
    "tn36": ("B worldview lock: 'drop piece in cup'; T bars never clicked", 6, 4),
    "sc25": ("B worldview lock: legend=grid, 3 failed submissions; cab unnoticed", 6, 4),
    "dc22": ("B worldview lock / flailing, game over at 114", 8, 5),
    "lf52": ("E lossy level-up: 'L2 impossible' from a wrong L1 rule", 14, 5),
    "cn04": ("C perception thrash: 6 reversals on the red squares", 8, 5),
}

def section(t, name):
    i = t.find(f"[{name}]")
    if i < 0: return ""
    j = t.find("\n[", i + len(name) + 2)
    return t[i + len(name) + 2 : j if j > 0 else None].strip()

out = []
for f in sorted(glob.glob(sys.argv[1] + "/*events.jsonl")):
    g = os.path.basename(f)[:4]
    if g not in GAMES: continue
    why, start, stride = GAMES[g]
    words, trans, turn = [], [], 0
    level = 1
    for line in open(f):
        e = json.loads(line)
        if e["type"] == "action":
            trans.append({"action": e["action_display"], "board_changed": str(e.get("board_changed")) == "True",
                          "level_completed": str(e.get("level_completed")) == "True", "game_over": str(e.get("game_over")) == "True"})
            level = int(e.get("level") or level)
            continue
        if e["type"] != "analysis": continue
        turn += 1
        # packet frozen BEFORE this turn runs: it sees the words of earlier turns only
        if turn >= start and (turn - start) % stride == 0 and words:
            out.append({"game": g, "why": why, "turn": turn, "level": level, "action_num": len(trans),
                        "recent_words": words[-5:], "transitions": trans[-12:]})
        t = e["transcript"]
        words.append({"seq": turn, "action_num": int(e["action_num"]) + 1,
                      "said": section(t, "ASSISTANT")[-2000:], "thinking": section(t, "THINKING")[-3000:]})
print(json.dumps(out))
print(f"{len(out)} packets: " + ", ".join(f"{g}={sum(p['game']==g for p in out)}" for g in GAMES), file=sys.stderr)
