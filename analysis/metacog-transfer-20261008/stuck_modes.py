"""Classify thinking on final unsolved level vs solved levels by marker rates (per 100K chars).
   python3 stuck_modes.py run_dir"""
import re, sys, json, glob
from pathlib import Path
out = Path(sys.argv[1]); b = json.load(open(out / "benchmark.json"))
M = {"sim/model": r"\b(simulat\w*|sim\d*\(|world model|predict\w*)", "search": r"\b(BFS|bfs|best-?first|A\*|search|expanded)\b",
     "mismatch": r"\bmismatch\w*|\bWRONG\b|doesn't match|does not match", "goal-doubt": r"(maybe|perhaps|what if|alternativ\w*|instead)[^.]{0,60}\b(goal|objective|win)",
     "sandbox-err": r"NameError|not allowed in the sandbox|ImportError|StaleState|timed? ?out|time limit", "recount": r"recount|re-?index|let me (re)?count"}
print("game lv phase kchars " + " ".join(M))
for g in b["game_runs"]:
    n = g["levels_completed"]; apl = g["actions_per_level"]; cut = sum(apl[:n])
    if n >= len(apl) or g["state"] == "won": continue
    t = open(glob.glob(str(out / f"transcripts/{g['game_id'][:4]}*.txt"))[0], errors="replace").read()
    seg = {"solved": "", "final": ""}
    for tt in re.split(r"^--- analysis_step=", t, flags=re.M)[1:]:
        a = int(re.match(r"\d+ \| action=(\d+)", tt).group(1))
        th = "".join(re.findall(r"\[THINKING\]\n(.*?)(?=\n\[[A-Z ]+\]|\Z)", tt, re.S))
        seg["final" if a > cut else "solved"] += th
    for ph, s in seg.items():
        if not s: continue
        k = len(s) / 1e5
        print(g["game_id"][:4], n, ph, int(len(s) / 1000), " ".join(f"{len(re.findall(p, s)) / k:6.0f}" for p in M.values()))
