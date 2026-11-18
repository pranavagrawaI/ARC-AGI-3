"""Hypothesis churn on the final unsolved level: share of goal-hypothesis statements whose content words
   overlap (Jaccard >= J) a statement made >= 5 actions earlier. Crude lexical proxy, my threshold.
   python3 churn.py run_dir game [game ...]"""
import re, sys, json, glob
from pathlib import Path
out = Path(sys.argv[1]); b = {g["game_id"][:4]: g for g in json.load(open(out / "benchmark.json"))["game_runs"]}
P = r"[^.\n]{0,120}\b(maybe|perhaps|what if|alternativ\w*)\b[^.\n]{0,60}\b(goal|objective|win)\b[^.\n]{0,160}"
STOP = set("the a an is to of and or in on it be that this what if maybe perhaps goal win objective condition level game".split())
J = 0.4
for gid in sys.argv[2:]:
    g = b[gid]; cut = sum(g["actions_per_level"][:g["levels_completed"]])
    t = open(glob.glob(str(out / f"transcripts/{gid}*.txt"))[0], errors="replace").read(); hs = []
    for tt in re.split(r"^--- analysis_step=", t, flags=re.M)[1:]:
        a = int(re.match(r"\d+ \| action=(\d+)", tt).group(1))
        if a <= cut: continue
        th = "".join(re.findall(r"\[THINKING\]\n(.*?)(?=\n\[[A-Z ]+\]|\Z)", tt, re.S))
        for m in re.finditer(P, th):
            w = {x for x in re.findall(r"[a-z]{3,}", m.group(0).lower()) if x not in STOP}
            if len(w) >= 4: hs.append((a, w))
    rep = sum(any(a - a2 >= 5 and len(w & w2) / len(w | w2) >= J for a2, w2 in hs) for a, w in hs)
    print(f"{gid}: {len(hs)} goal hypotheses on final level, {rep} ({rep/max(1,len(hs)):.0%}) restate one from >=5 actions earlier")
