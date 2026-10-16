"""Levels at matched cumulative completion tokens: Tinfield r3 vs the two c14 controls (no GPU).

Uses the r3 readout's own clock definitions (transcript turn boundaries, game-minute clock per game).
For each minute t, sums completion tokens and levels over all 25 games up to game-minute t.
"""
import importlib.util, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("r", REPO / "analysis/model-swap-kernels-20261013/tinfield-r3/tools/readout.py")
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
ARMS = {"tinfield": REPO / "analysis/model-swap-kernels-20261013/tinfield-r3/output", **r.CONTROL}
MIN = list(range(2, 41, 2))
out = {}
for name, path in ARMS.items():
    curve = r.levels.run(path)
    pts = []
    for t in MIN:
        tok = r.thinking_per_action(path, minutes=t)
        pts.append({"min": t, "tokens": tok["completion_tokens"], "actions": tok["executed_actions"],
                    "levels": sum(sum(v <= t for v in ls) for ls in curve.values())})
    out[name] = pts
    print(name, " ".join(f"{p['min']}:{p['tokens']/1e6:.2f}M/{p['levels']}" for p in pts))

def at_tokens(pts, x):  # linear interpolation of levels vs cumulative tokens
    for a, b in zip(pts, pts[1:]):
        if a["tokens"] <= x <= b["tokens"]:
            f = (x - a["tokens"]) / max(1, b["tokens"] - a["tokens"]); return a["levels"] + f * (b["levels"] - a["levels"])
    return None
summary = {}
for x in [0.25e6, 0.5e6, 0.75e6, 1.0e6, out["tinfield"][-1]["tokens"]]:
    row = {k: at_tokens(v, x) for k, v in out.items()}
    summary[f"{x/1e6:.3f}M"] = row
    print(f"levels at {x/1e6:.3f}M tokens:", {k: (round(v, 1) if v is not None else None) for k, v in row.items()})
json.dump({"curves": out, "matched": summary}, open(Path(__file__).with_name("matched.json"), "w"), indent=1)
