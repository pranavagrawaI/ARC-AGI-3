"""Render two complementary views of saved B1 replicates; no new runs."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PATHS = [ROOT / f"solvers/m2_b1/experiment/pair-1/b1-r{i}/output/benchmark.json" for i in (1, 2)]
runs = [{r["game_id"]: r for r in json.loads(p.read_text())["game_runs"]} for p in PATHS]
assert len(runs[0]) == len(runs[1]) == 25
assert runs[0].keys() == runs[1].keys()
for g in runs[0]:
    a, b = (r[g] for r in runs)
    assert a["number_of_levels"] == b["number_of_levels"]
    assert all(0 <= r["levels_completed"] <= r["number_of_levels"] for r in (a, b))

# Order by average completion fraction; show raw counts to preserve denominators.
games = sorted(runs[0], key=lambda g: (sum(r[g]["levels_completed"] / r[g]["number_of_levels"] for r in runs), g))
counts = np.array([[r[g]["levels_completed"] for r in runs] for g in games])
totals = np.array([runs[0][g]["number_of_levels"] for g in games])
fractions = counts / totals[:, None]
scores = np.array([[r[g]["final_score"] for r in runs] for g in games])
delta = scores[:, 1] - scores[:, 0]
summary = {
    "sources": [str(p.relative_to(ROOT)) for p in PATHS],
    "games": len(games),
    "total_levels": counts.sum(axis=0).tolist(),
    "mean_score": scores.mean(axis=0).tolist(),
    "same_completed_level_count": int((counts[:, 0] == counts[:, 1]).sum()),
    "more_levels_r2": int((counts[:, 1] > counts[:, 0]).sum()),
    "fewer_levels_r2": int((counts[:, 1] < counts[:, 0]).sum()),
    "both_at_most_two_levels": [g for i, g in enumerate(games) if max(counts[i]) <= 2],
    "limitation": "Two repetitions of the same build. Descriptive repeat variation, not a treatment comparison or confidence interval.",
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(10.7, 10), facecolor="white")
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.65], left=.11, right=.94, top=.77, bottom=.19, wspace=.58)
ax = fig.add_subplot(gs[0, 0])
cmap = LinearSegmentedColormap.from_list("completion", ["#f1f3f4", "#adcfc9", "#176f67"])
ax.imshow(fractions, vmin=0, vmax=1, cmap=cmap, aspect="auto")
ax.set_xticks([0, 1], ["B1 r1", "B1 r2"])
ax.xaxis.tick_top()
ax.set_yticks(range(len(games)), [g[:4] for g in games])
ax.tick_params(length=0, pad=8)
ax.set_title("Where do both runs struggle?", loc="left", pad=38, fontweight="bold", fontsize=12)
for i in range(len(games)):
    for j in range(2):
        ax.text(j, i, f"{counts[i,j]}/{totals[i]}", ha="center", va="center", fontsize=10, color="white" if fractions[i,j] > .7 else "#172b2a")
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_xlabel("Completed / available levels\nDarker = larger fraction completed", labelpad=14)

bx = fig.add_subplot(gs[0, 1])
y = np.arange(len(games))
bx.axvline(0, color="#8a9195", linewidth=1)
for i, d in enumerate(delta):
    c = "#176f67" if d > 0 else "#b85b36" if d < 0 else "#8a9195"
    bx.plot([0, d], [i, i], color=c, linewidth=2)
    bx.scatter([d], [i], color=c, s=25, zorder=3)
bx.set_ylim(len(games) - .5, -.5)
bx.set_xlim(-105, 105)
bx.set_yticks(y, [g[:4] for g in games])
bx.tick_params(axis="y", length=0)
bx.set_xticks([-100, -50, 0, 50, 100])
bx.set_xlabel("Final score difference: r2 − r1 (points)\n← r1 higher                          r2 higher →", labelpad=12)
bx.set_title("How much varies on repetition?", loc="left", pad=38, fontweight="bold", fontsize=12)
bx.spines["left"].set_visible(False)
bx.grid(axis="x", color="#eceeef", zorder=0)

fig.text(.065, .95, "Same build. Different outcomes.", fontsize=21, fontweight="bold", color="#172b2a")
fig.text(.065, .915, "Saved B1 public-25 runs · 10 October 2026 · all 25 games present in each run", fontsize=11, color="#58666d")
fig.text(.065, .878, f"r1: {counts[:,0].sum()} levels / {scores[:,0].mean():.2f} mean score     r2: {counts[:,1].sum()} levels / {scores[:,1].mean():.2f} mean score", fontsize=12)
fig.text(.065, .062, "Rows sorted by mean completion fraction. Score also reflects action efficiency; it differs from level count.", fontsize=9, color="#58666d")
fig.text(.065, .039, "These are repetitions, not treatment vs control. Two runs do not establish a noise distribution or a causal effect.", fontsize=9, color="#58666d")
fig.savefig(OUT / "b1-comparison.png", dpi=160)
fig.savefig(OUT / "b1-comparison.svg")
print(json.dumps(summary, indent=2))
