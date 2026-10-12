"""Readout figures for the 2026-10-12 pruning-transfer pair. Inputs are the JSON files in this directory.

python3 figures.py  -> startup.png, tempo.png, totals.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GRAY, BLUE, ORANGE, AQUA = "#9a9993", "#2a78d6", "#eb6834", "#1baf7a"
ARM = {  # run label -> (arm, colour); colour follows the arm, never the rank
    "base c1": ("base", GRAY), "m2ev ctl": ("base", GRAY),
    "s1 econ": ("prompt variant", GRAY), "m2ev trt": ("prompt variant", GRAY),
    "B1 r1": ("B1", BLUE), "B1 r2": ("B1", BLUE),
    "c14 E2": ("c14", ORANGE), "c14 prune-ctl": ("c14", ORANGE),
    "B1+c14": ("B1+c14", AQUA),
}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.spines.top": False, "axes.spines.right": False})


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)


def startup():
    rows = json.loads((HERE / "startup.json").read_text())
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    style(ax)
    for r in rows:
        x = r["precache_s"]
        if x is None:
            continue
        name = r["log"]
        mine = "pruning-transfer" in name
        if r["ready_s"] is not None:
            ax.plot(x, r["ready_s"], "o", ms=7, color=ORANGE if mine else GRAY, mec=SURFACE, mew=1.5, zorder=3)
        else:  # censored: still loading at the deadline
            ax.plot(x, 722, "^", ms=9, color=ORANGE if mine else INK2, mec=SURFACE, mew=1.5, zorder=3)
        if mine:
            label = name.split("pruning-transfer-20261011/")[1].split("/")[0]
            offset = {"control": (8, -4), "probe2": (8, -14), "candidate": (-12, -18)}[label]
            ax.annotate(label, (x, r["ready_s"] or 722), xytext=offset, textcoords="offset points", color=INK, fontsize=9)
    ax.axhline(722, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(1180, 714, "722 s deadline: benchmark released,\nserver may still be loading", ha="right", va="top", color=INK2, fontsize=8.5)
    ax.set_xlim(180, 1240); ax.set_ylim(440, 790)
    ax.set_xlabel("seconds to read the 185 GB input precache (machine disk speed)")
    ax.set_ylabel("seconds to server READY")
    ax.set_title("Startup is a disk lottery: 6 of 32 dfranzen-family kernels missed the deadline", loc="left", color=INK, fontsize=11)
    ax.text(0.0, -0.2, "Circles: READY time. Triangles: still loading at 722 s (READY not observed). Orange: today's three kernels.\n"
            "Stock dfranzen survives a miss (3 of those 5 kernels went on to play); only our added health assert made the candidate fatal.",
            transform=ax.transAxes, color=INK2, fontsize=8.5, va="top")
    fig.tight_layout(); fig.savefig(HERE / "startup.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def tempo():
    lt = json.loads((HERE / "levels_time.json").read_text())
    ss = json.loads((HERE / "serve_series.json").read_text())
    grid = list(range(0, 122, 2))
    curve = {k: [sum(m <= t for g in v.values() for m in g) for t in grid] for k, v in lt.items()}
    base = [(a + b) / 2 for a, b in zip(curve["base c1"], curve["m2ev ctl"])]
    fig, (top, bot) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    style(top); style(bot)
    for k, c in curve.items():
        arm, col = ARM[k]
        if arm in ("base", "prompt variant"):
            continue
        top.plot(grid, [a - b for a, b in zip(c, base)], color=col, linewidth=2)
    top.axhline(0, color=INK2, linewidth=1)
    ends = {}
    for k, c in curve.items():
        arm, col = ARM[k]
        if arm in ("base", "prompt variant"):
            continue
        ends[k] = c[-1] - base[-1]
    placed = []
    for k, y in sorted(ends.items(), key=lambda kv: kv[1]):
        ly = max([y] + [q + 1.6 for q in placed])  # keep end labels at least 1.6 levels apart
        placed.append(ly)
        top.annotate(f"{k}  {y:+.1f}", (121, y), xytext=(125, ly), textcoords="data", va="center", color=INK, fontsize=8.5)
    top.set_ylabel("levels ahead of base mean (c1, m2ev ctl)\ncumulative, same wall minute")
    top.set_title("c14 trades per-game speed for breadth: behind early, ahead late", loc="left", color=INK, fontsize=11)
    for k, v in ss.items():
        arm, col = ARM[k]
        if arm == "B1+c14":
            continue
        bot.plot([r["min"] for r in v if r["min"] < 115], [r["per_stream"] for r in v if r["min"] < 115],
                 color=col, linewidth=2 if arm != "base" else 1.5, alpha=0.9)
    bot.text(2, 76.5, "10 streams (base, B1)", color=INK2, fontsize=8.5)
    bot.text(2, 53, "14 streams (c14)", color=ORANGE, fontsize=8.5)
    bot.set_ylim(45, 85); bot.set_xlim(0, 140)
    bot.text(116, 47, "after 115 min games\nfinish; bins omitted", color=INK2, fontsize=7.5)
    bot.set_ylabel("decode tok/s per stream")
    bot.set_xlabel("wall minute since the 25 games started (all start together; 121 min horizon)")
    fig.tight_layout(); fig.savefig(HERE / "tempo.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def totals():
    lt = json.loads((HERE / "levels_time.json").read_text())
    total = {k: sum(len(g) for g in v.values()) for k, v in lt.items()}
    order = ["base", "prompt variant", "B1", "c14", "B1+c14"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    style(ax)
    for i, arm in enumerate(order):
        runs = [(k, t) for k, t in total.items() if ARM[k][0] == arm]
        ys = [t for _, t in runs]
        col = ARM[runs[0][0]][1]
        if arm in ("base", "B1", "c14") and len(ys) == 2:  # identical-config replicates
            ax.plot([i, i], ys, color=col, linewidth=2, alpha=0.5)
        for k, t in runs:
            ax.plot(i, t, "o", ms=9, color=col, mec=SURFACE, mew=1.5, zorder=3)
            ax.annotate(f"{t}  {k}", (i, t), xytext=(9, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(order)), order)
    ax.set_xlim(-0.4, len(order) - 0.1); ax.set_ylim(90, 122)
    ax.set_ylabel("levels, public-25, one pass")
    ax.set_title("Run totals: replicate gaps (4, 11, 15) are as large as the arm differences", loc="left", color=INK, fontsize=11)
    fig.tight_layout(); fig.savefig(HERE / "totals.png", dpi=160, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    startup(); tempo(); totals()
