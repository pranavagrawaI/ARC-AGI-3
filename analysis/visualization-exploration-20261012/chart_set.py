"""Render saved ARC records for visual inspection; never launches evaluations.

Run with python3 -B analysis/visualization-exploration-20261012/chart_set.py
Level labels are one-based. Missing, unfinished, and nonexistent levels differ.
"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Patch
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCES = {
    "Control c1": "analysis/m2-winners-20261001/runs/out-control",
    "Control pc": "solvers/m2_evidence/experiment/pair-1/control/output",
    "Evidence pt": "solvers/m2_evidence/experiment/pair-1/treatment/output",
    "B1 r1": "solvers/m2_b1/experiment/pair-1/b1-r1/output",
    "B1 r2": "solvers/m2_b1/experiment/pair-1/b1-r2/output",
}
TEAL, BLUE, ORANGE, GRAY = "#176f67", "#466daf", "#b85b36", "#aab2b8"
YELLOW = "#e8c575"
PALE, TEXT, GRID = "#edf0f2", "#172b2a", "#e7ecee"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": TEXT, "text.color": TEXT})
cmap = LinearSegmentedColormap.from_list("completion", ["#f1f3f4", "#adcfc9", TEAL])
runs, meta, audits, durations = {}, {}, [], []


def session_generated_tokens(r):
    """Solver finalization records analyzer.generated_tokens in solver_note."""
    m = re.fullmatch(r"tokens=(\d+)", r["solver_note"])
    assert m, (r["game_id"], r["solver_note"])
    return int(m[1])


for label, folder in SOURCES.items():
    p = ROOT / folder / "benchmark.json"
    d = json.loads(p.read_text())
    rr = {r["game_id"]: r for r in d["game_runs"]}
    assert len(rr) == len(d["game_runs"]) == 25
    runs[label], meta[label] = rr, d
    # Independently reconcile every result with the saved text summary.
    lines = [re.match(r"  ([^:]+): score=([\d.]+), levels=([\d.]+)/([\d]+)", l)
             for l in (ROOT / folder / "summary.txt").read_text().splitlines()]
    summary = {m[1]: m for m in lines if m}
    assert set(summary) == set(rr)
    for g, r in rr.items():
        h, apl = r["history"], r["actions_per_level"]
        assert isinstance(h, list) and isinstance(apl, list)
        assert len(h) == sum(apl)
        assert len(apl) == r["number_of_levels"]
        assert 0 <= r["levels_completed"] <= r["number_of_levels"]
        assert r["state"] == "won" or r["levels_completed"] < r["number_of_levels"]
        assert all(x["generated_tokens"] >= 0 for x in h)
        assert session_generated_tokens(r) >= sum(x["generated_tokens"] for x in h)
        assert all(a["wallclock_seconds"] <= b["wallclock_seconds"] for a, b in zip(h, h[1:]))
        assert h[-1]["wallclock_seconds"] <= r["final_wallclock_seconds"] + .01
        m = summary[g]
        assert float(m[3]) == r["levels_completed"]
        assert int(m[4]) == r["number_of_levels"]
        assert abs(float(m[2]) - r["final_score"]) <= .00501
    audits.append({"run": label, "source": str(p.relative_to(ROOT)),
                   "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "verified_games": 25})
assert all(set(r) == set(runs["B1 r1"]) for r in runs.values())
games = sorted(runs["B1 r1"], key=lambda g: (
    sum(runs[label][g]["levels_completed"] / runs[label][g]["number_of_levels"] for label in ["B1 r1", "B1 r2"]), g))
assert all(len({r[g]["number_of_levels"] for r in runs.values()}) == 1 for g in games)


def heading(fig, title, subtitle):
    fig.text(.07, .968, title, fontsize=19, fontweight="bold", va="top")
    fig.text(.07, .928, subtitle, fontsize=10.5, color="#58666d", va="top")


def save(fig, stem, footers):
    for i, line in enumerate(footers):
        fig.text(.07, .063 - i * .019, line, fontsize=9, color="#58666d", va="top")
    fig.savefig(OUT / (stem + ".png"), dpi=165, facecolor="white")
    fig.savefig(OUT / (stem + ".svg"), facecolor="white")
    plt.close(fig)


# 1. Outcome matrix, retaining actual repetitions rather than averaging columns.
labels = list(SOURCES)
counts = np.array([[runs[label][g]["levels_completed"] for label in labels] for g in games])
total = np.array([runs["B1 r1"][g]["number_of_levels"] for g in games])
fractions = counts / total[:, None]
fig = plt.figure(figsize=(10.5, 10.5))
heading(fig, "Outcomes across five saved runs", "25 games · raw completion counts · rows ordered by B1 mean completion fraction")
ax = fig.add_axes([.14, .17, .79, .65])
ax.imshow(fractions, vmin=0, vmax=1, cmap=cmap, aspect="auto")
dates = [meta[l]["start_time"][:10][5:] for l in labels]
ax.set_xticks(range(5), [f"{l}\n{date}\n{counts[:, j].sum()} levels" for j, (l, date) in enumerate(zip(labels, dates))])
ax.xaxis.tick_top()
ax.set_yticks(range(25), [g[:4] for g in games])
ax.tick_params(length=0, pad=10)
for i in range(25):
    for j in range(5):
        ax.text(j, i, f"{counts[i,j]}/{total[i]}", ha="center", va="center",
                color="white" if fractions[i,j] > .7 else TEXT)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xlabel("Completed / available levels · common 0–100% color scale", labelpad=15)
save(fig, "01-outcome-matrix", [
    "c1 and pc are controls from different dates; pc and pt form the fresh evidence comparison.",
    "B1 r1/r2 repeat the same build. B1 uses historical controls; this grid alone does not establish effects."])


# 2. Actual level states. Each game has two rows, with explicit labels.
fig = plt.figure(figsize=(11, 12.5))
heading(fig, "Which levels were solved, unfinished, or never reached?", "B1 repetitions · each game has an upper r1 row and lower r2 row · levels shown one-based")
ax = fig.add_axes([.16, .16, .78, .71])
symbols = {"solved": "✓", "unfinished": "×", "entered_no_action": "○", "not_reached": "·", "absent": ""}
colors = {"solved": TEAL, "unfinished": ORANGE, "entered_no_action": YELLOW, "not_reached": PALE, "absent": "white"}
level_records = []
for i, g in enumerate(games):
    for j, label in enumerate(["B1 r1", "B1 r2"]):
        r = runs[label][g]
        for level in range(10):
            state = "absent" if level >= r["number_of_levels"] else "solved" if level < r["levels_completed"] else "unfinished" if r["actions_per_level"][level] > 0 else "entered_no_action" if level == r["levels_completed"] else "not_reached"
            if state == "entered_no_action":
                p = ROOT / SOURCES[label] / "artifacts" / f"{g}_p0_events.jsonl"
                last_action = None
                for line in p.open():
                    e = json.loads(line)
                    if e["type"] == "action":
                        last_action = e
                assert last_action["level"] == level + 1 and last_action["state"] == "NOT_FINISHED"
            y = i * 2 + j
            ax.add_patch(Rectangle((level, y), .94, .94, facecolor=colors[state], edgecolor="none"))
            ax.text(level + .47, y + .47, symbols[state], ha="center", va="center", fontsize=11,
                    color="white" if state in ["solved", "unfinished"] else "#78858d")
            if state != "absent":
                level_records.append({"game": g, "run": label, "level_one_based": level + 1, "state": state})
ax.set_xlim(0, 10)
ax.set_ylim(50, 0)
ax.set_xticks(np.arange(10) + .47, [f"L{i+1}" for i in range(10)])
ax.xaxis.tick_top()
ax.set_yticks(np.arange(50) + .47, [f"{g[:4]}  r{j+1}" for g in games for j in range(2)], fontsize=9)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
fig.legend(handles=[Patch(color=TEAL, label="✓ solved"), Patch(color=ORANGE, label="× unfinished, actions recorded"),
                    Patch(color=YELLOW, label="○ entered, no actions"), Patch(color=PALE, label="· not reached"),
                    Patch(facecolor="white", edgecolor=GRAY, label="blank: no such level")],
           loc="lower center", bbox_to_anchor=(.55, .085), ncol=3, frameon=False, fontsize=9.5)
save(fig, "02-level-status", [
    "Yellow levels were entered before termination, confirmed by final action events. Later levels were not reached.",
    "Terminal wins come from benchmark state/completion counts; final levels need no separate solved event."])


# 3. Stacked time intervals, using recorded action allocation and solve boundaries.
fig = plt.figure(figsize=(13.5, 10.5))
heading(fig, "Where did each game spend its time?", "Game-local elapsed minutes · green segments complete a level · orange segments end unfinished")
axs = [fig.add_axes([.12, .17, .36, .66]), fig.add_axes([.59, .17, .36, .66])]
xmax = max(r["final_wallclock_seconds"] / 60 for label in ["B1 r1", "B1 r2"] for r in runs[label].values())
unassigned_minutes = []
for ax, label in zip(axs, ["B1 r1", "B1 r2"]):
    ax.set_title(label, fontweight="bold", pad=16)
    for i, g in enumerate(games):
        r = runs[label][g]
        h, apl = r["history"], r["actions_per_level"]
        k, start = 0, 0.
        for level, n in enumerate(apl):
            if not n and level != r["levels_completed"]:
                continue
            if not n:
                assert r["state"] != "won"
                end = r["final_wallclock_seconds"] / 60
                ax.barh(i, end-start, left=start, height=.74, color=YELLOW)
                durations.append({"run": label, "game": g, "level_one_based": level+1,
                                  "start_min": start, "end_min": end, "solved": False,
                                  "generated_tokens": None, "actions": 0,
                                  "token_note": "No action recorded on this level; pending response tokens are not measured here."})
                start = end
                continue
            k += n
            end = h[k-1]["wallclock_seconds"] / 60
            solved = level < r["levels_completed"]
            if not solved:
                end = r["final_wallclock_seconds"] / 60
            assert end >= start
            ax.barh(i, end-start, left=start, height=.74, color=TEAL if solved else ORANGE)
            if start > 0:
                ax.vlines(start, i-.37, i+.37, color="white", linewidth=.7)
            if end-start >= 8:
                ax.text((start+end)/2, i, f"L{level+1}", ha="center", va="center", color="white", fontsize=8.5)
            durations.append({"run": label, "game": g, "level_one_based": level+1,
                              "start_min": start, "end_min": end, "solved": solved,
                              "generated_tokens": sum(x["generated_tokens"] for x in h[k-n:k])})
            start = end
        if r["final_wallclock_seconds"]/60 - start > .001:
            gap = r["final_wallclock_seconds"]/60 - start
            ax.barh(i, gap, left=start, height=.74, color=GRAY)
            unassigned_minutes.append({"run": label, "game": g, "minutes": gap})
    ax.set_xlim(0, xmax*1.02)
    ax.set_ylim(24.7, -.7)
    ax.set_yticks(range(25), [g[:4] for g in games])
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.set_xlabel("Elapsed minutes since game start")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
save(fig, "03-time-allocation", [
    "Segment boundaries use cumulative actions_per_level and action timestamps; labels appear on segments ≥8 minutes.",
    "Orange includes time through termination; yellow = entered with no action; gray = finalization after a win."])


# 4. Outcome versus recorded token expenditure, retaining case identity and run pairing.
fig = plt.figure(figsize=(11.5, 7.5))
heading(fig, "Outcome versus session generated tokens", "Same B1 build · each line connects the same game in r1 and r2 · 25 paired games")
ax = fig.add_axes([.10, .24, .82, .60])
costs = {}
for g in games:
    pair = []
    for label in ["B1 r1", "B1 r2"]:
        r = runs[label][g]
        tokens = session_generated_tokens(r)
        costs[label, g] = tokens
        pair.append((tokens/1000, r["final_score"]))
    ax.plot([p[0] for p in pair], [p[1] for p in pair], color="#c5ced3", lw=1, zorder=1)
    ax.scatter(*pair[0], color=BLUE, marker="o", s=31, zorder=3)
    ax.scatter(*pair[1], color=TEAL, marker="^", s=37, zorder=3)
ax.set_xlim(0, max(costs.values())/1000 * 1.13)
ax.set_ylim(-4, 106)
ax.set_xlabel("Session generated tokens per game (thousands)")
ax.set_ylabel("Final competition score (0–100)")
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(color=GRID, zorder=0)
legend = ax.legend(handles=[plt.Line2D([], [], linestyle="", marker="o", color=BLUE, label="B1 r1"),
                   plt.Line2D([], [], linestyle="", marker="^", color=TEAL, label="B1 r2")], frameon=False, loc="upper left")
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
font = FontProperties(size=9)
occupied = [legend.get_window_extent(renderer)]
point_boxes = []
for g in games:
    for label in ["B1 r1", "B1 r2"]:
        xy = ax.transData.transform((costs[label, g]/1000, runs[label][g]["final_score"]))
        point_boxes.append(Bbox.from_bounds(xy[0]-5, xy[1]-5, 10, 10))
plot_box = ax.get_window_extent(renderer)

def area_overlap(a, b):
    return max(0, min(a.x1, b.x1)-max(a.x0, b.x0)) * max(0, min(a.y1, b.y1)-max(a.y0, b.y0))

for g in sorted(games, key=lambda g: (-runs["B1 r2"][g]["final_score"], costs["B1 r2", g])):
    xy_data = (costs["B1 r2", g]/1000, runs["B1 r2"][g]["final_score"])
    xy = ax.transData.transform(xy_data)
    width, height, descent = renderer.get_text_width_height_descent(g[:4], font, False)
    candidates = []
    for dx in [0, 15, -15, 30, -30, 50, -50, 75, -75]:
        for dy in [-14, 14, -28, 28, -44, 44, -65, 65]:
            box = Bbox.from_bounds(xy[0]+dx-width/2-3, xy[1]+dy-height/2-3, width+6, height+6)
            if not (plot_box.x0+4 <= box.x0 and box.x1 <= plot_box.x1-4 and plot_box.y0+4 <= box.y0 and box.y1 <= plot_box.y1-4):
                continue
            penalty = sum(area_overlap(box, b) for b in occupied)*1000 + sum(area_overlap(box, b) for b in point_boxes)*50 + dx*dx + dy*dy
            candidates.append((penalty, dx, dy, box))
    _, dx, dy, box = min(candidates, key=lambda x: x[0])
    occupied.append(box)
    dest = ax.transData.inverted().transform(xy+[dx, dy])
    ax.annotate(g[:4], xy=xy_data, xytext=dest, ha="center", va="center", fontsize=9,
                arrowprops={"arrowstyle": "-", "color": "#8b969d", "lw": .65}, zorder=4)
save(fig, "04-quality-cost", [
    "Costs use benchmark solver_note session totals. Action-history sums undercount these B1 runs by 2.7–4.3%.",
    "Labels attach to r2 triangles. Game difficulty differs; token expenditure is not GPU time or money."])


# 5. Oracle-hint outcome matrix, including unmeasured arms and admitted denominators.
oracle_path = ROOT / "analysis/oracle-hint-20261010/readout.json"
oracle = json.loads(oracle_path.read_text())
assert not oracle["result"]["integrity_problems"]
rows = oracle["rows"]
table = sorted(oracle["table"], key=lambda r: (r["klass"] == "never", r["key"]))
arms = ["A0", "A1", "A2"]
oracle_cells = []
for t in table:
    for arm in arms:
        attempts = [r for r in rows.values() if r["key"] == t["key"] and r["arm"] == arm and r["admitted"]]
        wins = sum(r["solved"] for r in attempts)
        n = len(attempts)
        v = None if not n else wins/n
        j = arms.index(arm)
        expected = t[f"p{j}"]
        assert n == t[f"n{j}"]
        assert (v is None and expected is None) or abs(v-expected) < 1e-10
        assert all(r["solved"] == (r["solved_event"] or r["won"]) for r in attempts)
        oracle_cells.append({"key": t["key"], "arm": arm, "solves": wins, "attempts": n})
fig = plt.figure(figsize=(11, 12))
heading(fig, "What happened when goals or routes were supplied?", "Oracle-hint attempts · 1,200-second cap · all 40 selected levels · counts include final-level wins")
ax = fig.add_axes([.17, .16, .73, .68])
for i, t in enumerate(table):
    for j, arm in enumerate(arms):
        c = oracle_cells[i*3+j]
        n = c["attempts"]
        face = cmap(c["solves"]/n) if n else "white"
        ax.add_patch(Rectangle((j, i), .96, .96, facecolor=face, edgecolor=GRAY if not n else "none", hatch="///" if not n else None, lw=.4))
        ax.text(j+.48, i+.48, f"{c['solves']}/{n}" if n else "not tested", ha="center", va="center",
                color="white" if n and c["solves"]/n > .7 else TEXT, fontsize=9.5)
ax.set_xlim(0, 3)
ax.set_ylim(40, 0)
ax.set_xticks(np.arange(3)+.48, ["No hint\nA0", "Goal hint\nA1", "Goal + route\nA2"])
ax.xaxis.tick_top()
ax.set_yticks(np.arange(40)+.48, [f"{t['key'].split('-L')[0]} L{int(t['key'].split('-L')[1])+1}" for t in table], fontsize=9)
ax.tick_params(length=0, pad=7)
for s in ax.spines.values():
    s.set_visible(False)
ax.axhline(26, color=GRAY, lw=1)
save(fig, "05-oracle-hint-outcomes", [
    "Top 26 levels have validated hints/routes and a prior sibling solve; bottom 14 had no solve in the three source runs.",
    "A1/A2 were not tested on those 14. Selection and the time cap limit generalization; labels convert stored L0 to L1."])


stats = {}
for label, rr in runs.items():
    stats[label] = {
        "date": meta[label]["start_time"][:10],
        "total_levels": sum(r["levels_completed"] for r in rr.values()),
        "mean_score": sum(r["final_score"] for r in rr.values())/25,
        "generated_tokens": sum(sum(x["generated_tokens"] for x in r["history"]) for r in rr.values()),
        "session_generated_tokens": sum(session_generated_tokens(r) for r in rr.values()),
        "job_minutes": (datetime.fromisoformat(meta[label]["end_time"])-datetime.fromisoformat(meta[label]["start_time"])).total_seconds()/60,
    }
audit = {"sources": audits, "run_statistics": stats, "level_states": level_records,
         "level_time_segments": durations, "post_win_finalization": unassigned_minutes,
         "token_scope": "Quality-cost plot uses solver_note session counters. Per-level token counts and legacy generated_tokens statistics use action history, which can omit other responses.",
         "oracle_source_sha256": hashlib.sha256(oracle_path.read_bytes()).hexdigest(),
         "oracle_cells": oracle_cells,
         "checks": ["All 125 game results match saved summaries", "All histories match action allocations",
                    "All timestamps are monotonic", "All game/level populations match",
                    "All 120 oracle cells match admitted attempts and recorded readout",
                    "Final-level wins count as successes without requiring solved events"]}
(OUT / "chart-set-data.json").write_text(json.dumps(audit, indent=2)+"\n")
print(json.dumps({"run_statistics": stats, "checks": audit["checks"], "figures": 5}, indent=2))
