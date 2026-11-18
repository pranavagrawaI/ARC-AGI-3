#!/usr/bin/env python3
"""Eight key-change steps: successive keys vs the board, doors, and co-changing cells.

Oracle names 19, 69, 76, 130, 132, 137, 178, 184. Step 19 is L0; the rest are L1.
This script re-derives the steps from the key box, then tests each pair (t-1, t).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

SEED = Path(__file__).resolve().parent / "seed"
sys.path.insert(0, str(SEED))
import symmetry as S  # noqa: E402

Q3C = Path(
    "/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/"
    "tall-humerus/analysis/local-probes/20260904T-host-control-loop-q3/"
    "results/nemo_solver/20260904_140403_ls20_mdfiles_q3c"
)

KEY_ROWS, KEY_COLS = range(54, 61), range(0, 12)
DOOR_ROWS, DOOR_COLS = range(61, 63), range(55, 64)
CLOCK_ROWS, CLOCK_COLS = range(61, 63), range(13, 55)
PLAY_ROWS = range(0, 54)
N4 = ((0, 1), (0, -1), (1, 0), (-1, 0))
KINDS = (
    ("none/fixed", "none", False, "fixed"),
    ("none/any", "none", False, "any"),
    ("cong/fixed", "full", False, "fixed"),
    ("cong/any", "full", False, "any"),
    ("sim/fixed", "full", True, "fixed"),
    ("sim/any", "full", True, "any"),
)


def load_step(n):
    o = json.loads((Q3C / f"steps/step_{n:04d}.json").read_text())
    return o["grid"], o.get("level"), o.get("level_completed")


def key9(g):
    return {(r, c): 9 for r in KEY_ROWS for c in KEY_COLS if g[r][c] == 9}


def region(g, rows, cols):
    return {(r, c): g[r][c] for r in rows for c in cols}


def slots(g):
    return [
        (f"slot{i}", {(r, c): g[r][c] for r in DOOR_ROWS for c in range(c0, c0 + 3)})
        for i, c0 in enumerate((55, 58, 61))
    ]


def components(g, rows, cols):
    seen, out = set(), []
    rowset, colset = set(rows), set(cols)
    h, w = len(g), len(g[0])
    for r in rows:
        for c in cols:
            if (r, c) in seen:
                continue
            col = g[r][c]
            stack = [(r, c)]
            seen.add((r, c))
            cells = {}
            while stack:
                y, x = stack.pop()
                cells[(y, x)] = col
                for dy, dx in N4:
                    ny, nx = y + dy, x + dx
                    if (
                        (ny, nx) not in seen
                        and ny in rowset
                        and nx in colset
                        and 0 <= ny < h
                        and 0 <= nx < w
                        and g[ny][nx] == col
                    ):
                        seen.add((ny, nx))
                        stack.append((ny, nx))
            out.append(cells)
    return out


def show(fig, title):
    if not fig:
        return f"{title}: empty"
    ys, xs = [y for y, _ in fig], [x for _, x in fig]
    lines = [f"{title} n={len(fig)} bbox=({min(ys)}:{max(ys)+1},{min(xs)}:{max(xs)+1})"]
    for y in range(min(ys), max(ys) + 1):
        lines.append(
            "".join(format(fig[(y, x)], "x") if (y, x) in fig else "." for x in range(min(xs), max(xs) + 1))
        )
    return "\n".join(lines)


def eq(a, b, point, scale, colours):
    if not a or not b:
        return False
    if scale:
        return S.similar(a, b, colours=colours)
    return S.canon(a, point, translate=True, colours=colours) == S.canon(
        b, point, translate=True, colours=colours
    )


def find_key_change_steps(n=203):
    prev, steps = None, []
    for t in range(n):
        g, lv, _ = load_step(t)
        box = tuple(tuple(g[r][c] for c in KEY_COLS) for r in KEY_ROWS)
        if prev is not None and box != prev:
            steps.append(t)
        prev = box
    return steps


def band(r, c):
    if r in KEY_ROWS and c in KEY_COLS:
        return "key"
    if r in DOOR_ROWS and c in DOOR_COLS:
        return "door"
    if r in CLOCK_ROWS and c in CLOCK_COLS:
        return "clock"
    if r in PLAY_ROWS:
        return "play"
    return "other"


def run():
    derived = find_key_change_steps()
    print("derived key-box change steps:", derived)
    print("levels:", [(t, load_step(t)[1]) for t in derived])

    hits = defaultdict(list)

    for t in derived:
        g0, lv0, _ = load_step(t - 1)
        g1, lv1, done = load_step(t)
        k0, k1 = key9(g0), key9(g1)
        core0, pk0 = S.primitive(k0)
        core1, pk1 = S.primitive(k1)

        diffs = [
            (r, c, g0[r][c], g1[r][c])
            for r in range(64)
            for c in range(64)
            if g0[r][c] != g1[r][c]
        ]
        by = defaultdict(list)
        for r, c, a, b in diffs:
            by[band(r, c)].append((r, c, a, b))

        print(f"\n======== t={t}  level {lv0}->{lv1}  completed={done} ========")
        print(show(k0, "key before"))
        print(show(k1, "key after"))
        print(f"primitive before k={pk0} n={len(core0)}; after k={pk1} n={len(core1)}")
        print(show(core0, "core before"))
        print(show(core1, "core after"))
        for name, sl in slots(g1):
            print(show(sl, f"door {name} after"))
        print(
            "diff counts:",
            {k: len(v) for k, v in sorted(by.items())},
        )
        if by["play"]:
            print("  playfield diffs (first 20):", by["play"][:20])
        if by["door"]:
            print("  door diffs:", by["door"])
        if by["other"]:
            print("  other diffs:", by["other"][:10])

        # successive keys
        for kname, point, scale, colours in KINDS:
            if eq(k0, k1, point, scale, colours):
                hits[("key_t-1 vs key_t", kname)].append(t)
            if eq(core0, core1, point, scale, colours):
                hits[("core_t-1 vs core_t", kname)].append(t)

        ccs_t = components(g1, PLAY_ROWS, range(64))
        ccs_prev = components(g0, PLAY_ROWS, range(64))
        door_figs = [("band", region(g1, DOOR_ROWS, DOOR_COLS))] + slots(g1)

        for kname, point, scale, colours in KINDS:
            for cc in ccs_t:
                if eq(k1, cc, point, scale, colours):
                    hits[("key_t vs play_t", kname)].append(
                        (t, len(cc), next(iter(cc.values())))
                    )
                if eq(core1, cc, point, scale, colours):
                    hits[("core_t vs play_t", kname)].append(
                        (t, len(cc), next(iter(cc.values())))
                    )
            for cc in ccs_prev:
                if eq(k1, cc, point, scale, colours):
                    hits[("key_t vs play_{t-1}", kname)].append(
                        (t, len(cc), next(iter(cc.values())))
                    )
            for dname, dfig in door_figs:
                if eq(k1, dfig, point, scale, colours):
                    hits[("key_t vs door", kname, dname)].append(t)
                if eq(core1, dfig, point, scale, colours):
                    hits[("core_t vs door", kname, dname)].append(t)

        # playfield cells that actually changed at t, as a figure
        play_after = {(r, c): g1[r][c] for r, c, _, _ in by["play"]}
        play_before = {(r, c): g0[r][c] for r, c, _, _ in by["play"]}
        if play_after:
            print(show(play_after, "playfield cells after (changed only)"))
            print(show(play_before, "playfield cells before (changed only)"))
            for kname, point, scale, colours in KINDS:
                if eq(k1, play_after, point, scale, colours):
                    hits[("key_t vs play-diff-after", kname)].append(t)
                if eq(k1, play_before, point, scale, colours):
                    hits[("key_t vs play-diff-before", kname)].append(t)
                if eq(core1, play_after, point, scale, colours):
                    hits[("core_t vs play-diff-after", kname)].append(t)

    print("\n========== HIT TABLE ==========")
    if not hits:
        print("NONE")
    else:
        for k, v in sorted(hits.items()):
            print(f"{k}: {v}")
    print("\nDONE")


if __name__ == "__main__":
    run()
