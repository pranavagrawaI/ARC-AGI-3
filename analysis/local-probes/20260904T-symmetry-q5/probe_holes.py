#!/usr/bin/env python3
"""The two holes on the L0 separator: colours='any', and key vs door slots.

Zero tokens. Reads q3c frames plus the recorded L0 terminal pair.
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
PLAY_ROWS = range(0, 54)
N4 = ((0, 1), (0, -1), (1, 0), (-1, 0))


def load_frames():
    frames = []
    for p in sorted(Q3C.glob("steps/step_*.json")):
        o = json.loads(p.read_text())
        frames.append(
            {
                "tag": p.stem,
                "step": int(p.stem.split("_")[1]),
                "level": o.get("level"),
                "level_completed": o.get("level_completed"),
                "grid": o["grid"],
            }
        )
    term = json.loads((Q3C / "ipc/level_0_terminal.json").read_text())

    def hex_grid(rows):
        return [[int(c, 16) for c in r] for r in rows]

    frames.append(
        {"tag": "pre_win", "step": None, "level": 0, "level_completed": False, "grid": hex_grid(term["pre_win_grid_rows"])}
    )
    frames.append(
        {
            "tag": "terminal",
            "step": term["step"],
            "level": 0,
            "level_completed": True,
            "grid": hex_grid(term["terminal_grid_rows"]),
        }
    )
    return frames


def fig_colour(grid, rows, cols, colour):
    return {(r, c): colour for r in rows for c in cols if grid[r][c] == colour}


def fig_region(grid, rows, cols, skip=()):
    return {(r, c): grid[r][c] for r in rows for c in cols if grid[r][c] not in skip}


def components(grid, rows, cols):
    """4-connected same-colour components."""
    seen = set()
    out = []
    h, w = len(grid), len(grid[0])
    rowset, colset = set(rows), set(cols)
    for r in rows:
        for c in cols:
            if (r, c) in seen:
                continue
            col = grid[r][c]
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
                        and grid[ny][nx] == col
                    ):
                        seen.add((ny, nx))
                        stack.append((ny, nx))
            out.append(cells)
    return out


def ascii_fig(fig, title):
    if not fig:
        return f"{title}: empty"
    ys = [y for y, _ in fig]
    xs = [x for _, x in fig]
    y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
    lines = [f"{title} n={len(fig)} bbox=({y0}:{y1+1},{x0}:{x1+1})"]
    for y in range(y0, y1 + 1):
        lines.append(
            "".join(format(fig[(y, x)], "x") if (y, x) in fig else "." for x in range(x0, x1 + 1))
        )
    return "\n".join(lines)


def match(a, b, **kw):
    if not a or not b:
        return False
    return S.canon(a, "none", translate=True, **kw) == S.canon(b, "none", translate=True, **kw)


def similar(a, b, **kw):
    if not a or not b:
        return False
    return S.similar(a, b, **kw)


def door_slots(grid):
    """Whole 2x9 band, each 2x3 slot, and same-colour CCs in the band."""
    whole = fig_region(grid, DOOR_ROWS, DOOR_COLS)
    slots = []
    for i, c0 in enumerate((55, 58, 61)):
        slots.append((f"slot{i}", fig_region(grid, DOOR_ROWS, range(c0, c0 + 3))))
    ccs = [(f"cc{i}_c{next(iter(cc.values()))}_n{len(cc)}", cc) for i, cc in enumerate(components(grid, DOOR_ROWS, DOOR_COLS))]
    return whole, slots, ccs


def run():
    frames = load_frames()
    print(f"frames {len(frames)} (203 steps + pre_win + terminal)")

    # --- show the figures on a few anchors ---
    anchors = {0, 31, 32, 76, 184}
    for fr in frames:
        if fr["tag"] in ("pre_win", "terminal") or fr["step"] in anchors:
            g = fr["grid"]
            key = fig_colour(g, KEY_ROWS, KEY_COLS, 9)
            print()
            print(f"=== {fr['tag']} step={fr['step']} level={fr['level']} ===")
            print(ascii_fig(key, "key colour-9"))
            whole, slots, ccs = door_slots(g)
            print(ascii_fig(whole, "door band 61-62,55-63"))
            for name, sl in slots:
                print(ascii_fig(sl, name))

    # --- A. colours='any' vs playfield CCs (the hole in the first probe) ---
    print("\n========== A. key vs playfield CCs, colours='any' ==========")
    any_hits = []
    sim_hits = []
    sizes_at_term = None
    for fr in frames:
        g = fr["grid"]
        key = fig_colour(g, KEY_ROWS, KEY_COLS, 9)
        ccs = components(g, PLAY_ROWS, range(64))
        if fr["tag"] == "terminal":
            sizes_at_term = sorted(
                {(next(iter(cc.values())), len(cc)) for cc in ccs},
                key=lambda t: -t[1],
            )
        for cc in ccs:
            if match(key, cc, colours="any"):
                any_hits.append((fr["tag"], fr["step"], len(cc), next(iter(cc.values()))))
            if similar(key, cc, colours="any"):
                sim_hits.append((fr["tag"], fr["step"], len(cc), next(iter(cc.values()))))
    print(f"canon(point='none', translate=True, colours='any') hits: {len(any_hits)}")
    for h in any_hits[:20]:
        print("  ", h)
    print(f"similar(colours='any') hits: {len(sim_hits)}")
    for h in sim_hits[:20]:
        print("  ", h)
    print("terminal playfield (colour, size):", sizes_at_term)

    # --- B. key vs door slots ---
    print("\n========== B. key vs door slots ==========")
    kinds = [
        ("fixed/none", dict(colours="fixed"), "none", False),
        ("any/none", dict(colours="any"), "none", False),
        ("fixed/similar", dict(colours="fixed"), "full", True),
        ("any/similar", dict(colours="any"), "full", True),
    ]
    counts = defaultdict(int)
    examples = defaultdict(list)
    for fr in frames:
        g = fr["grid"]
        key = fig_colour(g, KEY_ROWS, KEY_COLS, 9)
        whole, slots, ccs = door_slots(g)
        targets = [("whole", whole)] + slots + ccs
        for tname, tgt in targets:
            for kname, kw, point, scale in kinds:
                if scale:
                    hit = similar(key, tgt, **kw)
                else:
                    hit = S.canon(key, point, translate=True, **kw) == S.canon(
                        tgt, point, translate=True, **kw
                    ) if key and tgt else False
                if hit:
                    counts[(kname, tname.split("_")[0] if tname.startswith("cc") else tname)] += 1
                    if len(examples[(kname, tname)]) < 3:
                        examples[(kname, tname)].append((fr["tag"], fr["step"], len(key), len(tgt)))
    print("hits (kind, target) -> count:")
    if not counts:
        print("  NONE")
    for k, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {n}")
        for e in examples[k]:
            print("    e.g.", e)

    # door vs playfield CCs with colours=any (lock indicator matching a playfield lock)
    print("\n========== B2. door-slot figures vs playfield CCs, colours='any' ==========")
    d_hits = []
    d_sim = []
    for fr in frames:
        g = fr["grid"]
        whole, slots, _ = door_slots(g)
        ccs = components(g, PLAY_ROWS, range(64))
        figs = [("whole", whole)] + slots
        for tname, tgt in figs:
            for cc in ccs:
                if match(tgt, cc, colours="any"):
                    d_hits.append((fr["tag"], tname, len(tgt), len(cc), next(iter(cc.values()))))
                if similar(tgt, cc, colours="any"):
                    d_sim.append((fr["tag"], tname, len(tgt), len(cc), next(iter(cc.values()))))
    print(f"canon none/any hits: {len(d_hits)}")
    for h in d_hits[:15]:
        print("  ", h)
    print(f"similar/any hits: {len(d_sim)}")
    for h in d_sim[:15]:
        print("  ", h)

    print("\nDONE")


if __name__ == "__main__":
    run()
