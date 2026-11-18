#!/usr/bin/env python3
"""ar25 offline separator. Zero tokens.

Frames: Tycho uptake checkpoint level_0/turn_*.txt
(the live run dir was untracked; this checkpoint has the 17 L0 grids).

The uptake actor compared colour-4 and colour-11 L pieces, labels flattened,
translate=True, size 45==45, then transported. This script recomputes that
pair with the seeded module, and asks the §3 question: which relations are
true on the terminal frame and false on the others.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SEED = Path(__file__).resolve().parent / "seed"
sys.path.insert(0, str(SEED))
import symmetry as S  # noqa: E402

CKPT = Path(
    "/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/"
    "tall-humerus/analysis/matched-actor-v5-pilot/checkpoints/symmetry-uptake-ar25"
)
HEX = {c: i for i, c in enumerate("0123456789abcdef")}
N4 = ((0, 1), (0, -1), (1, 0), (-1, 0))
# Background, barrier, move-counter column, bottom HUD row.
SKIP_COLOURS = {9, 10}
SKIP_CELLS = lambda r, c: c == 63 or r == 63


def parse_txt(path: Path):
    grid = [[9] * 64 for _ in range(64)]
    for line in path.read_text().splitlines():
        m = re.match(r"y(\d{2}):\s*(.*)$", line)
        if not m:
            continue
        y = int(m.group(1))
        toks = [t for t in m.group(2).replace("|", " ").split() if t]
        for x, t in enumerate(toks[:64]):
            grid[y][x] = HEX.get(t, 9)
    return grid


def components(grid):
    seen, out = set(), []
    for r in range(64):
        for c in range(64):
            if (r, c) in seen or SKIP_CELLS(r, c):
                continue
            col = grid[r][c]
            if col in SKIP_COLOURS:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            cells = {}
            while stack:
                y, x = stack.pop()
                cells[(y, x)] = col
                for dy, dx in N4:
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < 64
                        and 0 <= nx < 64
                        and (ny, nx) not in seen
                        and not SKIP_CELLS(ny, nx)
                        and grid[ny][nx] == col
                    ):
                        seen.add((ny, nx))
                        stack.append((ny, nx))
            out.append(cells)
    return out


def by_colour(ccs):
    d = {}
    for cc in ccs:
        col = next(iter(cc.values()))
        d.setdefault(col, []).append(cc)
    return d


def origin(fig):
    return min(y for y, _ in fig), min(x for _, x in fig)


def show(fig, title):
    if not fig:
        return f"{title}: empty"
    ys, xs = [y for y, _ in fig], [x for _, x in fig]
    lines = [f"{title} n={len(fig)} origin={origin(fig)}"]
    for y in range(min(ys), max(ys) + 1):
        lines.append(
            "".join(format(fig[(y, x)], "x") if (y, x) in fig else "." for x in range(min(xs), max(xs) + 1))
        )
    return "\n".join(lines)


def load_turns(level_dir: Path):
    frames = []
    for p in sorted(level_dir.glob("turn_*.txt")):
        n = int(p.stem.split("_")[1])
        frames.append((n, parse_txt(p)))
    return frames


def rels(a, b):
    if not a or not b:
        return {}
    return {
        "cong_T_any": S.congruent(a, b, translate=True, colours="any"),
        "cong_T_fixed": S.congruent(a, b, translate=True, colours="fixed"),
        "cong_F_any": S.congruent(a, b, translate=False, colours="any"),
        "direct_T_any": S.directly_congruent(a, b, translate=True, colours="any"),
        "sim_T_any": S.similar(a, b, translate=True, colours="any"),
        "transporter_T_any": len(S.transporter(a, b, translate=True, colours="any")) > 0,
        "same_origin": origin(a) == origin(b),
        "na": len(a),
        "nb": len(b),
        "oa": origin(a),
        "ob": origin(b),
    }


def run():
    frames = load_turns(CKPT / "level_0")
    print(f"level_0 frames: {len(frames)}  path={CKPT/'level_0'}")
    last = frames[-1][0]
    print(f"turns {frames[0][0]}..{last}  (uptake completed on LEFT at turn 16)")

    # Show the pair on turn 0 and last two turns.
    for n in (0, 3, 15, 16):
        g = dict(frames)[n]
        ccs = by_colour(components(g))
        print(f"\n=== turn {n} CC sizes by colour ===")
        for col, lst in sorted(ccs.items()):
            print(f"  colour {col}: {[len(x) for x in lst]}")
        fig4 = max(ccs.get(4, [{}]), key=len)
        fig11 = max(ccs.get(11, [{}]), key=len)
        print(show(fig4, "colour 4"))
        print(show(fig11, "colour 11"))

    print("\n========== colour-4 vs colour-11, every L0 turn ==========")
    print(f"{'t':>3} {'n4':>4} {'n11':>4} {'o4':>10} {'o11':>10}  T/any T/fix F/any direct sim  xport same_orig")
    rows = []
    for n, g in frames:
        ccs = by_colour(components(g))
        fig4 = max(ccs.get(4, [{}]), key=len)
        fig11 = max(ccs.get(11, [{}]), key=len)
        r = rels(fig4, fig11)
        rows.append((n, r, bool(fig4), bool(fig11)))
        print(
            f"{n:3d} {r.get('na',0):4d} {r.get('nb',0):4d} {str(r.get('oa')):>10} {str(r.get('ob')):>10}  "
            f"{int(r['cong_T_any'])}     {int(r['cong_T_fixed'])}     {int(r['cong_F_any'])}     "
            f"{int(r['direct_T_any'])}      {int(r['sim_T_any'])}    {int(r['transporter_T_any'])}     "
            f"{int(r['same_origin'])}"
        )

    def sep(name, pred):
        """True on last frame, false on all earlier frames that still have both pieces."""
        terminal = None
        earlier_true = []
        earlier_false = []
        missing = []
        for n, r, has4, has11 in rows:
            if not (has4 and has11):
                missing.append(n)
                continue
            v = pred(r)
            if n == last:
                terminal = v
            else:
                (earlier_true if v else earlier_false).append(n)
        separates = terminal is True and not earlier_true
        print(
            f"  {name}: terminal={terminal}  earlier_true={earlier_true}  "
            f"earlier_false={len(earlier_false)}  missing_pair={missing}  "
            f"{'SEPARATES' if separates else 'does not separate'}"
        )
        return separates

    print("\n========== §3 separator (true at turn 16, false before) ==========")
    sep("cong translate=True  colours=any", lambda r: r["cong_T_any"])
    sep("cong translate=True  colours=fixed", lambda r: r["cong_T_fixed"])
    sep("cong translate=False colours=any", lambda r: r["cong_F_any"])
    sep("directly_congruent T any", lambda r: r["direct_T_any"])
    sep("similar T any", lambda r: r["sim_T_any"])
    sep("transporter nonempty T any", lambda r: r["transporter_T_any"])
    sep("same origin", lambda r: r["same_origin"])

    # Colour 4 vs 5 as a negative pair (actor treated 5 as a different piece).
    print("\n========== colour-4 vs colour-5 (should NOT be the matching pair) ==========")
    print(f"{'t':>3} {'n4':>4} {'n5':>4} T/any F/any")
    for n, g in frames:
        ccs = by_colour(components(g))
        fig4 = max(ccs.get(4, [{}]), key=len)
        fig5 = max(ccs.get(5, [{}]), key=len)
        if not fig4 or not fig5:
            print(f"{n:3d} missing")
            continue
        print(
            f"{n:3d} {len(fig4):4d} {len(fig5):4d} "
            f"{int(S.congruent(fig4, fig5, translate=True, colours='any'))}     "
            f"{int(S.congruent(fig4, fig5, translate=False, colours='any'))}"
        )

    print("\nDONE")


if __name__ == "__main__":
    run()
