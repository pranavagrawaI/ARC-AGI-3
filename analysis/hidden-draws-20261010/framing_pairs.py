"""Stuck-vs-solved framing pairs: what did the stuck run believe, and what did the solver realise?

  python3 -B framing_pairs.py [chars]

For each (game, level), print the stuck run's prose reasoning at its 3rd, middle and last analysis step
on that level, then the solver's last two steps before the level-up. The level index is 0-based, as in
redraw_bound.attempts(); transcripts report 1-based levels.
"""
import glob, re, sys
from pathlib import Path
from redraw_bound import ROOT, RUNS

CASES = [  # (game, level, stuck run, solver run)
    ("r11l", 2, "pc", "c2"), ("su15", 1, "s1", "pc"), ("s5i5", 1, "c1", "s1"), ("lf52", 1, "c1", "s1"),
    ("ft09", 4, "s1", "c2"), ("cd82", 0, "c2", "c1"), ("tr87", 1, "pc", "c2"),
]
HEAD = re.compile(r"^--- analysis_step=\d+ \| action=\d+ .*---$", re.M)
LEVEL = re.compile(r"^Current state: step \d+, level (\d+)\.", re.M)


def segments(run, game):
    text = Path(glob.glob(str(ROOT / RUNS[run] / "transcripts" / f"{game}-*_p0.txt"))[0]).read_text(errors="replace")
    parts = HEAD.split(text)[1:]
    out, level = [], 1
    for p in parts:
        seen = LEVEL.findall(p)
        if seen:
            level = int(seen[0])  # the state line at the top of the step
        out.append((level, p))
    return out


def prose(seg, n):
    keep, in_code = [], False
    for line in seg.splitlines():
        if line.startswith("<tool_call>") or line.startswith("```"):
            in_code = not in_code if line.startswith("```") else True
            continue
        if line.startswith("</tool_call>"):
            in_code = False; continue
        if in_code or line.startswith(("{", "  \"", "- ", "#")) or len(line) < 40:
            continue
        keep.append(line)
    s = "\n".join(keep)
    return s[-n:]


def main(n):
    for game, L, stuck, solver in CASES:
        print(f"\n{'#' * 100}\n{game} L{L}: stuck={stuck} solver={solver}")
        st = [p for lv, p in segments(stuck, game) if lv == L + 1]
        so = [p for lv, p in segments(solver, game) if lv == L + 1]
        print(f"stuck steps on level: {len(st)}; solver steps: {len(so)}")
        for tag, idx in (("STUCK early", 2), ("STUCK mid", len(st) // 2), ("STUCK late", len(st) - 1)):
            if 0 <= idx < len(st):
                print(f"\n--- {tag} (step {idx}) ---\n{prose(st[idx], n)}")
        for idx in (len(so) - 2, len(so) - 1):
            if idx >= 0:
                print(f"\n--- SOLVER (step {idx} of {len(so)}) ---\n{prose(so[idx], n)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
