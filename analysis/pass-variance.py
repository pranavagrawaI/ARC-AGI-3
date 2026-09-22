"""Does a second attempt beat a longer first attempt?

usage: pass-variance.py   (paths are the recorded runs; no arguments)

Three Duck-alone runs share the same five games at three per-game caps, with repeated passes:
Step 1 (600 s x 2), Step 3 incumbent (1200 s x 4), periodic-v1 control (3600 s x 4). Together
they give a dose-response curve for per-game time and a direct read of pass-to-pass variance
on identical games -- the two quantities that decide whether spare envelope should buy longer
games or more attempts.
"""
from __future__ import annotations

import collections
import glob
import json
import pathlib

BASE = pathlib.Path.home() / "projects/arc-agi-3-reference"
RUNS = [
    ("10 min", BASE / "duck-harness/runs/step1-duck-alone"),
    ("20 min", BASE / "duck-critic-adapter/runs/step3-paired/incumbent"),
    ("60 min", BASE / "duck-critic-adapter/runs/periodic-v1/control"),
]
GAMES = ["sb26", "su15", "lf52", "ft09", "ls20"]


def per_game(run: pathlib.Path) -> dict[str, list[int]]:
    d = collections.defaultdict(list)
    for p in sorted(glob.glob(f"{run}/kaggle-output/artifacts/*_events.jsonl")):
        stem = p.split("/")[-1].replace("_events.jsonl", "")
        ev = [json.loads(l) for l in open(p)]
        levels = max((int(e.get("level") or 1) for e in ev), default=1) - 1
        d[stem.rsplit("_", 1)[0].split("-")[0]].append(levels)
    return d


def main() -> None:
    print(f"{'cap':<8}" + "".join(f"{g:>9}" for g in GAMES) + f"{'E[games>=1]':>13}{'E[levels]':>11}")
    last = {}
    for cap, run in RUNS:
        d = per_game(run)
        last = d
        ps = [sum(1 for x in d[g] if x >= 1) / len(d[g]) for g in GAMES]
        lv = [sum(d[g]) / len(d[g]) for g in GAMES]
        print(f"{cap:<8}" + "".join(f"{p:>9.2f}" for p in ps) + f"{sum(ps):>13.2f}{sum(lv):>11.2f}")
    print("\ncell = fraction of passes reaching >=1 level, same five games throughout")

    one = [sum(1 for x in last[g] if x >= 1) / len(last[g]) for g in GAMES]
    two = [1 - (1 - p) ** 2 for p in one]
    print(f"\nat 60 min, one attempt : E[games>=1] = {sum(one):.2f} / 5")
    print(f"two 60 min attempts    : E[games>=1] = {sum(two):.2f} / 5  (+{sum(two)-sum(one):.2f}, independence assumed)")
    print("gain concentrates on the unreliable games: " +
          ", ".join(f"{g} {p:.2f}->{t:.2f}" for g, p, t in zip(GAMES, one, two) if t > p + 1e-9))


if __name__ == "__main__":
    main()
