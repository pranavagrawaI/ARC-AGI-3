"""How big must a level difference be to mean anything?

usage: level-noise-floor.py

Duck is stochastic: the same game, same cap, same agent yields different level counts run to
run. Two recorded runs replay identical games across repeated passes, which gives a direct
estimate of that within-game variance. Two commit runs over the same 25 games share the game
difficulty (it cancels), so the difference of their totals carries each run's stochastic noise:

    sd(totalA - totalB) = sqrt(2 * n_games * mean_within_game_variance)

Use it to set the bar BEFORE reading a result, not after.
"""
from __future__ import annotations

import collections
import glob
import json
import math
import pathlib
import statistics as st

BASE = pathlib.Path.home() / "projects/arc-agi-3-reference/duck-critic-adapter/runs"
RUNS = [("periodic-v1 control (60 min cap)", BASE / "periodic-v1/control"),
        ("step3 incumbent (20 min cap)", BASE / "step3-paired/incumbent")]
N_GAMES = 25


def per_game(run: pathlib.Path) -> dict[str, list[int]]:
    d = collections.defaultdict(list)
    for p in sorted(glob.glob(f"{run}/kaggle-output/artifacts/*_events.jsonl")):
        stem = p.split("/")[-1].replace("_events.jsonl", "")
        ev = [json.loads(l) for l in open(p)]
        d[stem.rsplit("_", 1)[0].split("-")[0]].append(
            max((int(e.get("level") or 1) for e in ev), default=1) - 1)
    return d


def main() -> None:
    for name, run in RUNS:
        d = per_game(run)
        variances = [st.pvariance(v) for v in d.values() if len(v) > 1]
        mv = st.mean(variances)
        sd = math.sqrt(2 * N_GAMES * mv)
        print(f"{name}")
        print(f"  per-game level variance: {[f'{v:.2f}' for v in variances]}  mean {mv:.2f}")
        print(f"  over {N_GAMES} games: sd(delta) = {sd:.1f} levels; 1-sigma {sd:.0f}, 2-sigma {2*sd:.0f}")
    print("\nAgainst the 2026-09-18 reference of 17 levels, a challenger needs roughly")
    print("  >= 26 levels for a 2-sigma claim; 21-25 is suggestive; 13-20 is indistinguishable.")


if __name__ == "__main__":
    main()
