"""Compute the pre-registered primary measure for one run root.

An ls20 frame carries a move counter in its bottom rows, and that counter ticks
whether or not the action did anything. So "did the frame change" is at ceiling
-- 43 of 44 control transitions changed something -- and says nothing. Strip the
counter band and the signal is stark: play-area change across those transitions
was exactly 50 cells or exactly 0, never a value between. An action either moved
the world or it did not.

That bimodality is what makes this measurable at 44 actions on a game neither
arm can finish. See COMPARISON.md; this file implements the measure defined
there and adds nothing to it.

The measure reads only ``worker/host-observations/step-*.json``, which both arms
archive identically from the frozen worker. It does not read the rendered image,
``competence.jsonl``, or anything the treatment alone produces, so neither arm
can be measured on evidence the other cannot supply.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

# Rows 61-62 carry the ls20 move counter. Verified on 43 archived control
# transitions: the only cells that ever changed at or below this row were in
# those two rows, and every transition either changed 50 play-area cells or none.
HUD_FIRST_ROW = 61


def _final_layer(pixels: Any) -> list[list[int]]:
    """Return the last layer, which is the one the environment renders.

    404 of 591 recorded tool calls in an earlier probe read ``pixels[0]`` and
    measured a layer the game was not drawing on.
    """

    layer = pixels
    while isinstance(layer[0][0], list):
        layer = layer[-1]
    return layer


def _observations(run_root: Path) -> list[list[list[int]]]:
    directory = run_root / "worker" / "host-observations"
    frames = []
    for path in sorted(directory.glob("step-*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        frames.append(_final_layer(record["pixels"]))
    return frames


def _actions(run_root: Path) -> dict[int, int]:
    chosen: dict[int, int] = {}
    path = run_root / "worker" / "process.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("kind") != "decision_validated":
            continue
        decision = record.get("decision") or {}
        action = decision.get("action")
        if isinstance(action, int):
            chosen[record["action_point_id"]] = action
    return chosen


def changed_cells(before: list[list[int]], after: list[list[int]]) -> list[tuple[int, int]]:
    return [
        (y, x)
        for y in range(len(before))
        for x in range(len(before[0]))
        if before[y][x] != after[y][x]
    ]


def measure(run_root: Path) -> dict[str, Any]:
    """Return the inert-action rate and the action distribution for one run."""

    frames = _observations(run_root)
    actions = _actions(run_root)
    transitions = []
    for step in range(len(frames) - 1):
        changed = changed_cells(frames[step], frames[step + 1])
        play_area = [cell for cell in changed if cell[0] < HUD_FIRST_ROW]
        transitions.append(
            {
                "step": step,
                "action": actions.get(step),
                "play_area_cells": len(play_area),
                "inert": not play_area,
            }
        )
    inert = sum(1 for item in transitions if item["inert"])
    scored = len(transitions)
    # A transition whose play-area change is neither 0 nor 50 would mean the
    # bimodality the measure rests on does not hold in this run. Report it
    # rather than folding it silently into one side.
    off_pattern = sorted(
        {item["play_area_cells"] for item in transitions if item["play_area_cells"] not in (0, 50)}
    )
    return {
        "run_root": str(run_root),
        "scored_transitions": scored,
        "inert_actions": inert,
        "inert_rate": (inert / scored) if scored else None,
        "action_distribution": dict(sorted(Counter(actions.values()).items())),
        "unexpected_play_area_sizes": off_pattern,
        "transitions": transitions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, nargs="+")
    parser.add_argument("--json", action="store_true", help="emit the full record")
    args = parser.parse_args(argv)

    results = [measure(root.resolve()) for root in args.run_root]
    if args.json:
        print(json.dumps(results, indent=1))
        return 0
    for result in results:
        rate = result["inert_rate"]
        print(f"{result['run_root']}")
        print(
            f"  inert {result['inert_actions']}/{result['scored_transitions']}"
            f" = {rate:.0%}" if rate is not None else "  no scored transitions"
        )
        print(f"  actions {result['action_distribution']}")
        if result["unexpected_play_area_sizes"]:
            print(f"  WARNING off-pattern play-area sizes: {result['unexpected_play_area_sizes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
