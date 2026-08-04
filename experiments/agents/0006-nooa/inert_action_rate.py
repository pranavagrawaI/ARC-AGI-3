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

The pre-registered measure reads only
``worker/host-observations/step-*.json``. A separate, explicitly post-hoc
terminal-supplement sensitivity analysis may fill the final successor from the
common runner trace after proving agreement at every overlapping step. Neither
analysis reads the rendered image, ``competence.jsonl``, or anything the
treatment alone produces.
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


def _host_observations(run_root: Path) -> dict[int, list[list[int]]]:
    directory = run_root / "worker" / "host-observations"
    frames = {}
    for path in sorted(directory.glob("step-*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        step = int(path.stem.removeprefix("step-"))
        frames[step] = _final_layer(record["pixels"])
    return frames


def _trace_observations(run_root: Path) -> dict[int, list[list[int]]]:
    path = run_root / "trace.jsonl"
    if not path.is_file():
        return {}
    frames = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        observation = record.get("observation")
        step = record.get("step")
        if (
            record.get("type") == "transition"
            and type(step) is int
            and isinstance(observation, dict)
            and "pixels" in observation
        ):
            frames[step] = _final_layer(observation["pixels"])
    return frames


def _terminal_trace_supplemented_observations(
    run_root: Path, *, final_step: int
) -> tuple[list[list[list[int]]], list[int]]:
    host = _host_observations(run_root)
    trace = _trace_observations(run_root)
    for step in host.keys() & trace.keys():
        if host[step] != trace[step]:
            raise ValueError(f"trace/host observation mismatch at step {step}")
    missing_host = [step for step in range(final_step) if step not in host]
    if missing_host:
        raise ValueError(f"missing host observations for steps {missing_host}")
    supplemented = [final_step] if final_step not in host and final_step in trace else []
    combined = host | {step: trace[step] for step in supplemented}
    missing = [step for step in range(final_step + 1) if step not in combined]
    if missing:
        raise ValueError(f"missing observations for steps {missing}")
    return [combined[step] for step in range(final_step + 1)], supplemented


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
        step = record.get("step")
        if isinstance(action, int):
            environment_step = (
                step if type(step) is int else record.get("action_point_id")
            )
            if type(environment_step) is int:
                chosen[environment_step] = action
    return chosen


def changed_cells(before: list[list[int]], after: list[list[int]]) -> list[tuple[int, int]]:
    return [
        (y, x)
        for y in range(len(before))
        for x in range(len(before[0]))
        if before[y][x] != after[y][x]
    ]


def _measure(
    run_root: Path,
    *,
    frames: list[list[list[int]]],
    trace_supplemented_steps: list[int],
) -> dict[str, Any]:
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
        "trace_supplemented_steps": trace_supplemented_steps,
        "unexpected_play_area_sizes": off_pattern,
        "transitions": transitions,
    }


def measure(run_root: Path) -> dict[str, Any]:
    """Return the pre-registered host-observation-only measure for one run."""

    host = _host_observations(run_root)
    missing = [step for step in range(max(host, default=-1) + 1) if step not in host]
    if missing:
        raise ValueError(f"missing host observations for steps {missing}")
    frames = [host[step] for step in range(len(host))]
    return _measure(run_root, frames=frames, trace_supplemented_steps=[])


def measure_terminal_trace_sensitivity(run_root: Path) -> dict[str, Any]:
    """Post-hoc analysis that may recover only the terminal trace successor."""

    final_step = len(_actions(run_root))
    frames, supplemented = _terminal_trace_supplemented_observations(
        run_root, final_step=final_step
    )
    return _measure(
        run_root,
        frames=frames,
        trace_supplemented_steps=supplemented,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, nargs="+")
    parser.add_argument("--json", action="store_true", help="emit the full record")
    parser.add_argument(
        "--terminal-trace-sensitivity",
        action="store_true",
        help="post-hoc: supplement only a missing terminal host successor from trace",
    )
    args = parser.parse_args(argv)

    measurement = (
        measure_terminal_trace_sensitivity
        if args.terminal_trace_sensitivity
        else measure
    )
    results = [measurement(root.resolve()) for root in args.run_root]
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
