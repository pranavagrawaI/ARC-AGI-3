#!/usr/bin/env python3
"""Is the stall separation a property of the signal, or of the thresholds I picked?

`stall_repetition`'s window, action floor and similarity cutoff were chosen by hand.
A separation that only survives at those values is a tuned result, so this sweeps them
against every failure/success trace and reports the direction at each setting.

It also sizes the endpoints for the live card: how many reviewable stall events an arm
would produce, and how often a stall recurs with no intervention at all -- the control
baseline for recommitment failure, measurable before spending anything.

    python3 sweep.py
"""

from __future__ import annotations

import itertools
import sys
from math import comb
from pathlib import Path

import replay

PROBES = Path(__file__).resolve().parent.parent / "local-probes"

# levels_completed is the arm label: 0 -> failure trace, 1 -> success trace.
TRACES: dict[str, tuple[str, int]] = {
    "q4": (
        (
            "20260904T-goal-candidate-q4/results/nemo_solver/"
            "20260904_165006_ls20_mdfiles_q4"
        ),
        0,
    ),
    "q3b": (
        (
            "20260904T-host-control-loop-q3/results/nemo_solver/"
            "20260904_035318_ls20_mdfiles_q3b"
        ),
        0,
    ),
    "q2d": (
        (
            "20260903T-skill-following-q2/results/nemo_solver/"
            "20260903_160733_ls20_mdfiles_q2d"
        ),
        1,
    ),
    "q3c": (
        (
            "20260904T-host-control-loop-q3/results/nemo_solver/"
            "20260904_140403_ls20_mdfiles_q3c"
        ),
        1,
    ),
}

WINDOWS = (4, 6, 8)
FLOORS = (15, 25, 40)
SIMILARITIES = (0.45, 0.55, 0.65)
RECURRENCE_STEPS = 20


def _rate(run: replay.Run, **kwargs: object) -> float:
    openings = replay.trigger_stall_repetition(run, **kwargs)  # type: ignore[arg-type]
    return len(openings) * 100.0 / max(run.max_step, 1)


def sweep(runs: dict[str, replay.Run]) -> list[tuple[int, int, float, float, float, float]]:
    fail = [k for k, (_, lv) in TRACES.items() if lv == 0]
    ok = [k for k, (_, lv) in TRACES.items() if lv == 1]
    rows = []
    for window, floor, similarity in itertools.product(WINDOWS, FLOORS, SIMILARITIES):
        kw = {"window": window, "min_actions": floor, "similarity": similarity}
        f = sum(_rate(runs[k], **kw) for k in fail) / len(fail)
        o = sum(_rate(runs[k], **kw) for k in ok) / len(ok)
        rows.append((window, floor, similarity, f, o, f / o if o else float("inf")))
    return rows


def endpoint_sizing(runs: dict[str, replay.Run]) -> str:
    lines = ["STALL EVENTS PER RUN -- the unit a mechanistic endpoint is measured in", ""]
    per_arm = {0: [], 1: []}
    for name, (_, levels) in TRACES.items():
        run = runs[name]
        steps = [o.step for o in replay.trigger_stall_repetition(run)]
        recur = sum(
            1 for i, s in enumerate(steps) if i and s - steps[i - 1] <= RECURRENCE_STEPS
        )
        share = f"{recur}/{len(steps)}" if steps else "0/0"
        per_arm[levels].append(len(steps))
        lines.append(
            
                f"  {name:<4} {levels} level(s)  {len(steps):>3} stall events over "
                f"{run.max_step:>3} steps   recurrence within "
                f"{RECURRENCE_STEPS} steps: {share}"
            
        )
    fail_mean = sum(per_arm[0]) / len(per_arm[0])
    ok_mean = sum(per_arm[1]) / len(per_arm[1])
    lines += [
        "",
        f"  failure traces average {fail_mean:.0f} stall events per run",
        f"  success traces average {ok_mean:.0f} stall events per run",
        (
            f"  a 3-run arm yields roughly {3 * ok_mean:.0f}-{3 * fail_mean:.0f} "
            "reviewable stall events"
        ),
    ]
    return "\n".join(lines)


def completion_power() -> str:
    """Completion is a coin flip on this chassis family; size the endpoint honestly."""
    observed = [lv for _, lv in TRACES.values()]
    base = sum(observed) / len(observed)

    def pmf(n: int, k: int, p: float) -> float:
        return comb(n, k) * p**k * (1 - p) ** (n - k)

    lines = [
        "POWER OF A COMPLETION ENDPOINT AT n=3 PER ARM",
        "",
        (
            f"  observed base rate: {sum(observed)} of {len(observed)} runs cleared a "
            f"level (p~{base:.2f})"
        ),
        "",
        f"  {'treatment p':>12} {'P(0/3 vs 3/3)':>15} {'P(treat>control)':>18}",
    ]
    null = sum(
        pmf(3, c, base) * pmf(3, t, base) for c in range(4) for t in range(4) if t > c
    )
    for pt in (0.7, 0.8, 0.9):
        clean = pmf(3, 0, base) * pmf(3, 3, pt)
        better = sum(
            pmf(3, c, base) * pmf(3, t, pt) for c in range(4) for t in range(4) if t > c
        )
        lines.append(f"  {pt:>12} {clean:>15.2f} {better:>18.2f}")
    lines += [
        f"  {'(null)':>12} {'-':>15} {null:>18.2f}",
        "",
        "  Even a treatment lifting completion to 0.9 gives under a 1-in-10 chance of an",
        (
            "  unambiguous split, and 'treatment > control' only moves from "
            f"{null:.2f} to 0.76."
        ),
        "  Completion is a reportable secondary. It cannot be the gate at n=3.",
    ]
    return "\n".join(lines)


def main() -> int:
    runs = {}
    for name, (rel, _) in TRACES.items():
        path = PROBES / rel
        if not path.exists():
            print(f"missing trace: {path}", file=sys.stderr)
            return 2
        runs[name] = replay.load_run(path)

    rows = sweep(runs)
    print("SENSITIVITY SWEEP -- stall_repetition separation (failure rate / success rate)")
    print()
    print(f"  {'window':>6} {'floor':>6} {'simil':>6} | {'fail/100':>8} {'ok/100':>7} {'ratio':>7}")
    for window, floor, similarity, f, o, ratio in rows:
        print(f"  {window:>6} {floor:>6} {similarity:>6} | {f:>8.1f} {o:>7.1f} {ratio:>7.2f}")
    correct = sum(1 for r in rows if r[5] > 1.0)
    print()
    print(f"  correct direction at {correct}/{len(rows)} settings; "
          f"ratio {min(r[5] for r in rows):.2f}x .. {max(r[5] for r in rows):.2f}x")
    print()
    print(endpoint_sizing(runs))
    print()
    print(completion_power())
    return 0 if correct == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
