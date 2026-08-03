"""PROTOTYPE: score the completed trials against mechanical ground truth.

    python experiments/agents/0006-nooa/prototypes/commitment_controller/score_trials.py

Computes the post-run measures that require comparing the model's own labels
against the pixel diff or reading the completed artifact:

  * moved-versus-blocked classification accuracy;
  * player-identity reversals;
  * controller route-replacement and rejection counts;
  * whether the model wrote the round-level NOTES block.

Reads only recorded artifacts. No provider call, no spend.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TREATMENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TREATMENT_ROOT / "host"))

from commitment_controller import classify  # noqa: E402

RUNS = TREATMENT_ROOT / "artifacts" / "runs"
MOVED_CELLS = 52
QUIET_CELLS = 2


def changed_cells(before: dict, after: dict) -> int:
    grid_a, grid_b = before["pixels"][-1], after["pixels"][-1]
    return sum(
        1
        for row in range(len(grid_a))
        for col in range(len(grid_a[0]))
        if grid_a[row][col] != grid_b[row][col]
    )


def score(run_root: Path) -> dict[str, object] | None:
    summary_path = run_root / "summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    events_path = run_root / "worker" / "competence.jsonl"
    if not events_path.is_file():
        return None
    events = [json.loads(line) for line in events_path.open()]
    observations = {
        event["record"]["step"]: event["record"]["public"]
        for event in events
        if event.get("type") == "observation"
    }
    steps = sorted(observations)
    truth = [
        classify(
            changed_cells(observations[steps[i]], observations[steps[i + 1]]),
            moved=MOVED_CELLS,
            quiet=QUIET_CELLS,
        )
        for i in range(len(steps) - 1)
    ]

    process_path = run_root / "worker" / "process.jsonl"
    blocks = [
        json.loads(line)
        for line in process_path.open()
        if '"commitment_controller_state"' in line
    ]
    claimed: list[str] = []
    for record in blocks:
        for field in record["state"].splitlines():
            if field.startswith("LAST_OUTCOME:"):
                claimed.append(field.split(":", 1)[1].strip())
    # The block written at step N reports the outcome of transition N-1.
    claimed = claimed[1:]
    compared = list(zip(truth, claimed))
    correct = sum(1 for actual, said in compared if actual == said)

    bindings = [
        next(
            field.split(":", 1)[1].strip()
            for field in record["state"].splitlines()
            if field.startswith("PLAYER_BINDING:")
        )
        for record in blocks
    ]
    reversals = sum(1 for a, b in zip(bindings, bindings[1:]) if a != b)
    notes_written = sum(1 for record in blocks if record.get("notes_written"))

    controller = summary.get("commitment_controller") or {}
    return {
        "run": run_root.name,
        "condition": summary.get("knowledge_condition"),
        "actions": summary.get("actions"),
        "levels_completed": summary.get("levels_completed"),
        "stop_reason": summary.get("stop_reason"),
        "transitions_scored": len(compared),
        "classification_correct": correct,
        "classification_accuracy": (
            round(correct / len(compared), 3) if compared else None
        ),
        "truth_distribution": {
            outcome: truth.count(outcome) for outcome in sorted(set(truth))
        },
        "claimed_distribution": {
            outcome: claimed.count(outcome) for outcome in sorted(set(claimed))
        },
        "identity_reversals": reversals,
        "route_replacements": controller.get("route_replacements"),
        "controller_rejections": controller.get("rejections"),
        "notes_written_actions": f"{notes_written}/{len(blocks)}" if blocks else None,
    }


def main() -> int:
    rows = [
        result
        for run_root in sorted(RUNS.glob("controller-probe-*"))
        if (result := score(run_root)) is not None
    ]
    if not rows:
        print("no scored trials yet")
        return 0
    for row in rows:
        print(json.dumps(row, indent=1))
    print("\n=== summary")
    header = f"{'run':46s} {'act':>4s} {'lvl':>4s} {'class acc':>10s} {'rev':>4s} {'notes':>7s}"
    print(header)
    for row in rows:
        print(
            f"{row['run']:46s} {str(row['actions']):>4s} "
            f"{str(row['levels_completed']):>4s} "
            f"{str(row['classification_accuracy']):>10s} "
            f"{str(row['identity_reversals']):>4s} "
            f"{str(row['notes_written_actions']):>7s}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
