#!/usr/bin/env python3
"""Deterministically summarize the bounded epistemic-context A/B/C runs."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = {
    "A": ROOT
    / "results/arm-A/20260730_092305_ls20_memory_control_gemma-modelrun-fp4_cap44",
    "B": ROOT
    / "results/arm-B/20260730_092740_ls20_memory_evidence_only_gemma-modelrun-fp4_cap44",
    "C": ROOT
    / "results/arm-C/20260730_093940_ls20_memory_contradiction_first_gemma-modelrun-fp4_cap44",
}
TREATMENTS = {
    "A": "control",
    "B": "evidence_only",
    "C": "contradiction_first",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def components(grid: list[list[int]], color: int) -> list[dict]:
    points = {
        (row, col)
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value == color
    }
    found = []
    while points:
        todo = [points.pop()]
        component = []
        while todo:
            point = todo.pop()
            component.append(point)
            row, col = point
            for neighbor in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if neighbor in points:
                    points.remove(neighbor)
                    todo.append(neighbor)
        rows = [point[0] for point in component]
        cols = [point[1] for point in component]
        found.append(
            {
                "size": len(component),
                "bbox": [min(rows), min(cols), max(rows), max(cols)],
            }
        )
    return sorted(found, key=lambda item: (-item["size"], item["bbox"]))


def tracked_projection(grid: list[list[int]]) -> dict:
    """Projection Gemma itself tracked in C: color-12 wall and 15-cell color-9 player."""
    color_12 = components(grid, 12)
    color_9 = components(grid, 9)
    players = [
        item
        for item in color_9
        if item["size"] == 15 and 20 <= item["bbox"][0] < 55
    ]
    return {
        "wall_color_12": color_12[0] if color_12 else None,
        "player_color_9_size_15": players[0] if players else None,
    }


def longest_run(records: list[dict], predicate) -> dict:
    best_length = 0
    best_action = None
    best_start = None
    best_end = None
    current_length = 0
    current_action = None
    current_start = None
    for record in records:
        if predicate(record) and record["action"] == current_action:
            current_length += 1
        elif predicate(record):
            current_length = 1
            current_action = record["action"]
            current_start = record["step"]
        else:
            current_length = 0
            current_action = None
            current_start = None
        if current_length > best_length:
            best_length = current_length
            best_action = current_action
            best_start = current_start
            best_end = record["step"]
    return {
        "length": best_length,
        "action": best_action,
        "start_step": best_start,
        "end_step": best_end,
    }


def summarize(arm: str, run: Path) -> dict:
    result = json.loads((run / "result.json").read_text())
    states = read_jsonl(run / "ipc/states.jsonl")
    submitted_batches = read_jsonl(run / "ipc/actions.jsonl")
    events = read_jsonl(run / "agent_logs/nemo/team_leader/events.jsonl")

    executed = [
        action
        for state in states
        for action in (state.get("action_results") or [])
    ]
    batch_sizes = [
        len(state["action_results"])
        for state in states
        if state.get("action_results")
    ]

    step_records = []
    previous_projection = None
    for path in sorted((run / "steps").glob("step_*.json")):
        step = json.loads(path.read_text())
        projection = tracked_projection(step["grid"])
        if step["step"] > 0:
            step_records.append(
                {
                    "step": step["step"],
                    "action": step["action_chosen"],
                    "diff_pixels": step["diff_pixels"],
                    "projection": projection,
                    "projection_unchanged": projection == previous_projection,
                }
            )
        previous_projection = projection

    calls = [event for event in events if event.get("event") == "llm_call"]
    repl = [event for event in events if event.get("event") == "repl_execute"]
    agent_turns = [event for event in events if event.get("event") == "agent_turn"]
    assistant_text = "\n".join(
        path.read_text()
        for path in sorted(
            (run / "agent_logs/nemo/team_leader/messages").glob("*_assistant.md")
        )
    )

    submitted_count = sum(len(batch.get("actions") or []) for batch in submitted_batches)
    executed_count = len(executed)
    return {
        "arm": arm,
        "treatment": TREATMENTS[arm],
        "run_dir": run.relative_to(ROOT).as_posix(),
        "termination_reason": result["termination_reason"],
        "levels_completed": result["levels_completed"],
        "environment_actions": result["total_steps"],
        "hard_cap_exact": result["total_steps"] == 44,
        "submitted_action_batches": len(submitted_batches),
        "submitted_actions_after_agent_cap": submitted_count,
        "host_executed_actions": executed_count,
        "host_truncated_actions": submitted_count - executed_count,
        "batch_sizes_executed": batch_sizes,
        "maximum_executed_batch": max(batch_sizes),
        "action_distribution": dict(sorted(Counter(x["action"] for x in executed).items())),
        "pixel_changed_actions": sum(int(x["diff_pixels"] or 0) > 0 for x in executed),
        "pixel_noop_actions": sum(int(x["diff_pixels"] or 0) == 0 for x in executed),
        "longest_identical_action_run": longest_run(step_records, lambda _: True),
        "longest_identical_pixel_noop_run": longest_run(
            step_records, lambda item: item["diff_pixels"] == 0
        ),
        "projection_unchanged_actions": sum(
            item["projection_unchanged"] for item in step_records
        ),
        "pixel_changed_but_projection_unchanged_actions": sum(
            item["diff_pixels"] > 0 and item["projection_unchanged"]
            for item in step_records
        ),
        "longest_identical_projection_unchanged_run": longest_run(
            step_records, lambda item: item["projection_unchanged"]
        ),
        "model_calls": len(calls),
        "zero_token_model_calls": sum(
            event.get("input_tokens", 0) == 0 and event.get("output_tokens", 0) == 0
            for event in calls
        ),
        "input_tokens": sum(event.get("input_tokens", 0) for event in calls),
        "output_tokens": sum(event.get("output_tokens", 0) for event in calls),
        "reasoning_tokens": sum(event.get("reasoning_tokens", 0) for event in calls),
        "model_latency_seconds": round(
            sum(event.get("latency_s", 0.0) for event in calls), 3
        ),
        "wall_time_seconds": result["wall_time_seconds"],
        "repl_executions": len(repl),
        "repl_errors": sum(not event.get("success") for event in repl),
        "harness_fallback_actions": sum(
            "force-advance" in (event.get("rationale") or "") for event in agent_turns
        ),
        "explicit_reconciliation_status_counts": {
            word: len(re.findall(rf"\b{word}\b", assistant_text))
            for word in ("KEEP", "REVISE", "ABANDON")
        },
    }


def main() -> None:
    summaries = [summarize(arm, run) for arm, run in RUNS.items()]
    output = {
        "schema": "nooa-gemma-epistemic-context-summary-v1",
        "comparison_scope": "three bounded single-seed mechanism probes; not full-game runs",
        "projection_definition": (
            "largest color-12 component plus the 15-cell color-9 board component; "
            "the entities Gemma explicitly tracked in arm C"
        ),
        "runs": summaries,
    }
    destination = ROOT / "run-summary.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
