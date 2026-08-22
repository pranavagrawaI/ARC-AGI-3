#!/usr/bin/env python3
"""Test whether mixed Builder jobs explain the selected turn-18 timeout."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    ROOT
    / "runs/tycho-nooa-trigger-replacement-gemma4-31b-ar25-20260821T154601Z"
    / "game"
)
OUTPUT = Path(__file__).resolve().parent
ACCURACY = re.compile(
    r"(?:accepted|exact)_transition_accuracy \(simulation_accuracy\) = ([0-9.]+)"
)
REPLAY = re.compile(
    r"transition_replay = matched (\d+) / evaluated (\d+) / observed (\d+)"
)


def main() -> None:
    calls = [
        row
        for row in _jsonl(SOURCE / "builder-tool-trace.jsonl")
        if row.get("level") == 0 and row.get("turn_in_level") == 18
    ]
    trigger_rows = [
        row
        for row in _jsonl(SOURCE / "trigger-builder-trace.jsonl")
        if row.get("level") == 0 and row.get("turn_in_level") == 18
    ]
    trigger = _one(trigger_rows, "mechanical_trigger")
    deadline = _one(trigger_rows, "builder_deadline_exceeded")
    if [row["call_index"] for row in calls] != list(range(1, 29)):
        raise RuntimeError("turn-18 call sequence drifted")
    if deadline.get("partial_tool_calls") != len(calls):
        raise RuntimeError("deadline and tool-call counts disagree")
    if trigger.get("cause") != "dynamics":
        raise RuntimeError("turn-18 request is no longer dynamics-scoped")

    classified = [_classify(row) for row in calls]
    counts = Counter(row["job"] for row in classified)
    scores = [_score(row) for row in calls if _score(row) is not None]
    final_score = scores[-1]
    if final_score != {
        "call_index": 27,
        "accuracy": 0.5,
        "matched": 8,
        "evaluated": 16,
        "observed": 18,
    }:
        raise RuntimeError("turn-18 score commitment drifted")

    caps = [5, 10, 12, 20, 26, 27, 28]
    cap_sweep = [
        {"call_cap": cap, "latest_score": _latest_score(scores, cap)} for cap in caps
    ]
    dominant = counts.most_common(1)[0]
    dominant_share = dominant[1] / len(calls)
    mixed_job_prediction_met = len(counts) >= 2 and dominant_share <= 0.8
    result = {
        "schema": "tycho-nooa-builder-job-scope-replay-v1",
        "experiment_card": str(
            (OUTPUT / "EXPERIMENT-CARD.md").relative_to(ROOT)
        ),
        "source_run": str(SOURCE.parent.relative_to(ROOT)),
        "source_boundary": {"level": 0, "turn_in_level": 18},
        "request_job": trigger["cause"],
        "request_reason": trigger["reason"],
        "deadline": deadline,
        "classified_calls": classified,
        "job_counts": dict(sorted(counts.items())),
        "dominant_job": dominant[0],
        "dominant_job_share": dominant_share,
        "score_trajectory": scores,
        "call_cap_sweep": cap_sweep,
        "prediction": {
            "mixed_job_cause_requires_distinct_jobs": 2,
            "maximum_dominant_share": 0.8,
            "met": mixed_job_prediction_met,
        },
        "observation": (
            "The request was explicitly dynamics-scoped and all 28 calls remained "
            "mechanics work. The first positive score appeared at call 27."
        ),
        "judgment": "contradicted_as_selected_timeout_cause",
        "decision": "do_not_implement_job_allocator",
        "next_test": (
            "continue a fresh mechanics Builder from the retained 8/16 candidate and "
            "compare score gained per additional call"
        ),
        "claim_ceiling": (
            "This rules out job mixing as the cause of the selected turn-18 timeout; "
            "it does not show that typed job allocation is never useful elsewhere."
        ),
        "status": "pass",
        "execution": {"provider_model_calls": 0, "arc_environment_calls": 0},
    }
    _write_json(OUTPUT / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def _classify(row: dict) -> dict:
    name = row["name"]
    arguments = row.get("arguments") or {}
    if name == "ls" or (
        name == "read_file"
        and arguments.get("path")
        in {"notes/actor_beliefs.md", "notes/world_model.md", "world_model.py"}
    ):
        subtype = "mechanics_context"
    elif name == "run_python" and "wmlib" in str(arguments.get("code", "")):
        subtype = "mechanics_probe"
    elif name == "edit_function" and arguments.get("name") in {
        "init_state",
        "transition",
        "render",
    }:
        subtype = "mechanics_model_edit"
    else:
        raise RuntimeError(f"unclassifiable call {row['call_index']}: {name}")
    return {"call_index": row["call_index"], "tool": name, "job": "mechanics", "subtype": subtype}


def _score(row: dict) -> dict | None:
    result = str(row.get("result", ""))
    accuracy = ACCURACY.search(result)
    replay = REPLAY.search(result)
    if accuracy is None and replay is None:
        return None
    if accuracy is None or replay is None:
        raise RuntimeError(f"partial verifier score at call {row['call_index']}")
    return {
        "call_index": row["call_index"],
        "accuracy": float(accuracy.group(1)),
        "matched": int(replay.group(1)),
        "evaluated": int(replay.group(2)),
        "observed": int(replay.group(3)),
    }


def _latest_score(scores: list[dict], cap: int) -> dict | None:
    eligible = [score for score in scores if score["call_index"] <= cap]
    return eligible[-1] if eligible else None


def _one(rows: list[dict], event: str) -> dict:
    matches = [row for row in rows if row.get("event") == event]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {event}, found {len(matches)}")
    return matches[0]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
