"""Summarize preserved NOOA ARC runs without reading credentials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(run_dir: Path) -> dict[str, Any] | None:
    result_path = run_dir / "result.json"
    if not result_path.exists():
        return None
    result = _read_json(result_path)
    events = _read_jsonl(run_dir / "agent_logs/nemo/team_leader/events.jsonl")
    submitted = _read_jsonl(run_dir / "ipc/actions.jsonl")
    llm_calls = [event for event in events if event.get("event") == "llm_call"]
    repl = [event for event in events if event.get("event") == "repl_execute"]
    agent_turns = [event for event in events if event.get("event") == "agent_turn"]
    return {
        "run_dir": str(run_dir),
        "game_id": result.get("game_id"),
        "variant": result.get("variant"),
        "termination_reason": result.get("termination_reason"),
        "levels_completed": result.get("levels_completed"),
        "environment_steps": result.get("total_steps"),
        "wall_time_seconds": result.get("wall_time_seconds"),
        "model_calls": len(llm_calls),
        "prompt_tokens": sum(int(event.get("input_tokens") or 0) for event in llm_calls),
        "completion_tokens": sum(int(event.get("output_tokens") or 0) for event in llm_calls),
        "model_latency_seconds": round(
            sum(float(event.get("latency_s") or 0.0) for event in llm_calls), 3
        ),
        "repl_executions": len(repl),
        "repl_errors": sum(event.get("success") is False for event in repl),
        "submitted_batches": submitted,
        "executed_turns": [
            {
                "turn": event.get("turn"),
                "step": event.get("step"),
                "actions": event.get("actions"),
                "rationale": event.get("rationale"),
            }
            for event in agent_turns
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = []
    for result_path in sorted(args.results_root.rglob("result.json")):
        item = summarize(result_path.parent)
        if item is not None:
            runs.append(item)
    payload = {"schema": "nooa-gemma-run-summary-v1", "runs": runs}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
