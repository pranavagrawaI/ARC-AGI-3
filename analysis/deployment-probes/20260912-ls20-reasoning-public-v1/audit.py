#!/usr/bin/env python3
"""Audit the immutable LS20 reasoning-on Kaggle run. Makes no network calls."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def actor_text(record: dict) -> str:
    parts = [str(record.get("response_content") or "")]
    for call in record.get("tool_calls") or []:
        parts.append(str(call.get("arguments") or ""))
    return "\n".join(parts)


def main() -> None:
    result_path = RAW / "qualification" / "result.json"
    result = json.loads(result_path.read_text())
    workspaces = list((RAW / "qualification" / "solver-workspaces").glob("*/ipc"))
    if len(workspaces) != 1:
        raise RuntimeError(f"expected one solver workspace, found {len(workspaces)}")
    ipc = workspaces[0]
    calls = rows(ipc / "model_calls.jsonl")
    actor = [row for row in calls if row.get("request_purpose") == "actor"]
    critic = [row for row in calls if row.get("request_purpose") == "critic"]
    lifecycle = rows(ipc / "request_lifecycle.jsonl")
    actions = rows(ipc / "actions.jsonl")

    predicted = [
        item.get("expect")
        for action in actions
        for item in (action.get("predicted_path") or [])
    ]
    terminal_predictions = [value for value in predicted if value in {"level_up", "terminal"}]
    patterns = {
        "context_evicted": re.compile(r"context (?:was |has been )?evicted", re.I),
        "budget_100_100": re.compile(r"budget exhausted\s*\(100/100\)", re.I),
        "budget_exhausted": re.compile(r"(?:model|action) budget (?:is )?exhausted", re.I),
    }
    phrase_hits = []
    for index, record in enumerate(actor):
        text = actor_text(record)
        for name, pattern in patterns.items():
            for match in pattern.finditer(text):
                request = json.dumps(record.get("request_messages") or [], ensure_ascii=False)
                phrase_hits.append({
                    "actor_index": index,
                    "turn": record.get("turn"),
                    "step": record.get("step"),
                    "pattern": name,
                    "claim": match.group(0),
                    "request_contains_claim": bool(pattern.search(request)),
                    "actor_text": text,
                })
    admitted = [row for row in lifecycle if row.get("event") == "admitted"]
    completed = [row for row in lifecycle if row.get("event") == "completed"]
    failed = [row for row in lifecycle if row.get("event") == "failed"]
    games = result.get("launcher_result", result).get("games", [])
    game = games[0] if len(games) == 1 else {}
    reasoning_tokens = [int(row.get("reasoning_tokens") or 0) for row in actor]
    reasoning_present = [bool(row.get("reasoning_present")) for row in actor]
    evidence_complete = all(
        isinstance(row.get("request_messages"), list)
        and isinstance(row.get("tool_results"), list)
        for row in calls
    )
    audit = {
        "schema": "ls20-reasoning-public-v1-audit",
        "result_path": str(result_path),
        "ipc": str(ipc),
        "run_status": result.get("status"),
        "game": game,
        "levels_completed": game.get("levels_completed"),
        "executed_actions": len(actions),
        "actor_calls": len(actor),
        "critic_calls": len(critic),
        "calls_per_action": len(actor) / len(actions) if actions else None,
        "reasoning": {
            "calls_with_tokens": sum(value > 0 for value in reasoning_tokens),
            "calls_with_content": sum(reasoning_present),
            "total_tokens": sum(reasoning_tokens),
            "max_tokens": max(reasoning_tokens, default=0),
        },
        "prediction_classes": dict(Counter(value for value in predicted if value)),
        "terminal_class_predictions": terminal_predictions,
        "false_host_event_candidates": phrase_hits,
        "model_evidence_complete": evidence_complete,
        "request_lifecycle": {
            "admitted": len(admitted),
            "completed": len(completed),
            "failed": len(failed),
            "reconciles": len(admitted) == len(completed) + len(failed)
            and len(calls) == len(completed),
        },
        "validity_checks": {
            "result_passed": result.get("status") == "passed",
            "one_game": len(games) == 1,
            "reasoning_active": any(reasoning_tokens) and any(reasoning_present),
            "model_evidence_complete": evidence_complete,
            "request_lifecycle_reconciles": len(admitted) == len(completed) + len(failed)
            and len(calls) == len(completed),
        },
    }
    audit["valid"] = all(audit["validity_checks"].values())
    (HERE / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
