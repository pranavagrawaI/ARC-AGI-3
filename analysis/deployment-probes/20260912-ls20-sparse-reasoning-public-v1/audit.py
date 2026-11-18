#!/usr/bin/env python3
"""Audit the immutable LS20 sparse-reasoning Kaggle run. No network calls."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw" / "qualification"


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    result = json.loads((RAW / "result.json").read_text())
    system_before = json.loads((RAW / "system-before.json").read_text())
    shutdown = json.loads((RAW / "shutdown.json").read_text())
    teardown = json.loads((RAW / "teardown-log-classification.json").read_text())
    request_counts = json.loads((RAW / "request-count-facts.json").read_text())
    allocation = rows(RAW / "budget-allocations.jsonl")
    workspaces = list((RAW / "solver-workspaces").glob("*/ipc"))
    if len(workspaces) != 1:
        raise RuntimeError(f"expected one solver workspace, found {len(workspaces)}")
    ipc = workspaces[0]
    calls = rows(ipc / "model_calls.jsonl")
    lifecycle = rows(ipc / "request_lifecycle.jsonl")
    actions = rows(ipc / "actions.jsonl")
    policy_attempts = rows(ipc / "policy_attempts.jsonl")
    deep = rows(ipc / "deep_investigations.jsonl")
    deep_outcomes = rows(ipc / "deep_outcomes.jsonl")
    admitted = [row for row in lifecycle if row.get("event") == "admitted"]
    completed = [row for row in lifecycle if row.get("event") == "completed"]
    failed = [row for row in lifecycle if row.get("event") == "failed"]
    routine_reasoning_tokens = sum(int(row.get("reasoning_tokens") or 0) for row in calls)
    routine_reasoning_present = sum(bool(row.get("reasoning_present")) for row in calls)
    evidence_complete = all(
        isinstance(row.get("request_messages"), list)
        and isinstance(row.get("tool_results"), list)
        for row in calls
    )
    launcher = result.get("official_launcher", result.get("launcher_result", {}))
    games = launcher.get("games", [])
    game = games[0] if len(games) == 1 else {}
    sessions = launcher.get("solver_sessions", {})
    session_games = sessions.get("games", {})
    episode = {}
    if len(session_games) == 1:
        episodes = next(iter(session_games.values())).get("episodes", [])
        if len(episodes) == 1:
            episode = episodes[0]
    quick = episode.get("quick_metrics", {})
    launcher_actions = int(game.get("actions") or 0)
    submitted_actions = int(quick.get("submitted_actions") or 0)
    skipped_actions = int(quick.get("host_skipped_actions") or 0)
    gpu_text = system_before.get("gpu", {}).get("stdout", "")
    model_config = json.loads((ipc.parent / "launch.json").read_text()).get("model_config", {})
    routine_config = json.loads(model_config.get("ARC_LLM_EXTRA_BODY", "{}"))
    routine_config_off = (
        routine_config.get("chat_template_kwargs", {}).get("enable_thinking") is False
        and routine_config.get("reasoning", {}).get("enabled") is False
    )
    wall_seconds = [float(row.get("wall_seconds") or 0) for row in calls]
    completion_tokens = [int(row.get("completion_tokens") or 0) for row in calls]
    deep_shape_valid = all(
        isinstance(row.get("question"), str) and bool(row["question"].strip())
        and isinstance(row.get("evidence_refs"), list)
        and 1 <= len(row["evidence_refs"]) <= 8
        and int(row.get("completion_tokens") or 0) <= 2048
        and int(row.get("reasoning_tokens") or 0) > 0
        for row in deep
    )
    lifecycle_reconciles = (
        len(admitted) == len(completed) + len(failed)
        and len(calls) == len(completed)
        and not failed
    )
    checks = {
        "result_passed": result.get("status") == "passed",
        "one_game": len(games) == 1,
        "routine_reasoning_off": routine_reasoning_tokens == 0 and routine_reasoning_present == 0,
        "sparse_call_cap": len(deep) <= 1,
        "sparse_call_shape": deep_shape_valid,
        "model_evidence_complete": evidence_complete,
        "request_lifecycle_reconciles": lifecycle_reconciles,
        "runtime_contract_matches": (
            result.get("official_launcher", {}).get("solver_sessions", {})
            .get("games", {}).get(game.get("game_id"), {})
            .get("worker_metadata", {}).get("contract_sha256")
            == "864838a6acde0530e4e54d6cd74b607ad2046c35dcbdff3b15a222f6ad836c2a"
        ),
        "routine_config_off": routine_config_off,
        "gpu_is_rtx_pro_6000": "NVIDIA RTX PRO 6000 Blackwell Server Edition" in gpu_text,
        "request_count_verified": request_counts.get("whole_model_request_count_verified") is True,
        "shutdown_passed": shutdown.get("status") == "passed" and shutdown.get("containment_passed") is True,
        "teardown_allowed": teardown.get("allowed") is True and teardown.get("forbidden_count") == 0,
        "admission_allocation_600s": (
            len(allocation) == 1 and float(allocation[0].get("admission_seconds") or 0) == 600.0
        ),
    }
    audit = {
        "schema": "ls20-sparse-reasoning-public-v1-audit",
        "run_status": result.get("status"),
        "game": game,
        "levels_completed": game.get("levels_completed"),
        "stop_reason": game.get("stop_reason"),
        "action_batches": len(actions),
        "policy_attempts": len(policy_attempts),
        "submitted_actions": submitted_actions,
        "launcher_executed_actions": launcher_actions,
        "host_skipped_actions": skipped_actions,
        "routine_model_calls": len(calls),
        "actor_calls": sum(row.get("request_purpose") == "actor" for row in calls),
        "critic_calls": sum(row.get("request_purpose") == "critic" for row in calls),
        "routine_reasoning_tokens": routine_reasoning_tokens,
        "routine_reasoning_present_calls": routine_reasoning_present,
        "mean_routine_wall_seconds": sum(wall_seconds) / len(wall_seconds) if wall_seconds else None,
        "mean_routine_completion_tokens": sum(completion_tokens) / len(completion_tokens) if completion_tokens else None,
        "max_routine_wall_seconds": max(wall_seconds, default=None),
        "max_routine_completion_tokens": max(completion_tokens, default=None),
        "gpu": gpu_text.strip(),
        "deep_investigations": deep,
        "deep_outcomes": deep_outcomes,
        "request_lifecycle": {
            "admitted": len(admitted), "completed": len(completed),
            "failed": len(failed), "reconciles": lifecycle_reconciles,
        },
        "validity_checks": checks,
        "valid": all(checks.values()),
    }
    (HERE / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
