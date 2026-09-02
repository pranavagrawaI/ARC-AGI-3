"""Validate a resumable public-25 callback against the outer server evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import resumable_public25 as harness

OUTER_CONTRACT_SCHEMA = "tycho-nooa-public25-outer-callback-contract-v1"
LIFECYCLE_REQUIREMENT = "accepted_r8_l3b_term_only_unchanged"
NOTEBOOK_SUBPROCESS_REQUIREMENT = "check_true"
QUALIFICATION_STATUSES = {"complete", "partial_budget", "partial_failure"}


def _read_regular_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is not one regular file: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{label} is not one JSON object")
    return value


def _validated_callback(
    callback_path: Path, checkpoint_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    callback = _read_regular_json(callback_path, "resumable callback result")
    if (
        callback.get("schema") != harness.RESULT_SCHEMA
        or callback.get("status") != "passed"
    ):
        raise RuntimeError("resumable callback identity or execution status differs")
    qualification = callback.get("qualification_status")
    if qualification not in QUALIFICATION_STATUSES:
        raise RuntimeError("resumable callback qualification status differs")

    checkpoint_path = checkpoint_root / "checkpoint.json"
    if callback.get("checkpoint_manifest_sha256") != harness.sha256_file(
        checkpoint_path
    ):
        raise RuntimeError("checkpoint manifest hash differs")
    contract_sha256 = callback.get("contract_sha256")
    if not isinstance(contract_sha256, str):
        raise TypeError("resumable callback contract hash is absent")
    checkpoint = harness.verify_checkpoint(checkpoint_root, contract_sha256)

    metrics = callback.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError("resumable callback metrics are absent")
    completed = checkpoint["completed_games"]
    if metrics != harness.aggregate(checkpoint):
        raise RuntimeError("callback metrics differ from the verified checkpoint")

    attempted = callback.get("games_attempted_this_run")
    current_completed = callback.get("games_completed_this_run")
    if (
        not isinstance(attempted, list)
        or not isinstance(current_completed, list)
        or len(attempted) != len(set(attempted))
        or len(current_completed) != len(set(current_completed))
        or any(game not in harness.PUBLIC_GAMES for game in attempted)
        or current_completed != attempted[: len(current_completed)]
        or any(game not in completed for game in current_completed)
    ):
        raise RuntimeError("callback current-generation game inventory differs")
    predecessor_games = set(completed) - set(current_completed)
    expected_attempted = [
        game for game in harness.RUN_ORDER if game not in predecessor_games
    ][: len(attempted)]
    if attempted != expected_attempted:
        raise RuntimeError("callback current-generation run order differs")

    failure = callback.get("failure")
    all25_complete = callback.get("all25_complete")
    remaining = metrics["remaining_games"]
    if qualification == "complete":
        if (
            failure is not None
            or attempted != current_completed
            or all25_complete is not True
            or remaining
        ):
            raise RuntimeError("complete callback disposition is inconsistent")
    elif qualification == "partial_budget":
        if (
            failure is not None
            or attempted != current_completed
            or all25_complete is not False
            or not remaining
        ):
            raise RuntimeError("partial-budget callback disposition is inconsistent")
    else:
        if (
            not isinstance(failure, dict)
            or len(attempted) != len(current_completed) + 1
            or attempted[-1] != failure.get("game")
            or all25_complete is not False
            or not remaining
        ):
            raise RuntimeError("partial-failure callback disposition is inconsistent")
        checkpoint_failures = checkpoint.get("failures", [])
        matches = [
            record
            for record in checkpoint_failures
            if record.get("game") == failure.get("game")
        ]
        expected_failure = {
            key: matches[0].get(key) if matches else None
            for key in ("game", "returncode", "result_status")
        }
        if len(matches) != 1 or failure != expected_failure:
            raise RuntimeError("callback failure differs from manifested failure evidence")
    return callback, checkpoint


def _request_count_contract(
    callback: dict[str, Any],
    warmup_facts: dict[str, Any],
    post_callback_facts: dict[str, Any],
) -> tuple[int, int, str | None]:
    if (
        warmup_facts.get("chat_completion_accesses") != 1
        or warmup_facts.get("successful_chat_completion_accesses") != 1
        or warmup_facts.get("successful_freeze_gc_accesses") != 1
        or warmup_facts.get("ready_marker_count") != 1
    ):
        raise RuntimeError("outer warmup request contract differs")
    total = post_callback_facts.get("chat_completion_accesses")
    successful = post_callback_facts.get("successful_chat_completion_accesses")
    if (
        type(total) is not int
        or total < 1
        or successful != total
        or post_callback_facts.get("successful_freeze_gc_accesses") != 1
        or post_callback_facts.get("ready_marker_count") != 1
    ):
        raise RuntimeError("outer callback request-success contract differs")
    observed = total - 1
    minimum = int(
        callback["qualification_status"] == "complete"
        or bool(callback["games_completed_this_run"])
    )
    if observed < minimum:
        raise RuntimeError(
            "outer callback made fewer Solver requests than its disposition requires"
        )
    zero_disposition: str | None = None
    if observed == 0:
        if callback["qualification_status"] == "partial_failure":
            zero_disposition = "manifest_valid_pre_request_partial_failure"
        elif (
            callback["qualification_status"] == "partial_budget"
            and not callback["games_attempted_this_run"]
        ):
            zero_disposition = "manifest_valid_no_game_partial_budget"
        else:
            raise RuntimeError("zero Solver requests are not valid for this callback")
    return minimum, observed, zero_disposition


def validate_outer_callback(
    callback_path: Path,
    checkpoint_root: Path,
    *,
    warmup_facts: dict[str, Any],
    post_callback_facts: dict[str, Any],
) -> dict[str, Any]:
    """Return the schema-bound callback disposition accepted by a future outer join."""
    callback, _ = _validated_callback(callback_path, checkpoint_root)
    minimum, observed, zero_disposition = _request_count_contract(
        callback, warmup_facts, post_callback_facts
    )
    return {
        "schema": OUTER_CONTRACT_SCHEMA,
        "status": "passed",
        "qualification_status": callback["qualification_status"],
        "minimum_solver_requests": minimum,
        "observed_solver_requests": observed,
        "zero_request_disposition": zero_disposition,
        "checkpoint_verified": True,
        "lifecycle_requirement": LIFECYCLE_REQUIREMENT,
        "notebook_subprocess_requirement": NOTEBOOK_SUBPROCESS_REQUIREMENT,
    }
