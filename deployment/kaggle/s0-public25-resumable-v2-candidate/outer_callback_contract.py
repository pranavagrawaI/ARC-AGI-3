"""Validate a resumable public-25 callback against the outer server evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import resumable_public25 as harness

OUTER_CONTRACT_SCHEMA = "tycho-nooa-public25-outer-callback-contract-v2"
LIFECYCLE_REQUIREMENT = "accepted_r8_l3b_term_only_unchanged"
NOTEBOOK_SUBPROCESS_REQUIREMENT = "check_true"
QUALIFICATION_STATUSES = {"complete", "partial_budget", "partial_failure"}
STORAGE_COMPONENT_KEYS = {
    "storage_root",
    "predecessor",
    "checkpoint",
    "workspace",
    "scratch",
    "published_bundle",
    "failure_evidence",
}


def _read_regular_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is not one regular file: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{label} is not one JSON object")
    return value


def _expected_storage_events(
    attempted: list[str], completed: list[str], qualification: str
) -> list[tuple[str, str | None]]:
    events: list[tuple[str, str | None]] = [
        ("after_predecessor_transport_import", None),
        ("after_predecessor_copy", None),
    ]
    for game in attempted:
        events.extend(
            [
                ("pre_game_admission", game),
                ("post_game_prepublication", game),
            ]
        )
        if game in completed:
            events.extend(
                [
                    ("post_publication_pre_cleanup", game),
                    ("post_publication_cleanup", game),
                ]
            )
        elif qualification == "partial_failure":
            events.extend(
                [("pre_failure_copy", game), ("post_failure_copy", game)]
            )
    return events


def _validate_storage_ledger(
    callback: dict[str, Any],
    attempted: list[str],
    completed: list[str],
    qualification: str,
    expected_storage_projection: dict[str, int],
) -> dict[str, Any]:
    safety = callback.get("storage_safety")
    expected_safety_fields = {
        "ledger_schema",
        "ledger_path",
        "ledger_sha256",
        "ledger_record_count",
        "next_scratch_bound_bytes",
        "storage_publication_margin_bytes",
    }
    if not isinstance(safety, dict) or set(safety) != expected_safety_fields:
        raise RuntimeError("callback storage safety binding differs")
    if safety.get("ledger_schema") != harness.STORAGE_LEDGER_SCHEMA:
        raise RuntimeError("callback storage ledger schema differs")
    ledger_path_value = safety.get("ledger_path")
    if not isinstance(ledger_path_value, str) or not ledger_path_value:
        raise RuntimeError("callback storage ledger path differs")
    ledger_path = Path(ledger_path_value)
    if not ledger_path.is_absolute() or ledger_path.name != "storage-ledger.jsonl":
        raise RuntimeError("callback storage ledger path differs")
    if not ledger_path.is_file() or ledger_path.is_symlink():
        raise RuntimeError("callback storage ledger is not one regular file")
    if safety.get("ledger_sha256") != harness.sha256_file(ledger_path):
        raise RuntimeError("callback storage ledger hash differs")
    scratch_bound = safety.get("next_scratch_bound_bytes")
    publication_margin = safety.get("storage_publication_margin_bytes")
    if (
        type(scratch_bound) is not int
        or scratch_bound <= 0
        or type(publication_margin) is not int
        or publication_margin <= 0
    ):
        raise RuntimeError("callback storage byte bounds differ")
    if expected_storage_projection != {
        "next_scratch_bound_bytes": scratch_bound,
        "storage_publication_margin_bytes": publication_margin,
    }:
        raise RuntimeError("callback storage bounds differ from the frozen contract")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(ledger_path.read_text().splitlines(), 1):
        if not line.strip():
            raise RuntimeError(
                f"callback storage ledger contains a blank row: {line_number}"
            )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"callback storage ledger contains invalid JSON: {line_number}"
            ) from error
        if not isinstance(row, dict):
            raise TypeError("callback storage ledger row is not an object")
        rows.append(row)
    if safety.get("ledger_record_count") != len(rows):
        raise RuntimeError("callback storage ledger record count differs")
    expected_events = _expected_storage_events(attempted, completed, qualification)
    if [(row.get("event"), row.get("game")) for row in rows] != expected_events:
        raise RuntimeError("callback storage ledger event sequence differs")

    expected_row_fields = {
        "schema",
        "sequence",
        "captured_at",
        "event",
        "game",
        "storage_root",
        "filesystem",
        "components",
        "admission",
    }
    storage_root: str | None = None
    required_available = 2 * scratch_bound + publication_margin
    for sequence, row in enumerate(rows):
        if (
            set(row) != expected_row_fields
            or row.get("schema") != harness.STORAGE_LEDGER_SCHEMA
            or row.get("sequence") != sequence
            or not isinstance(row.get("captured_at"), str)
            or not row["captured_at"]
            or not isinstance(row.get("storage_root"), str)
            or not Path(row["storage_root"]).is_absolute()
        ):
            raise RuntimeError("callback storage ledger row identity differs")
        if storage_root is None:
            storage_root = row["storage_root"]
        elif row["storage_root"] != storage_root:
            raise RuntimeError("callback storage ledger root changes between rows")
        filesystem = row.get("filesystem")
        if (
            not isinstance(filesystem, dict)
            or set(filesystem)
            != {"fragment_bytes", "total_bytes", "free_bytes", "available_bytes"}
            or any(type(value) is not int or value < 0 for value in filesystem.values())
        ):
            raise RuntimeError("callback storage ledger filesystem facts differ")
        components = row.get("components")
        if not isinstance(components, dict) or set(components) != STORAGE_COMPONENT_KEYS:
            raise RuntimeError("callback storage ledger component inventory differs")
        for component in components.values():
            if (
                not isinstance(component, dict)
                or set(component) != {"path", "exists", "regular_file_bytes"}
                or (
                    component["path"] is not None
                    and not isinstance(component["path"], str)
                )
                or type(component["exists"]) is not bool
                or type(component["regular_file_bytes"]) is not int
                or component["regular_file_bytes"] < 0
            ):
                raise RuntimeError("callback storage ledger component facts differ")

        if row["event"] == "pre_game_admission":
            admission = row.get("admission")
            if admission != {
                "next_scratch_bound_bytes": scratch_bound,
                "storage_publication_margin_bytes": publication_margin,
                "required_available_bytes": required_available,
                "available_bytes": admission.get("available_bytes")
                if isinstance(admission, dict)
                else None,
                "passed": True,
            }:
                raise RuntimeError("callback storage admission facts differ")
            if (
                type(admission["available_bytes"]) is not int
                or admission["available_bytes"] < required_available
            ):
                raise RuntimeError("callback storage admission did not preserve its reserve")
        elif row.get("admission") is not None:
            raise RuntimeError("callback storage ledger has admission outside pre-game")

        scratch_exists = components["scratch"]["exists"]
        published_exists = components["published_bundle"]["exists"]
        failure_exists = components["failure_evidence"]["exists"]
        if row["event"] == "post_publication_pre_cleanup" and (
            not scratch_exists or not published_exists
        ):
            raise RuntimeError("callback storage publication evidence differs")
        if row["event"] == "post_publication_cleanup" and (
            scratch_exists or not published_exists
        ):
            raise RuntimeError("callback storage cleanup evidence differs")
        if row["event"] == "pre_failure_copy" and (
            not scratch_exists or failure_exists
        ):
            raise RuntimeError("callback storage pre-failure evidence differs")
        if row["event"] == "post_failure_copy" and (
            not scratch_exists or not failure_exists
        ):
            raise RuntimeError("callback storage failure-copy evidence differs")
    return {
        "ledger_sha256": safety["ledger_sha256"],
        "ledger_record_count": len(rows),
        "next_scratch_bound_bytes": scratch_bound,
        "storage_publication_margin_bytes": publication_margin,
    }


def _validated_callback(
    callback_path: Path,
    checkpoint_root: Path,
    expected_storage_projection: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    storage = _validate_storage_ledger(
        callback,
        attempted,
        current_completed,
        qualification,
        expected_storage_projection,
    )
    return callback, checkpoint, storage


def _request_count_contract(
    callback: dict[str, Any],
    checkpoint: dict[str, Any],
    checkpoint_root: Path,
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
    completed = callback["games_completed_this_run"]
    expected = sum(checkpoint["games"][game]["actor_calls"] for game in completed)
    if callback["qualification_status"] == "partial_failure":
        failure_game = callback["failure"]["game"]
        failure_root = checkpoint_root / "failures" / failure_game
        trace_paths = sorted(failure_root.rglob("llm-trace.jsonl"))
        for path in trace_paths:
            if not path.is_file() or path.is_symlink():
                raise RuntimeError("failed-game model trace is not one regular file")
            for line_number, line in enumerate(path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"failed-game model trace is invalid: {path}:{line_number}"
                    ) from error
                if (
                    not isinstance(record, dict)
                    or record.get("source") != "actor"
                    or record.get("phase") != "decision"
                ):
                    raise RuntimeError("failed-game model trace contains a hidden lane")
                expected += 1
        maximum = harness.FIRST_LEVEL_ACTION_CAPS[failure_game] + 5
        failed_calls = expected - sum(
            checkpoint["games"][game]["actor_calls"] for game in completed
        )
        if failed_calls > maximum:
            raise RuntimeError("failed-game Actor calls exceed the frozen envelope")
    if observed != expected:
        raise RuntimeError(
            "outer callback Solver request count differs from manifested Actor calls: "
            f"observed={observed}, expected={expected}"
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
    return expected, observed, zero_disposition


def validate_outer_callback(
    callback_path: Path,
    checkpoint_root: Path,
    *,
    warmup_facts: dict[str, Any],
    post_callback_facts: dict[str, Any],
    expected_storage_projection: dict[str, int],
) -> dict[str, Any]:
    """Return the schema-bound callback disposition accepted by a future outer join."""
    callback, checkpoint, storage = _validated_callback(
        callback_path,
        checkpoint_root,
        expected_storage_projection,
    )
    expected, observed, zero_disposition = _request_count_contract(
        callback, checkpoint, checkpoint_root, warmup_facts, post_callback_facts
    )
    return {
        "schema": OUTER_CONTRACT_SCHEMA,
        "status": "passed",
        "qualification_status": callback["qualification_status"],
        "expected_solver_requests": expected,
        "observed_solver_requests": observed,
        "whole_model_request_count_verified": True,
        "zero_request_disposition": zero_disposition,
        "checkpoint_verified": True,
        "storage_ledger_verified": True,
        "storage_ledger_sha256": storage["ledger_sha256"],
        "storage_ledger_record_count": storage["ledger_record_count"],
        "lifecycle_requirement": LIFECYCLE_REQUIREMENT,
        "notebook_subprocess_requirement": NOTEBOOK_SUBPROCESS_REQUIREMENT,
    }
