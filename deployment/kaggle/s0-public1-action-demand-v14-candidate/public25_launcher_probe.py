"""Run the exact official launcher over the 25 mounted public environments."""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

PUBLIC_GAMES = [
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59",
    "lf52", "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26",
    "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33",
    "wa30",
]
requested_games = os.getenv("TYCHO_PUBLIC_GAME_IDS")
EXPECTED_GAMES = (
    requested_games.split(",") if requested_games else PUBLIC_GAMES
)
if (
    not EXPECTED_GAMES
    or len(EXPECTED_GAMES) != len(set(EXPECTED_GAMES))
    or any(game not in PUBLIC_GAMES for game in EXPECTED_GAMES)
):
    raise RuntimeError("TYCHO_PUBLIC_GAME_IDS must name a unique public-game subset")
PUBLIC_MAX_ACTIONS = int(os.getenv("TYCHO_KAGGLE_MAX_ACTIONS", "80"))
LIST_HOST = "127.0.0.1"
LIST_PORT = 18_001
WORKING_ROOT = Path(os.getenv("TYCHO_PUBLIC25_WORKING_ROOT", "/kaggle/working"))
LEVEL_BOUNDARY_POLICY = "deterministic_level_checkpoint"


class GameListHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/api/games":
            self.send_error(404)
            return
        body = json.dumps([{"game_id": game} for game in EXPECTED_GAMES]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=int, default=30_000)
    return parser.parse_args()


def official_launcher_environment(
    *,
    base_url: str,
    model: str,
    workspace_root: Path,
    worker_python: Path,
) -> dict[str, str]:
    if not workspace_root.is_absolute():
        raise ValueError("Solver evidence workspace root must be absolute")
    return {
        **os.environ,
        "ARC_LLM_API_KEY": "local-model-no-key",
        "ARC_LLM_BASE_URL": base_url,
        "ARC_LLM_MODEL": model,
        "ARC_LLM_PROVIDER_ENDPOINT": "local",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "MPLBACKEND": "agg",
        "TYCHO_KAGGLE_HARDENED_HOST": "1",
        "TYCHO_KAGGLE_WORKSPACE_ROOT": str(workspace_root),
        "TYCHO_NOOA_WORKER_PYTHON": str(worker_python),
        "TYCHO_SANDBOX_RUNTIME": "host",
    }


def log_facts(path: Path) -> dict[str, object]:
    value = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
        "tail": value[-8_192:].decode("utf-8", errors="replace"),
    }


def parse_scorecard(log_path: Path) -> dict[str, Any]:
    marker = "--- FINAL SCORECARD REPORT ---"
    text = log_path.read_text(errors="replace")
    if marker not in text:
        raise RuntimeError("official launcher log has no final scorecard marker")
    remainder = text.rsplit(marker, 1)[1]
    start = remainder.find("{")
    if start < 0:
        raise RuntimeError("final scorecard marker has no JSON object")
    scorecard, _ = json.JSONDecoder().raw_decode(remainder[start:])
    if not isinstance(scorecard, dict):
        raise TypeError("final scorecard is not a JSON object")
    environments = scorecard.get("environments", [])
    if not isinstance(environments, list):
        raise TypeError("final scorecard environments is not a list")
    environment_ids = sorted(environment.get("id") for environment in environments)
    if environment_ids and environment_ids != EXPECTED_GAMES:
        raise RuntimeError(
            f"scorecard environment IDs differ: {environment_ids}"
        )
    return scorecard


def recordings_dir_from_env(dotenv_path: Path) -> Path:
    for raw_line in dotenv_path.read_text().splitlines():
        key, separator, value = raw_line.partition("=")
        if separator and key.strip() == "RECORDINGS_DIR":
            path = Path(value.strip())
            if not path.is_absolute():
                raise RuntimeError("RECORDINGS_DIR is not absolute")
            return path
    raise RuntimeError("official launcher .env has no RECORDINGS_DIR")


def summarize_recordings(recordings_dir: Path) -> dict[str, Any]:
    if not recordings_dir.is_dir():
        raise RuntimeError(f"recordings directory is absent: {recordings_dir}")
    paths = sorted(recordings_dir.glob("*.recording.jsonl"))
    by_game: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"recording is not a regular file: {path}")
        game = path.name.split(".", 1)[0]
        if game not in EXPECTED_GAMES:
            raise RuntimeError(f"unexpected public recording: {path.name}")
        if game in by_game:
            raise RuntimeError(f"multiple recordings found for public game: {game}")
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid recording JSON: {path.name}:{line_number}: {error}"
                ) from error
            data = event.get("data") if isinstance(event, dict) else None
            if not isinstance(data, dict) or not isinstance(data.get("game_id"), str):
                raise TypeError(
                    f"recording row lacks frame data: {path.name}:{line_number}"
                )
            if data["game_id"].split("-", 1)[0] != game:
                raise RuntimeError(
                    f"recording game identity differs: {path.name}:{line_number}"
                )
            rows.append(data)
        if not rows:
            raise RuntimeError(f"recording has no frames: {path.name}")

        final = rows[-1]
        final_state = final.get("state")
        frame_count = len(rows)
        if final_state not in {"NOT_FINISHED", "GAME_OVER", "WIN"}:
            raise RuntimeError(f"recording final state differs for {game}: {final_state}")
        if not 1 <= frame_count <= 2 * PUBLIC_MAX_ACTIONS + 1:
            raise RuntimeError(
                f"recording action count exceeds the official safety envelope: "
                f"{game} has {frame_count} frames"
            )
        action_ids = [
            data.get("action_input", {}).get("id")
            for data in rows
            if isinstance(data.get("action_input"), dict)
        ]
        level_values = [
            value
            for data in rows
            if isinstance((value := data.get("levels_completed")), int)
        ]
        if len(action_ids) != frame_count or len(level_values) != frame_count:
            raise RuntimeError(f"recording frame fields are incomplete: {game}")
        boundary_events: list[dict[str, Any]] = []
        # Offline public recordings may begin with the first Solver action
        # rather than a bootstrap RESET.  The Solver always starts on level 0,
        # so a completion or WIN in row zero belongs to that known level.
        first = rows[0]
        first_level_complete = first["levels_completed"] > 0
        first_won = first.get("state") == "WIN"
        if first_level_complete or first_won:
            boundary_events.append(
                {
                    "result_index": 0,
                    "level": 0,
                    "level_complete": first_level_complete,
                    "win": first_won,
                }
            )
        for result_index, (before, after) in enumerate(
            pairwise(rows),
            1,
        ):
            before_level = before["levels_completed"]
            after_level = after["levels_completed"]
            level_complete = after_level > before_level
            won = after.get("state") == "WIN"
            if level_complete or won:
                boundary_events.append(
                    {
                        "result_index": result_index,
                        "level": before_level,
                        "level_complete": level_complete,
                        "win": won,
                    }
                )
        by_game[game] = {
            "recording": log_facts(path),
            "versioned_game_id": final["game_id"],
            "recorded_action_results": frame_count,
            "action_ids": {
                str(action_id): action_ids.count(action_id)
                for action_id in sorted(set(action_ids), key=lambda value: str(value))
            },
            "final_state": final_state,
            "final_levels_completed": final.get("levels_completed"),
            "max_levels_completed": max(level_values),
            "win_levels": final.get("win_levels"),
            "full_reset_results": sum(bool(data.get("full_reset")) for data in rows),
            "boundary_levels": sorted(
                {event["level"] for event in boundary_events}
            ),
            "boundary_events": boundary_events,
        }

    actual_games = sorted(by_game)
    if actual_games != EXPECTED_GAMES:
        raise RuntimeError(f"recording game IDs differ: {actual_games}")
    return {
        "schema": "tycho-nooa-public-progress-v1",
        "evidence_kind": "official_agent_recordings_not_official_score",
        "game_count": len(by_game),
        "total_recorded_action_results": sum(
            game["recorded_action_results"] for game in by_game.values()
        ),
        "total_final_levels_completed": sum(
            game["final_levels_completed"] for game in by_game.values()
        ),
        "total_max_levels_completed": sum(
            game["max_levels_completed"] for game in by_game.values()
        ),
        "won_games": sum(game["final_state"] == "WIN" for game in by_game.values()),
        "games": by_game,
    }


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is not one regular file: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"invalid {label} JSON: {path}:{line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise TypeError(f"{label} row is not an object: {path}:{line_number}")
        records.append(record)
    if not records:
        raise RuntimeError(f"{label} has no records: {path}")
    return records


def require_workspace_regular(
    workspace: Path,
    relative: Path,
    label: str,
) -> Path:
    """Require one regular, non-symlink file below the exact game workspace."""
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"{label} path escapes the game workspace: {relative}")
    candidate = workspace / relative
    cursor = workspace
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(f"{label} traverses a symlink: {cursor}")
    if not candidate.is_file():
        raise RuntimeError(f"{label} is not one regular file: {candidate}")
    try:
        candidate.resolve(strict=True).relative_to(workspace.resolve(strict=True))
    except ValueError as error:
        raise RuntimeError(f"{label} resolves outside the game workspace") from error
    return candidate


def summarize_level_boundaries(
    workspace: Path,
    game: str,
    expected_levels: list[int],
) -> dict[str, Any]:
    """Verify every public level boundary has the deterministic V7 checkpoint."""
    if expected_levels != sorted(set(expected_levels)) or any(
        not isinstance(level, int) or level < 0 for level in expected_levels
    ):
        raise RuntimeError(f"public boundary levels differ for {game}: {expected_levels}")
    actual_paths = list(workspace.glob("level_*/level_boundary_checkpoint.json"))
    if any(path.is_symlink() or not path.is_file() for path in actual_paths):
        raise RuntimeError(f"level boundary checkpoint inventory is unsafe for {game}")
    actual_by_level: dict[int, Path] = {}
    for path in actual_paths:
        parent = path.parent.name
        prefix = "level_"
        if not parent.startswith(prefix) or not parent[len(prefix):].isdigit():
            raise RuntimeError(f"invalid level boundary path for {game}: {path}")
        level = int(parent[len(prefix):])
        if level in actual_by_level:
            raise RuntimeError(f"duplicate level boundary for {game} level {level}")
        actual_by_level[level] = path
    if sorted(actual_by_level) != expected_levels:
        raise RuntimeError(
            f"level boundary checkpoint inventory differs for {game}: "
            f"expected {expected_levels}, got {sorted(actual_by_level)}"
        )

    checkpoints: list[dict[str, Any]] = []
    for level in expected_levels:
        checkpoint_relative = Path(
            f"level_{level}/level_boundary_checkpoint.json"
        )
        checkpoint_path = require_workspace_regular(
            workspace,
            checkpoint_relative,
            f"level boundary checkpoint for {game} level {level}",
        )
        expected_note = f"notes/level_{level}_insights.md"
        expected = {
            "schema_version": 1,
            "source": "tycho_nooa host treatment",
            "treatment": "single_decision_discipline",
            "policy": LEVEL_BOUNDARY_POLICY,
            "model_scribe_invoked": False,
            "level": level,
            "carry_forward_note": expected_note,
        }
        observed = json.loads(checkpoint_path.read_text())
        if observed != expected:
            raise RuntimeError(
                f"level boundary checkpoint differs for {game} level {level}: "
                f"{observed}"
            )
        note_path = require_workspace_regular(
            workspace,
            Path(expected_note),
            f"level carry-forward note for {game} level {level}",
        )
        checkpoints.append(
            {
                "level": level,
                "checkpoint": log_facts(checkpoint_path),
                "carry_forward_note": log_facts(note_path),
            }
        )
    return {
        "policy": LEVEL_BOUNDARY_POLICY,
        "boundary_count": len(checkpoints),
        "levels": expected_levels,
        "checkpoints": checkpoints,
    }


def verify_action_horizon_request(
    *,
    game: str,
    sequence: int,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Require the exact host action horizon in messages and action-tool wording."""
    if not 0 <= sequence < PUBLIC_MAX_ACTIONS:
        raise RuntimeError(
            f"action horizon sequence differs for {game}: {sequence}"
        )
    actions_remaining = PUBLIC_MAX_ACTIONS - sequence
    final_permitted_action = actions_remaining == 1
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise RuntimeError(f"action horizon messages differ for {game} sequence {sequence}")
    model_view = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    expected_horizon = (
        f"Action horizon: {sequence} of {PUBLIC_MAX_ACTIONS} permitted environment "
        f"actions have been used; {actions_remaining} remain including this decision."
    )
    expected_fields = (
        f"'sequence': {sequence}",
        f"'action_limit': {PUBLIC_MAX_ACTIONS}",
        f"'actions_used': {sequence}",
        f"'actions_remaining': {actions_remaining}",
        f"'final_permitted_action': {final_permitted_action}",
    )
    final_notice = (
        "This is the final permitted environment action. Its result will be recorded, "
        "but no next model decision is guaranteed."
    )
    if (
        expected_horizon not in model_view
        or any(field not in model_view for field in expected_fields)
        or (final_notice in model_view) is not final_permitted_action
    ):
        raise RuntimeError(
            f"action horizon message differs for {game} sequence {sequence}"
        )

    tools = body.get("tools")
    if not isinstance(tools, list):
        raise RuntimeError(f"action horizon tools differ for {game} sequence {sequence}")
    action_tools = {
        tool.get("name"): tool
        for tool in tools
        if tool.get("name") in {"take_action", "take_actions"}
    }
    if set(action_tools) != {"take_actions"}:
        raise RuntimeError(
            f"action horizon tool inventory differs for {game} sequence {sequence}"
        )
    batch_cap = min(4, actions_remaining)
    batch_tool = action_tools["take_actions"]
    batch_description = batch_tool.get("description")
    batch_parameters = batch_tool.get("parameters")
    try:
        schema_min = batch_parameters["properties"]["actions"]["minItems"]
        schema_cap = batch_parameters["properties"]["actions"]["maxItems"]
    except (KeyError, TypeError):
        schema_min = None
        schema_cap = None
    if (
        not isinstance(batch_description, str)
        or "The only action-commitment tool" not in batch_description
        or f"Commit 1-{batch_cap} ordered environment actions" not in batch_description
        or "observes its public result" not in batch_description
        or schema_min != 1
        or schema_cap != batch_cap
    ):
        raise RuntimeError(
            f"action horizon take_actions contract differs for {game} sequence {sequence}"
        )
    return {
        "action_limit": PUBLIC_MAX_ACTIONS,
        "actions_used": sequence,
        "actions_remaining": actions_remaining,
        "final_permitted_action": final_permitted_action,
        "action_horizon_verified": True,
        "batch_cap": batch_cap,
        "batch_contract_verified": True,
    }


def summarize_model_evidence(
    session: Path,
    game: str,
    expected_actions: int,
) -> dict[str, Any]:
    trace_path = session / "llm-trace.jsonl"
    trace = read_jsonl(trace_path, "Solver LLM trace")
    expected_call_indices = list(range(1, len(trace) + 1))
    if [record.get("call_index") for record in trace] != expected_call_indices:
        raise RuntimeError(f"Solver model call indices differ for {game}")

    by_sequence: dict[int, list[dict[str, Any]]] = {}
    for record in trace:
        if record.get("source") != "actor":
            raise RuntimeError(f"non-Actor model call found for {game}: {record}")
        if record.get("phase") != "decision":
            raise RuntimeError(f"non-decision model call found for {game}: {record}")
        sequence = record.get("sequence")
        if not isinstance(sequence, int) or not 0 <= sequence < expected_actions:
            raise RuntimeError(f"model call sequence differs for {game}: {record}")
        by_sequence.setdefault(sequence, []).append(record)

    if 0 not in by_sequence:
        raise RuntimeError(f"initial model call sequence is absent for {game}")
    initial_limit = int(os.environ["ARC_LLM_MAX_ACTOR_CALLS"])
    successor_limit = int(os.environ["ARC_LLM_MAX_ACTOR_CALLS_AFTER_INITIAL"])
    if not 1 <= len(by_sequence[0]) <= initial_limit:
        raise RuntimeError(f"initial Actor call count differs for {game}")
    for sequence, records in by_sequence.items():
        if sequence > 0 and len(records) != successor_limit:
            raise RuntimeError(
                f"successor Actor call count differs for {game} sequence {sequence}"
            )
    for sequence, records in by_sequence.items():
        action_calls = sum(
            record.get("tool_calls", []).count("take_actions")
            for record in records
        )
        if action_calls != 1:
            raise RuntimeError(
                f"Actor action-call count differs for {game} sequence {sequence}"
            )

    calls_root = session / "decision-checkpoints" / "calls"
    if not calls_root.is_dir() or calls_root.is_symlink():
        raise RuntimeError(f"decision checkpoints are absent for {game}")
    call_dirs = sorted(calls_root.iterdir())
    expected_names = [f"{index:06d}" for index in expected_call_indices]
    if (
        [path.name for path in call_dirs] != expected_names
        or any(not path.is_dir() or path.is_symlink() for path in call_dirs)
    ):
        raise RuntimeError(f"decision checkpoint inventory differs for {game}")

    initial_tokens = int(os.environ["ARC_LLM_MAX_TOKENS"])
    successor_tokens = int(os.environ["ARC_LLM_MAX_TOKENS_AFTER_INITIAL"])
    action_demand_tokens = int(os.environ["ARC_LLM_ACTION_DEMAND_MAX_TOKENS"])
    expected_action_demand_extra_body = {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    context_window = int(os.environ["ARC_LLM_CONTEXT_WINDOW"])
    request_facts: list[dict[str, Any]] = []
    for record, call_dir in zip(trace, call_dirs, strict=True):
        required = [
            call_dir / "request.json",
            call_dir / "response.json",
            call_dir / "workspace.json",
        ]
        if any(not path.is_file() or path.is_symlink() for path in required):
            raise RuntimeError(f"decision checkpoint is incomplete for {game}: {call_dir}")
        request = json.loads(required[0].read_text())
        sequence = record["sequence"]
        body = request.get("request_body", {})
        parameters = body.get("parameters", {})
        tools = body.get("tools", [])
        tool_names = [tool.get("name") for tool in tools]
        client_config = request.get("replay_context", {}).get("client_config", {})
        retries = client_config.get("num_retries")
        if (
            request.get("call_index") != record["call_index"]
            or request.get("source") != "actor"
            or request.get("decision_point", {}).get("sequence") != sequence
            or client_config.get("context_window") != context_window
            or retries != 0
        ):
            raise RuntimeError(f"decision checkpoint identity differs for {game}")
        action_horizon = verify_action_horizon_request(
            game=game,
            sequence=sequence,
            body=body,
        )
        tool_choice = parameters.get("tool_choice")
        action_tool_names = ["take_actions"]
        is_action_demand = tool_choice == "required"
        expected_tokens = (
            action_demand_tokens
            if is_action_demand
            else initial_tokens if sequence == 0 else successor_tokens
        )
        if parameters.get("max_tokens") != expected_tokens:
            raise RuntimeError(f"request token ceiling differs for {game} sequence {sequence}")
        extra_body = parameters.get("extra_body")
        if is_action_demand:
            if extra_body != expected_action_demand_extra_body:
                raise RuntimeError(
                    f"action-demand thinking policy differs for {game} sequence {sequence}"
                )
        elif extra_body is not None:
            raise RuntimeError(
                f"exploratory request unexpectedly disables thinking for {game} "
                f"sequence {sequence}"
            )
        if sequence > 0:
            if tool_choice != "required" or tool_names != action_tool_names:
                raise RuntimeError(
                    f"successor action obligation differs for {game} sequence {sequence}"
                )
        elif tool_choice == "required":
            if tool_names != action_tool_names or record is not by_sequence[0][-1]:
                raise RuntimeError(f"initial final-action request differs for {game}")
        elif tool_choice != "auto" or not {
            "ls",
            "take_actions",
        }.issubset(tool_names):
            raise RuntimeError(f"initial exploratory request differs for {game}")
        request_facts.append(
            {
                "call_index": record["call_index"],
                "sequence": sequence,
                "max_tokens": parameters["max_tokens"],
                "context_window": context_window,
                "tool_choice": tool_choice,
                "tool_names": tool_names,
                "extra_body": extra_body,
                "action_demand": is_action_demand,
                **action_horizon,
            }
        )

    stdout_path = session / "worker-stdout.log"
    stderr_path = session / "worker-stderr.log"
    for path, label in (
        (stdout_path, "worker stdout"),
        (stderr_path, "worker stderr"),
    ):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"{label} is incomplete for {game}: {path}")
    if stderr_path.read_bytes():
        raise RuntimeError(f"worker stderr is nonempty for {game}")
    return {
        "actor_call_count": len(trace),
        "actor_calls_by_sequence": {
            str(sequence): len(records)
            for sequence, records in sorted(by_sequence.items())
        },
        "model_decision_sequences": sorted(by_sequence),
        "zero_call_action_sequences": sorted(set(range(expected_actions)) - set(by_sequence)),
        "llm_trace": log_facts(trace_path),
        "request_facts": request_facts,
        "worker_stdout": log_facts(stdout_path),
        "worker_stderr": log_facts(stderr_path),
    }


def summarize_environment_actions(
    session: Path,
    game: str,
    expected_actions: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconstruct every host action from guarded batch releases."""
    tool_trace = read_jsonl(session / "tool-trace.jsonl", "Solver tool trace")
    actions_by_sequence: dict[int, dict[str, Any]] = {}
    completed_action_tools = [
        event.get("name")
        for event in tool_trace
        if event.get("status") == "complete"
        and event.get("name") in {"take_action", "take_actions"}
    ]
    if any(name != "take_actions" for name in completed_action_tools):
        raise RuntimeError(f"removed action primitive executed for {game}")

    batch_path = session / "action-batch-trace.jsonl"
    batch_trace = (
        read_jsonl(batch_path, "Solver action-batch trace")
        if batch_path.is_file() and not batch_path.is_symlink()
        else []
    )
    for event in batch_trace:
        if event.get("event") != "released":
            continue
        sequence = event.get("sequence")
        action = event.get("action")
        if (
            not isinstance(sequence, int)
            or not isinstance(action, dict)
            or sequence in actions_by_sequence
        ):
            raise RuntimeError(f"batch-release sequence differs for {game}: {event}")
        actions_by_sequence[sequence] = {
            "sequence": sequence,
            "source": "take_actions",
            "batch_id": event.get("batch_id"),
            "batch_offset": event.get("offset"),
            "validated_arguments": action,
        }

    observed_actions = len(actions_by_sequence)
    required_actions = observed_actions if expected_actions is None else expected_actions
    if sorted(actions_by_sequence) != list(range(required_actions)):
        raise RuntimeError(
            f"Solver action sequences differ for {game}: {sorted(actions_by_sequence)}"
        )
    submissions = [event for event in batch_trace if event.get("event") == "submitted"]
    releases = [event for event in batch_trace if event.get("event") == "released"]
    invalidations = [event for event in batch_trace if event.get("event") == "invalidated"]
    avoided = sum(
        isinstance(event.get("offset"), int) and event["offset"] > 0
        for event in releases
    )
    batch_summary = {
        "trace": log_facts(batch_path) if batch_trace else None,
        "submission_count": len(submissions),
        "submitted_action_count": sum(
            int(event.get("batch_length", 0)) for event in submissions
        ),
        "release_count": len(releases),
        "invalidation_count": len(invalidations),
        "invalidation_reasons": [event.get("reason") for event in invalidations],
        "model_calls_avoided": avoided,
        "uptake": bool(submissions),
    }
    return [actions_by_sequence[index] for index in range(required_actions)], batch_summary


def summarize_solver_sessions(
    workspace_root: Path,
    expected_action_counts: dict[str, int] | None = None,
    terminal_states: dict[str, str] | None = None,
    expected_boundary_levels: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    if not workspace_root.is_dir() or workspace_root.is_symlink():
        raise RuntimeError(
            f"Solver evidence workspace root is not an exact directory: {workspace_root}"
        )
    if expected_action_counts is not None and terminal_states is not None:
        raise RuntimeError("choose expected action counts or terminal states, not both")
    if expected_action_counts is None and terminal_states is None:
        expected_action_counts = {
            game: PUBLIC_MAX_ACTIONS for game in EXPECTED_GAMES
        }
    if expected_action_counts is not None and (
        set(expected_action_counts) != set(EXPECTED_GAMES)
        or any(
            not isinstance(count, int) or not 1 <= count <= PUBLIC_MAX_ACTIONS
            for count in expected_action_counts.values()
        )
    ):
        raise RuntimeError("expected Solver action counts differ from the public cap")
    if terminal_states is not None and (
        set(terminal_states) != set(EXPECTED_GAMES)
        or any(
            state not in {"NOT_FINISHED", "GAME_OVER", "WIN"}
            for state in terminal_states.values()
        )
    ):
        raise RuntimeError("terminal state inventory differs from the public games")
    if expected_boundary_levels is None:
        expected_boundary_levels = {game: [] for game in EXPECTED_GAMES}
    if set(expected_boundary_levels) != set(EXPECTED_GAMES):
        raise RuntimeError("public boundary inventory differs from the public games")
    expected_metadata = {
        "agent": "tycho_nooa.TychoNooaAgent",
        "client_kind": "live",
        "decision_envelope": {
            "force_action_tool": os.environ.get("ARC_LLM_FORCE_ACTION_TOOL") == "1",
            "max_actor_calls": int(os.environ["ARC_LLM_MAX_ACTOR_CALLS"]),
            "max_actor_calls_after_initial": int(
                os.environ["ARC_LLM_MAX_ACTOR_CALLS_AFTER_INITIAL"]
            ),
            "max_output_tokens_per_call": int(os.environ["ARC_LLM_MAX_TOKENS"]),
            "max_output_tokens_per_call_after_initial": int(
                os.environ["ARC_LLM_MAX_TOKENS_AFTER_INITIAL"]
            ),
            "model_retries": int(os.environ["ARC_LLM_RETRIES"]),
            "model_timeout_seconds": float(os.environ["ARC_LLM_TIMEOUT_SECONDS"]),
            "max_actions_per_batch": 4,
        },
        "mode": os.environ["TYCHO_MODE"],
        "sandbox_runtime": "host",
        "treatment": os.environ["TYCHO_NOOA_TREATMENT"],
        "level_boundary_policy": LEVEL_BOUNDARY_POLICY,
    }
    sessions: dict[str, dict[str, Any]] = {}
    realized_action_counts: dict[str, int] = {}
    for game in EXPECTED_GAMES:
        matches = [
            path
            for path in workspace_root.iterdir()
            if (
                path.is_dir()
                and not path.is_symlink()
                and (path / "workspace" / game).is_dir()
                and not (path / "workspace" / game).is_symlink()
            )
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one Solver session for {game}, found {matches}")
        session = matches[0]
        metadata_path = session / "worker-metadata.json"
        tool_trace_path = session / "tool-trace.jsonl"
        if (
            not metadata_path.is_file()
            or metadata_path.is_symlink()
            or not tool_trace_path.is_file()
            or tool_trace_path.is_symlink()
        ):
            raise RuntimeError(f"Solver evidence is incomplete for {game}")
        metadata = json.loads(metadata_path.read_text())
        expected_workspace = session / "workspace" / game
        if metadata.get("workspace") != str(expected_workspace):
            raise RuntimeError(
                f"Solver workspace identity differs for {game}: "
                f"{metadata.get('workspace')!r} != {str(expected_workspace)!r}"
            )
        observed_metadata = {
            key: metadata.get(key)
            for key in expected_metadata
        }
        if observed_metadata != expected_metadata:
            raise RuntimeError(
                f"Solver decision envelope differs for {game}: {observed_metadata}"
            )
        expected_actions = (
            expected_action_counts[game]
            if expected_action_counts is not None
            else None
        )
        actions, action_batches = summarize_environment_actions(
            session,
            game,
            expected_actions,
        )
        expected_actions = len(actions) if expected_actions is None else expected_actions
        if terminal_states is not None:
            terminal_state = terminal_states[game]
            if terminal_state == "WIN":
                if not 1 <= expected_actions <= PUBLIC_MAX_ACTIONS:
                    raise RuntimeError(f"terminal Solver action count differs for {game}")
            elif expected_actions != PUBLIC_MAX_ACTIONS:
                raise RuntimeError(
                    f"non-winning Solver session ended before its cap for {game}"
                )
        if len(actions) != expected_actions:
            raise RuntimeError(
                f"Solver action count differs for {game}: "
                f"expected {expected_actions}, got {len(actions)}"
            )
        model_evidence = summarize_model_evidence(session, game, expected_actions)
        batch_sequences = {
            action["sequence"]
            for action in actions
            if action["source"] == "take_actions" and action["batch_offset"] > 0
        }
        if set(model_evidence["zero_call_action_sequences"]) != batch_sequences:
            raise RuntimeError(f"zero-call batch continuation evidence differs for {game}")
        level_boundaries = summarize_level_boundaries(
            expected_workspace,
            game,
            expected_boundary_levels[game],
        )
        realized_action_counts[game] = expected_actions
        sessions[game] = {
            "session": str(session),
            "worker_metadata": metadata,
            "actions": actions,
            "action_batches": action_batches,
            "tool_trace": log_facts(tool_trace_path),
            "model_evidence": model_evidence,
            "level_boundaries": level_boundaries,
        }
    summary = {
        "schema": "tycho-nooa-public-solver-sessions-v1",
        "game_count": len(sessions),
        "total_solver_actions": sum(len(value["actions"]) for value in sessions.values()),
        "total_actor_calls": sum(
            value["model_evidence"]["actor_call_count"]
            for value in sessions.values()
        ),
        "total_batch_submissions": sum(
            value["action_batches"]["submission_count"] for value in sessions.values()
        ),
        "total_model_calls_avoided": sum(
            value["action_batches"]["model_calls_avoided"] for value in sessions.values()
        ),
        "games": sessions,
    }
    max_calls = sum(
        int(os.environ["ARC_LLM_MAX_ACTOR_CALLS"])
        + (expected_actions - 1)
        * int(os.environ["ARC_LLM_MAX_ACTOR_CALLS_AFTER_INITIAL"])
        for expected_actions in realized_action_counts.values()
    )
    if summary["total_actor_calls"] > max_calls:
        raise RuntimeError("aggregate Actor call count exceeds the frozen cap")
    return summary


def contain_group(process: subprocess.Popen[bytes], *, timed_out: bool) -> bool:
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + (30 if timed_out else 10)
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    os.killpg(process.pid, signal.SIGKILL)
    return True


def main() -> int:
    args = parse_args()
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        raise RuntimeError("public-25 qualification refuses competition rerun mode")
    if args.timeout_s <= 60:
        raise ValueError("public-25 launcher timeout must exceed 60 seconds")

    official_root = WORKING_ROOT / "ARC-AGI-3-Agents"
    runner = WORKING_ROOT / "run_official_agent.py"
    main_path = official_root / "main.py"
    for required in (runner, main_path, official_root / "local_solver.py"):
        if not required.is_file():
            raise FileNotFoundError(f"prepared launcher file is absent: {required}")

    args.workspace_root.mkdir(parents=True, exist_ok=True)
    environment = official_launcher_environment(
        base_url=args.base_url,
        model=args.model,
        workspace_root=args.workspace_root,
        worker_python=Path(sys.executable),
    )
    command = [
        sys.executable,
        str(runner),
        str(main_path),
        "--agent",
        "myagent",
    ]
    stdout_path = args.output.with_name("public25-launcher.stdout.log")
    stderr_path = args.output.with_name("public25-launcher.stderr.log")
    list_server = http.server.ThreadingHTTPServer(
        (LIST_HOST, LIST_PORT), GameListHandler
    )
    list_thread = threading.Thread(target=list_server.serve_forever, daemon=True)
    list_thread.start()

    started_at = datetime.now(UTC).isoformat()
    timed_out = False
    residual_group_term_required = False
    returncode = 1
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                command,
                cwd=official_root,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=args.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                residual_group_term_required = contain_group(process, timed_out=True)
                try:
                    returncode = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    returncode = 124
            else:
                residual_group_term_required = contain_group(process, timed_out=False)
    finally:
        list_server.shutdown()
        list_server.server_close()
        list_thread.join(timeout=5)

    effective_returncode = (
        124
        if timed_out
        else 71
        if residual_group_term_required
        else returncode
    )
    scorecard: dict[str, Any] | None = None
    scorecard_error: str | None = None
    scorecard_status: str | None = None
    try:
        scorecard = parse_scorecard(official_root / "logs.log")
        scorecard_status = (
            "available"
            if scorecard.get("environments")
            else "unavailable_due_local_initial_reset_contract"
        )
    except Exception as error:  # noqa: BLE001 - preserve qualification failure
        scorecard_error = f"{type(error).__name__}: {error}"
        if effective_returncode == 0:
            effective_returncode = 72

    public_progress: dict[str, Any] | None = None
    public_progress_error: str | None = None
    try:
        recordings_dir = recordings_dir_from_env(official_root / ".env")
        public_progress = summarize_recordings(recordings_dir)
    except Exception as error:  # noqa: BLE001 - preserve qualification failure
        public_progress_error = f"{type(error).__name__}: {error}"
        if effective_returncode == 0:
            effective_returncode = 73

    solver_sessions: dict[str, Any] | None = None
    solver_sessions_error: str | None = None
    try:
        if public_progress is None:
            raise RuntimeError("public progress is unavailable for Solver reconciliation")
        terminal_states = {
            game: record["final_state"]
            for game, record in public_progress["games"].items()
        }
        expected_boundary_levels = {
            game: record["boundary_levels"]
            for game, record in public_progress["games"].items()
        }
        solver_sessions = summarize_solver_sessions(
            args.workspace_root,
            terminal_states=terminal_states,
            expected_boundary_levels=expected_boundary_levels,
        )
    except Exception as error:  # noqa: BLE001 - preserve qualification failure
        solver_sessions_error = f"{type(error).__name__}: {error}"
        if effective_returncode == 0:
            effective_returncode = 74

    record = {
        "schema": "tycho-nooa-public25-launcher-result-v2",
        "status": "passed" if effective_returncode == 0 else "failed",
        "started_at": started_at,
        "ended_at": datetime.now(UTC).isoformat(),
        "returncode": effective_returncode,
        "launcher_returncode": returncode,
        "timed_out": timed_out,
        "residual_group_term_required": residual_group_term_required,
        "command": command,
        "stdout": log_facts(stdout_path),
        "stderr": log_facts(stderr_path),
        "operation_mode": "offline",
        "public_games": EXPECTED_GAMES,
        "public_game_count": len(EXPECTED_GAMES),
        "solver_workspace_root": str(args.workspace_root),
        "scorecard": scorecard,
        "scorecard_status": scorecard_status,
        "scorecard_error": scorecard_error,
        "public_progress": public_progress,
        "public_progress_error": public_progress_error,
        "solver_sessions": solver_sessions,
        "solver_sessions_error": solver_sessions_error,
        "model": args.model,
        "model_base_url": args.base_url,
    }
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if effective_returncode != 0:
        print(record["stdout"]["tail"], file=sys.stdout)
        print(record["stderr"]["tail"], file=sys.stderr)
        return effective_returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
