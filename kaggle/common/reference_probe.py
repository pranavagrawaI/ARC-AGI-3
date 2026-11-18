"""Run the official ARC-AGI-3 launcher over the public games through one reference arm.

Both arms — upstream NOOA and upstream Tycho — sit behind the same official
``MyAgent`` seam and are measured by the same instrument: the official recordings
under ``RECORDINGS_DIR``. Only the per-arm launcher environment and the per-arm
Solver-session summary differ, so a difference in the measured outcome is a
difference between the harnesses rather than between two measuring rigs.

Phase A only. The probe refuses competition-rerun mode and creates no submission.
"""

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
from pathlib import Path
from typing import Any

PUBLIC_GAMES = [
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59",
    "lf52", "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26",
    "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33",
    "wa30",
]
requested_games = os.getenv("TYCHO_PUBLIC_GAME_IDS", ",".join(PUBLIC_GAMES))
EXPECTED_GAMES = requested_games.split(",")
if (
    not EXPECTED_GAMES
    or len(EXPECTED_GAMES) != len(set(EXPECTED_GAMES))
    or any(game not in PUBLIC_GAMES for game in EXPECTED_GAMES)
):
    raise RuntimeError("TYCHO_PUBLIC_GAME_IDS must name a unique public-game subset")
PUBLIC_MAX_ACTIONS = int(os.getenv("TYCHO_KAGGLE_MAX_ACTIONS", "16"))
ARM = os.environ["REFERENCE_ARM"]
if ARM not in {"nooa-upstream", "tycho-upstream"}:
    raise RuntimeError(f"unknown REFERENCE_ARM {ARM!r}")
EXPECTED_JOIN = f"s0-{ARM}-join"
LIST_HOST = "127.0.0.1"
LIST_PORT = 18_001
WORKING_ROOT = Path(os.getenv("TYCHO_PUBLIC25_WORKING_ROOT", "/kaggle/working"))


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
    parser.add_argument("--timeout-s", type=int, default=7_200)
    return parser.parse_args()


def log_facts(path: Path) -> dict[str, Any]:
    value = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
        "tail": value[-8_192:].decode("utf-8", errors="replace"),
    }


def recordings_dir_from_env(dotenv_path: Path) -> Path:
    for raw_line in dotenv_path.read_text().splitlines():
        key, separator, value = raw_line.partition("=")
        if separator and key.strip() == "RECORDINGS_DIR":
            path = Path(value.strip())
            if not path.is_absolute():
                raise RuntimeError("RECORDINGS_DIR is not absolute")
            return path
    raise RuntimeError("official launcher .env has no RECORDINGS_DIR")


def _frame_hash(frame: Any) -> str:
    return hashlib.sha256(
        json.dumps(frame, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------------------
# The shared instrument: what the official recordings say happened
# --------------------------------------------------------------------------


def summarize_recordings(
    recordings_dir: Path, *, require_complete: bool = True
) -> dict[str, Any]:
    if not recordings_dir.is_dir():
        raise RuntimeError(f"recordings directory is absent: {recordings_dir}")
    paths = sorted(recordings_dir.glob("*.recording.jsonl"))
    by_game: dict[str, dict[str, Any]] = {}
    for path in paths:
        game = path.name.split(".", 1)[0]
        if game not in EXPECTED_GAMES:
            raise RuntimeError(f"unexpected public recording: {path.name}")
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        data_rows = [
            event.get("data")
            for event in rows
            if isinstance(event, dict) and isinstance(event.get("data"), dict)
        ]
        if not data_rows:
            raise RuntimeError(f"recording has no frames: {path.name}")
        final = data_rows[-1]
        action_ids = [
            row.get("action_input", {}).get("id")
            for row in data_rows
            if isinstance(row.get("action_input"), dict)
        ]
        levels = [
            row["levels_completed"]
            for row in data_rows
            if isinstance(row.get("levels_completed"), int)
        ]
        hashes = [_frame_hash(row.get("frame")) for row in data_rows]
        by_game[game] = {
            "recording": log_facts(path),
            "recorded_action_results": len(data_rows),
            "recording_action_input_ids": action_ids,
            "grid_hash_changes": sum(
                1 for left, right in zip(hashes, hashes[1:]) if left != right
            ),
            "final_state": final.get("state"),
            "final_levels_completed": final.get("levels_completed"),
            "max_levels_completed": max(levels) if levels else 0,
        }
    missing = [game for game in EXPECTED_GAMES if game not in by_game]
    if not by_game:
        raise RuntimeError("no public recordings")
    if require_complete and missing:
        raise RuntimeError(f"recording game IDs differ: {sorted(by_game)}")
    return {
        "schema": "reference-arm-public-progress-v1",
        "arm": ARM,
        "game_count": len(by_game),
        "missing_games": missing,
        "total_recorded_action_results": sum(
            game["recorded_action_results"] for game in by_game.values()
        ),
        "total_grid_hash_changes": sum(
            game["grid_hash_changes"] for game in by_game.values()
        ),
        "total_levels_completed": sum(
            game["final_levels_completed"] or 0 for game in by_game.values()
        ),
        "games": by_game,
    }


# --------------------------------------------------------------------------
# Per-arm Solver-session evidence
# --------------------------------------------------------------------------


def _session_metadata(session: Path, game: str) -> dict[str, Any]:
    metadata = json.loads((session / "worker-metadata.json").read_text())
    if metadata.get("join") != EXPECTED_JOIN:
        raise RuntimeError(f"worker join identity differs for {game}: {metadata}")
    return metadata


def _summarize_nooa_session(session: Path, game: str) -> dict[str, Any]:
    metadata = _session_metadata(session, game)
    actions_path = session / "ipc" / "actions.jsonl"
    states_path = session / "ipc" / "states.jsonl"
    if not actions_path.is_file() or not states_path.is_file():
        raise RuntimeError(f"NOOA IPC is absent for {game}")
    action_lines = [line for line in actions_path.read_text().splitlines() if line.strip()]
    state_lines = [line for line in states_path.read_text().splitlines() if line.strip()]
    if not action_lines:
        raise RuntimeError(f"NOOA submitted no actions for {game}")
    batches = [json.loads(line) for line in action_lines]
    submitted = [action for batch in batches for action in (batch.get("actions") or [])]
    return {
        "session": str(session),
        "worker_metadata": metadata,
        "submitted_batches": len(action_lines),
        "submitted_env_actions": len(submitted),
        "submitted_action_names": submitted,
        "state_lines": len(state_lines),
        "first_batch": batches[0],
    }


def _summarize_tycho_session(session: Path, game: str) -> dict[str, Any]:
    metadata = _session_metadata(session, game)
    record_path = session / "tycho-run-record.json"
    record = json.loads(record_path.read_text()) if record_path.is_file() else None
    trace = (record or {}).get("trace") or []
    submitted = [str(step.get("action")) for step in trace if step.get("action")]
    if record is None and not submitted:
        # run_env writes its record only when it returns; a run cut short by the
        # host budget leaves none. That is a truncation, not an absent session.
        return {
            "session": str(session),
            "worker_metadata": metadata,
            "run_record": None,
            "submitted_batches": 0,
            "submitted_env_actions": 0,
            "submitted_action_names": [],
            "state_lines": 0,
            "first_batch": None,
            "truncated_before_run_record": True,
        }
    return {
        "session": str(session),
        "worker_metadata": metadata,
        "run_record": {
            key: record.get(key)
            for key in (
                "levels_completed",
                "total_actions",
                "resets",
                "final_state",
                "env_score",
                "stop_reason",
                "error",
                "noop_actions",
                "distinct_frames",
                "revisits",
                "actions_before_first_level",
                "inference_budget",
                "builder_invocations",
            )
        }
        if record
        else None,
        # One Tycho turn commits exactly one action, so batches == actions here.
        "submitted_batches": len(submitted),
        "submitted_env_actions": len(submitted),
        "submitted_action_names": submitted,
        "state_lines": len(trace),
        "first_batch": {"actions": submitted[:1]} if submitted else None,
        "truncated_before_run_record": False,
    }


def _summarize_session(session: Path, game: str) -> dict[str, Any]:
    if ARM == "nooa-upstream":
        return _summarize_nooa_session(session, game)
    return _summarize_tycho_session(session, game)


def summarize_solver_sessions(
    workspace_root: Path, *, require_complete: bool = True
) -> dict[str, Any]:
    sessions: dict[str, Any] = {}
    missing: list[str] = []
    if not workspace_root.is_dir():
        raise RuntimeError(f"solver workspace root is absent: {workspace_root}")
    for game in EXPECTED_GAMES:
        matches = [
            path
            for path in workspace_root.iterdir()
            if path.is_dir() and (path / "workspace" / game).is_dir()
        ]
        if not matches:
            missing.append(game)
            continue
        episodes = [_summarize_session(path, game) for path in sorted(matches)]
        submitted = [
            action for episode in episodes for action in episode["submitted_action_names"]
        ]
        sessions[game] = {
            "session": episodes[-1]["session"],
            "episode_count": len(episodes),
            "episodes": episodes,
            "worker_metadata": episodes[0]["worker_metadata"],
            "submitted_batches": sum(episode["submitted_batches"] for episode in episodes),
            "submitted_env_actions": len(submitted),
            "submitted_action_names": submitted,
            "state_lines": sum(episode["state_lines"] for episode in episodes),
            "first_batch": episodes[0]["first_batch"],
        }
    if not sessions:
        raise RuntimeError(f"{ARM} produced no Solver sessions")
    if require_complete and missing:
        raise RuntimeError(f"expected one Solver session for {missing}, found none")
    return {
        "schema": "reference-arm-sessions-v1",
        "arm": ARM,
        "game_count": len(sessions),
        "missing_games": missing,
        "total_submitted_batches": sum(v["submitted_batches"] for v in sessions.values()),
        "total_submitted_env_actions": sum(
            v["submitted_env_actions"] for v in sessions.values()
        ),
        "games": sessions,
    }


# --------------------------------------------------------------------------
# Launch
# --------------------------------------------------------------------------


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


def arm_environment(*, base_url: str, model: str) -> dict[str, str]:
    """The per-arm configuration each upstream reads from its own environment."""
    if ARM == "nooa-upstream":
        # nooa-pin/examples/arc_agi_3/arc_llm.py: ARC_LLM_MODEL/BASE_URL/API_KEY.
        return {
            "ARC_LLM_API_KEY": "local-model-no-key",
            "ARC_LLM_BASE_URL": base_url,
            "ARC_LLM_MODEL": model,
            "ARC_LLM_PROVIDER_ENDPOINT": "local",
            "ARC_LLM_USE_RESPONSES": "0",
        }
    # tycho/serving/llm_client.py LLMConfig.from_env: the openai backend is the
    # OpenAI-compatible chat-completions transport, which is what SGLang serves.
    return {
        "LLM_BACKEND": "openai",
        "LLM_BASE_URL": base_url,
        "LLM_MODEL": model,
        "LLM_API_KEY": "EMPTY",
    }


def official_launcher_environment(
    *,
    base_url: str,
    model: str,
    workspace_root: Path,
    worker_python: Path,
) -> dict[str, str]:
    join_root = Path(os.environ.get("REFERENCE_JOIN_ROOT", f"/kaggle/working/{ARM}-join"))
    return {
        **os.environ,
        **arm_environment(base_url=base_url, model=model),
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "MPLBACKEND": "agg",
        "NOOA_JOIN_ROOT": str(join_root),
        "TYCHO_UPSTREAM_JOIN_ROOT": str(join_root),
        "PYTHONUNBUFFERED": "1",
        "TYCHO_KAGGLE_WORKSPACE_ROOT": str(workspace_root),
        "TYCHO_NOOA_WORKER_PYTHON": str(worker_python),
        # Upstream Tycho runs agent-authored Python in a Docker/Finch container.
        # Kaggle has neither; `host` is upstream's own supported alternative.
        "TYCHO_SANDBOX_RUNTIME": "host",
    }


def main() -> int:
    args = parse_args()
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        raise RuntimeError(f"{ARM} Phase A qualifier refuses competition rerun mode")
    if args.timeout_s <= 60:
        raise ValueError("launcher timeout must exceed 60 seconds")

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
    command = [sys.executable, str(runner), str(main_path), "--agent", "myagent"]
    stdout_path = args.output.with_name(f"{ARM}-launcher.stdout.log")
    stderr_path = args.output.with_name(f"{ARM}-launcher.stderr.log")
    list_server = http.server.ThreadingHTTPServer((LIST_HOST, LIST_PORT), GameListHandler)
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
        124 if timed_out else 71 if residual_group_term_required else returncode
    )
    require_complete = (not timed_out) and returncode == 0
    public_progress = None
    public_progress_error = None
    try:
        recordings_dir = recordings_dir_from_env(official_root / ".env")
        public_progress = summarize_recordings(
            recordings_dir, require_complete=require_complete
        )
        played = (
            public_progress["total_recorded_action_results"] >= 1
            and (
                public_progress["total_grid_hash_changes"] >= 1
                or public_progress["total_recorded_action_results"] >= 2
            )
        )
        if not played:
            raise RuntimeError(f"{ARM} produced no recorded environment steps")
    except Exception as error:  # noqa: BLE001 - preserve qualification failure
        public_progress_error = f"{type(error).__name__}: {error}"
        if effective_returncode == 0:
            effective_returncode = 73

    solver_sessions = None
    solver_sessions_error = None
    try:
        solver_sessions = summarize_solver_sessions(
            args.workspace_root, require_complete=require_complete
        )
        if solver_sessions["total_submitted_env_actions"] < 1:
            raise RuntimeError(f"{ARM} submitted no actions")
    except Exception as error:  # noqa: BLE001 - preserve qualification failure
        solver_sessions_error = f"{type(error).__name__}: {error}"
        if effective_returncode == 0:
            effective_returncode = 74

    record = {
        "schema": "reference-arm-launcher-result-v1",
        "arm": ARM,
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
        "public_max_actions": PUBLIC_MAX_ACTIONS,
        "solver_workspace_root": str(args.workspace_root),
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
