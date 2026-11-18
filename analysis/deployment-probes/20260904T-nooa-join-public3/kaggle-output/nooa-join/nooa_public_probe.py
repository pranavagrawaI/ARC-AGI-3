"""Run the official launcher over a few public games through the NOOA join."""

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
requested_games = os.getenv("TYCHO_PUBLIC_GAME_IDS", "ls20,s5i5,cd82")
EXPECTED_GAMES = requested_games.split(",")
if (
    not EXPECTED_GAMES
    or len(EXPECTED_GAMES) != len(set(EXPECTED_GAMES))
    or any(game not in PUBLIC_GAMES for game in EXPECTED_GAMES)
):
    raise RuntimeError("TYCHO_PUBLIC_GAME_IDS must name a unique public-game subset")
PUBLIC_MAX_ACTIONS = int(os.getenv("TYCHO_KAGGLE_MAX_ACTIONS", "16"))
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


def summarize_recordings(recordings_dir: Path) -> dict[str, Any]:
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
        by_game[game] = {
            "recording": log_facts(path),
            "recorded_action_results": len(data_rows),
            "solver_action_ids": [
                action_id for action_id in action_ids if action_id not in {None, 0}
            ],
            "final_state": final.get("state"),
            "final_levels_completed": final.get("levels_completed"),
            "max_levels_completed": max(levels) if levels else 0,
        }
    if sorted(by_game) != sorted(EXPECTED_GAMES):
        raise RuntimeError(f"recording game IDs differ: {sorted(by_game)}")
    return {
        "schema": "nooa-join-public-progress-v1",
        "game_count": len(by_game),
        "total_recorded_action_results": sum(
            game["recorded_action_results"] for game in by_game.values()
        ),
        "total_solver_actions": sum(
            len(game["solver_action_ids"]) for game in by_game.values()
        ),
        "total_levels_completed": sum(
            game["final_levels_completed"] or 0 for game in by_game.values()
        ),
        "games": by_game,
    }


def summarize_nooa_sessions(workspace_root: Path) -> dict[str, Any]:
    sessions: dict[str, Any] = {}
    for game in EXPECTED_GAMES:
        matches = [
            path
            for path in workspace_root.iterdir()
            if (
                path.is_dir()
                and (path / "workspace" / game).is_dir()
            )
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one Solver session for {game}, found {matches}")
        session = matches[0]
        metadata = json.loads((session / "worker-metadata.json").read_text())
        if metadata.get("join") != "s0-nooa-join":
            raise RuntimeError(f"worker join identity differs for {game}: {metadata}")
        if metadata.get("chassis") != "q3c-mdfiles-after":
            raise RuntimeError(f"chassis pin differs for {game}: {metadata}")
        actions_path = session / "ipc" / "actions.jsonl"
        states_path = session / "ipc" / "states.jsonl"
        if not actions_path.is_file() or not states_path.is_file():
            raise RuntimeError(f"NOOA IPC is absent for {game}")
        action_lines = [line for line in actions_path.read_text().splitlines() if line.strip()]
        state_lines = [line for line in states_path.read_text().splitlines() if line.strip()]
        if not action_lines:
            raise RuntimeError(f"NOOA submitted no actions for {game}")
        sessions[game] = {
            "session": str(session),
            "worker_metadata": metadata,
            "submitted_batches": len(action_lines),
            "state_lines": len(state_lines),
            "first_batch": json.loads(action_lines[0]),
        }
    return {
        "schema": "nooa-join-sessions-v1",
        "game_count": len(sessions),
        "total_submitted_batches": sum(
            value["submitted_batches"] for value in sessions.values()
        ),
        "games": sessions,
    }


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


def official_launcher_environment(
    *,
    base_url: str,
    model: str,
    workspace_root: Path,
    worker_python: Path,
) -> dict[str, str]:
    join_root = Path(os.environ.get("NOOA_JOIN_ROOT", "/kaggle/working/nooa-join"))
    return {
        **os.environ,
        "ARC_LLM_API_KEY": "local-model-no-key",
        "ARC_LLM_BASE_URL": base_url,
        "ARC_LLM_MODEL": model,
        "ARC_LLM_PROVIDER_ENDPOINT": "local",
        "ARC_LLM_USE_RESPONSES": "0",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "MPLBACKEND": "agg",
        "NOOA_JOIN_ROOT": str(join_root),
        "PYTHONUNBUFFERED": "1",
        "TYCHO_KAGGLE_WORKSPACE_ROOT": str(workspace_root),
        "TYCHO_NOOA_WORKER_PYTHON": str(worker_python),
        "TYCHO_SANDBOX_RUNTIME": "host",
    }


def main() -> int:
    args = parse_args()
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        raise RuntimeError("NOOA join qualifier refuses competition rerun mode")
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
    command = [
        sys.executable,
        str(runner),
        str(main_path),
        "--agent",
        "myagent",
    ]
    stdout_path = args.output.with_name("nooa-join-launcher.stdout.log")
    stderr_path = args.output.with_name("nooa-join-launcher.stderr.log")
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
    public_progress = None
    public_progress_error = None
    try:
        recordings_dir = recordings_dir_from_env(official_root / ".env")
        public_progress = summarize_recordings(recordings_dir)
        if public_progress["total_solver_actions"] < 1:
            raise RuntimeError("NOOA join produced no non-RESET environment actions")
    except Exception as error:  # noqa: BLE001 - preserve qualification failure
        public_progress_error = f"{type(error).__name__}: {error}"
        if effective_returncode == 0:
            effective_returncode = 73

    solver_sessions = None
    solver_sessions_error = None
    try:
        solver_sessions = summarize_nooa_sessions(args.workspace_root)
    except Exception as error:  # noqa: BLE001 - preserve qualification failure
        solver_sessions_error = f"{type(error).__name__}: {error}"
        if effective_returncode == 0:
            effective_returncode = 74

    record = {
        "schema": "nooa-join-public3-launcher-result-v1",
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
