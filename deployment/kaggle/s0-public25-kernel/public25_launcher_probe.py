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


class GameListHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
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
        raise RuntimeError("final scorecard is not a JSON object")
    environments = scorecard.get("environments", [])
    if not isinstance(environments, list):
        raise RuntimeError("final scorecard environments is not a list")
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
                raise RuntimeError(
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
        expected_nonterminal_frames = PUBLIC_MAX_ACTIONS
        if final_state == "NOT_FINISHED" and frame_count != expected_nonterminal_frames:
            raise RuntimeError(
                f"nonterminal recording did not exhaust the configured official loop: "
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


def summarize_solver_sessions(workspace_root: Path) -> dict[str, Any]:
    sessions: dict[str, dict[str, Any]] = {}
    for game in EXPECTED_GAMES:
        matches = [
            path
            for path in workspace_root.iterdir()
            if path.is_dir() and (path / "workspace" / game).is_dir()
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one Solver session for {game}, found {matches}")
        session = matches[0]
        metadata_path = session / "worker-metadata.json"
        tool_trace_path = session / "tool-trace.jsonl"
        if not metadata_path.is_file() or not tool_trace_path.is_file():
            raise RuntimeError(f"Solver evidence is incomplete for {game}")
        metadata = json.loads(metadata_path.read_text())
        actions: list[dict[str, Any]] = []
        for line_number, line in enumerate(tool_trace_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid Solver tool trace: {tool_trace_path}:{line_number}"
                ) from error
            if event.get("name") == "take_action" and event.get("status") == "complete":
                actions.append(
                    {
                        "sequence": event.get("sequence"),
                        "validated_arguments": event.get("validated_arguments"),
                    }
                )
        if len(actions) != PUBLIC_MAX_ACTIONS:
            raise RuntimeError(
                f"Solver action count differs for {game}: "
                f"expected {PUBLIC_MAX_ACTIONS}, got {len(actions)}"
            )
        if [action["sequence"] for action in actions] != list(range(len(actions))):
            raise RuntimeError(f"Solver action sequences differ for {game}")
        sessions[game] = {
            "session": str(session),
            "worker_metadata": metadata,
            "actions": actions,
            "tool_trace": log_facts(tool_trace_path),
        }
    return {
        "schema": "tycho-nooa-public-solver-sessions-v1",
        "game_count": len(sessions),
        "total_solver_actions": sum(len(value["actions"]) for value in sessions.values()),
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
    environment = {
        **os.environ,
        "ARC_LLM_API_KEY": "local-model-no-key",
        "ARC_LLM_BASE_URL": args.base_url,
        "ARC_LLM_MODEL": args.model,
        "ARC_LLM_PROVIDER_ENDPOINT": "local",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "MPLBACKEND": "agg",
        "TYCHO_KAGGLE_HARDENED_HOST": "1",
        "TYCHO_NOOA_WORKER_PYTHON": sys.executable,
        "TYCHO_SANDBOX_RUNTIME": "host",
    }
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
    except Exception as error:  # preserved as a typed qualification failure
        scorecard_error = f"{type(error).__name__}: {error}"
        if effective_returncode == 0:
            effective_returncode = 72

    public_progress: dict[str, Any] | None = None
    public_progress_error: str | None = None
    try:
        recordings_dir = recordings_dir_from_env(official_root / ".env")
        public_progress = summarize_recordings(recordings_dir)
    except Exception as error:
        public_progress_error = f"{type(error).__name__}: {error}"
        if effective_returncode == 0:
            effective_returncode = 73

    solver_sessions: dict[str, Any] | None = None
    solver_sessions_error: str | None = None
    try:
        solver_sessions = summarize_solver_sessions(args.workspace_root)
    except Exception as error:
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
