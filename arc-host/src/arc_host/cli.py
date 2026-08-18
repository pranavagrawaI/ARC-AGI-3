"""Run an ARC game against one persistent solver subprocess."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import arc_agi
from arc_agi.base import OperationMode
from dotenv import load_dotenv

from arc_host.process import JsonlWorker
from arc_host.runner import run_episode
from arc_host.trace import RecordingSolver


def _default_run_dir(game_id: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"{stamp}-{game_id}"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_environment(path: Path | None) -> None:
    """Load an explicitly selected secrets file without overriding the shell."""
    if path is None:
        return
    if not path.is_file():
        raise SystemExit(f"environment file does not exist: {path}")
    load_dotenv(dotenv_path=path, override=False)


def _close_resources(*resources: tuple[str, object]) -> list[dict[str, str]]:
    """Close supported resources without letting cleanup hide a run failure."""
    errors: list[dict[str, str]] = []
    for name, resource in resources:
        close = getattr(resource, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except Exception as error:  # noqa: BLE001 - preserve cleanup diagnostics
            errors.append(
                {
                    "resource": name,
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument("--stop-after-levels", type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--worker-timeout", type=float, default=3_600)
    parser.add_argument(
        "--env-file",
        type=Path,
        help="explicit dotenv file loaded before ARC and worker configuration",
    )
    parser.add_argument(
        "worker_command",
        nargs=argparse.REMAINDER,
        help="worker command after --, for example: -- uv run python -m solver.worker",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _load_environment(args.env_file)
    worker_command = list(args.worker_command)
    if worker_command[:1] == ["--"]:
        worker_command = worker_command[1:]
    if not worker_command:
        raise SystemExit("a worker command is required after --")
    api_key = os.environ.get("ARC_API_KEY")
    if not api_key:
        raise SystemExit("ARC_API_KEY is required")

    run_dir = (args.run_dir or _default_run_dir(args.game)).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    workspace_root = run_dir / "workspace"
    workspace_root.mkdir()
    arcade = arc_agi.Arcade(
        arc_api_key=api_key,
        operation_mode=OperationMode.ONLINE,
        recordings_dir=str(run_dir / "raw-recordings"),
    )
    environment = arcade.make(args.game, seed=args.seed, save_recording=True)
    if environment is None:
        raise RuntimeError(f"ARC service did not create {args.game!r}")

    worker = JsonlWorker(
        worker_command,
        stdout_log=run_dir / "worker-stdout.log",
        stderr_log=run_dir / "worker-stderr.log",
        timeout_seconds=args.worker_timeout,
    )
    solver = RecordingSolver(worker, run_dir / "transcript.jsonl")
    result = None
    run_error: Exception | None = None
    run_traceback = None
    try:
        result = run_episode(
            environment,
            solver,
            game_id=args.game,
            seed=args.seed,
            workspace_root=workspace_root,
            max_actions=args.max_actions,
            stop_after_levels=args.stop_after_levels,
        )
    except Exception as error:  # noqa: BLE001 - write the failure record, then re-raise
        run_error = error
        run_traceback = error.__traceback__
        solver.record_error(error)

    cleanup_errors = _close_resources(("solver", solver), ("environment", environment))
    if result is not None:
        summary = {
            "schema": "arc-host-summary-v1",
            "game_id": args.game,
            "seed": args.seed,
            "stop_reason": result.stop_reason,
            "actions": result.actions,
            "levels_completed": result.levels_completed,
            "success": result.levels_completed > result.initial.levels_completed,
            "worker_command": worker_command,
        }
    else:
        assert run_error is not None
        summary = {
            "schema": "arc-host-summary-v1",
            "game_id": args.game,
            "seed": args.seed,
            "stop_reason": "exception",
            "actions": solver.actions_recorded,
            "levels_completed": solver.latest_levels_completed,
            "success": False,
            "worker_command": worker_command,
            "error": {
                "type": type(run_error).__name__,
                "message": str(run_error),
            },
        }
    if cleanup_errors:
        summary["cleanup_errors"] = cleanup_errors
    _write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if run_error is not None:
        raise run_error.with_traceback(run_traceback)
    if cleanup_errors:
        raise RuntimeError("resource cleanup failed; see summary.json")
