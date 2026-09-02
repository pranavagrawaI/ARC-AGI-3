"""Invoke the pinned official launcher against the live local model route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=int, default=32_340)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout_s <= 60:
        raise ValueError("official launcher timeout must exceed 60 seconds")
    if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        raise RuntimeError("official launcher may run only in a competition rerun")

    official_root = Path("/kaggle/working/ARC-AGI-3-Agents")
    runner = Path("/kaggle/working/run_official_agent.py")
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
    started_at = datetime.now(UTC).isoformat()
    stdout_path = args.output.with_name("official-launcher.stdout.log")
    stderr_path = args.output.with_name("official-launcher.stderr.log")
    timed_out = False
    residual_group_term_required = False
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
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait(timeout=10)
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            pass
        else:
            residual_group_term_required = True
            os.killpg(process.pid, signal.SIGTERM)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                os.killpg(process.pid, signal.SIGKILL)

    effective_returncode = (
        124
        if timed_out
        else 71
        if residual_group_term_required
        else returncode
    )

    def log_facts(path: Path) -> dict[str, object]:
        value = path.read_bytes()
        return {
            "path": str(path),
            "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
            "tail": value[-8_192:].decode("utf-8", errors="replace"),
        }

    record = {
        "schema": "tycho-nooa-official-launcher-result-v1",
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
        "operation_mode": "competition",
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
