"""One host-owned construction episode per game from this run's executed prefix."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from host import ConstructionLimits, FixtureTransport, HttpTransport
from prefix import project_transitions

MARKER = "CONSTRUCTION.json"
TERMINAL = {"CANDIDATE_READY", "NEEDS_EVIDENCE", "BUDGET_EXHAUSTED"}
JOB = Path(__file__).resolve().parent / "construction_job.py"


def load_transport() -> Any:
    kind = os.environ.get("ARC_CONSTRUCTOR_TRANSPORT")
    if not kind:
        raise ValueError("ARC_CONSTRUCTOR_TRANSPORT required")
    if kind == "http":
        url = os.environ.get("ARC_CONSTRUCTOR_URL")
        if not url:
            raise ValueError("ARC_CONSTRUCTOR_URL required for http transport")
        return HttpTransport(url)
    if kind != "fixture":
        raise ValueError(f"unknown ARC_CONSTRUCTOR_TRANSPORT {kind!r}")
    path = os.environ.get("ARC_CONSTRUCTOR_FIXTURE")
    if not path:
        raise ValueError("ARC_CONSTRUCTOR_FIXTURE required for fixture transport")
    responses = json.loads(Path(path).read_text())
    if not isinstance(responses, list):
        raise ValueError("constructor fixture must be a JSON list of provider responses")
    return FixtureTransport(responses)


def _limits() -> ConstructionLimits:
    # Donor stops when remaining wall <= 20s, so wall must exceed 20 or zero requests run.
    return ConstructionLimits(
        wall_seconds=int(os.environ.get("ARC_CONSTRUCTOR_WALL_SECONDS", "240")),
        max_model_requests=int(os.environ.get("ARC_CONSTRUCTOR_MAX_REQUESTS", "6")),
    )


def _reap(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.wait()


def _promote(work: Path, run_dir: Path) -> None:
    for name, mode in (("FROZEN.json", 0o444), ("RESULT.json", 0o644)):
        source = work / name
        if not source.is_file():
            continue
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(run_dir / name, flags, mode)
        try:
            os.write(fd, source.read_bytes())
            os.fsync(fd)
        finally:
            os.close(fd)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def construct_from_run(run_dir: Path, ipc_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    marker = run_dir / MARKER
    frozen = run_dir / "FROZEN.json"
    if frozen.exists():
        return {"construction_status": "ALREADY_FROZEN", "frozen": True, "capped": True, "playable": True}
    if marker.exists():
        existing = json.loads(marker.read_text())
        if existing.get("capped"):
            return existing
    transitions = Path(ipc_dir) / "transitions.jsonl"
    if not transitions.is_file() or not transitions.read_text().strip():
        return {"construction_status": "NO_PREFIX", "frozen": False, "capped": False, "playable": True}
    try:
        project_transitions(transitions)
    except Exception as error:
        return {
            "construction_status": "FAILED",
            "frozen": False,
            "supported_model": False,
            "capped": False,
            "playable": True,
            "error": f"{type(error).__name__}: {error}",
        }
    # Persist one-shot admission before spawning; failure/crash cannot retry this episode.
    marker.write_text(json.dumps({"construction_status": "WORKING", "frozen": False, "capped": True, "playable": True}) + "\n")
    hard = float(os.environ.get("ARC_CONSTRUCTOR_HARD_SECONDS", "245"))
    work = run_dir / "constructor-job"
    work.mkdir(exist_ok=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(JOB), "--ipc", str(ipc_dir), "--out", str(work)],
        start_new_session=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pid = proc.pid
    try:
        try:
            proc.wait(timeout=hard)
        except subprocess.TimeoutExpired:
            _reap(proc)
            marker.write_text(json.dumps({"construction_status": "FAILED", "frozen": False, "capped": True, "playable": True, "error": f"HARD_DEADLINE:{hard}", "evidence_directory": str(work)}) + "\n")
            return {
                "construction_status": "FAILED",
                "frozen": False,
                "supported_model": False,
                "capped": True,
                "playable": True,
                "error": f"HARD_DEADLINE:{hard}",
                "evidence_directory": str(work),
                "worker_pid": pid,
                "worker_alive": _alive(pid),
            }
    except BaseException:
        _reap(proc)
        raise
    if proc.returncode != 0:
        error = "constructor job failed"
        job_error = work / "JOB-ERROR.json"
        if job_error.is_file():
            error = json.loads(job_error.read_text()).get("error", error)
        marker.write_text(json.dumps({"construction_status": "FAILED", "frozen": False, "capped": True, "playable": True, "error": error, "evidence_directory": str(work)}) + "\n")
        return {
            "construction_status": "FAILED",
            "frozen": False,
            "supported_model": False,
            "capped": True,
            "playable": True,
            "error": error,
            "evidence_directory": str(work),
            "worker_pid": pid,
            "worker_alive": _alive(pid),
        }
    _promote(work, run_dir)
    result_path = run_dir / "RESULT.json"
    record = json.loads(result_path.read_text()) if result_path.is_file() else {
        "construction_status": "FAILED",
        "frozen": False,
    }
    record["playable"] = True
    record["worker_pid"] = pid
    record["worker_alive"] = _alive(pid)
    if record.get("construction_status") in TERMINAL or (run_dir / "FROZEN.json").exists():
        record["capped"] = True
        marker.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    else:
        record["capped"] = False
    return record
