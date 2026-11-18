#!/usr/bin/env python3
"""v11 + overlay ConstructionAwareAgent with live docs/limits. Overlay headless.py is not edited."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OVERLAY = Path(__file__).resolve().parent
V11 = Path(__file__).resolve().parent / "nooa-join"
LIVE = Path(__file__).resolve().parent
sys.path.insert(0, str(LIVE))
sys.path.insert(0, str(OVERLAY))
sys.path.insert(0, str(V11))
sys.path.insert(0, str(V11 / "chassis"))

spec = importlib.util.spec_from_file_location("v11_headless", V11 / "headless.py")
v11 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v11)

import construction_service  # noqa: E402
from live_agent import LiveConstructionAgent  # noqa: E402

construction_service.JOB = LIVE / "persist_job.py"
_orig_promote = construction_service._promote


def _promote_with_progress(work: Path, run_dir: Path) -> None:
    _orig_promote(work, run_dir)
    for name, mode in (
        ("tool-events.jsonl", 0o644),
        ("CONSTRUCTOR-PROGRESS.json", 0o644),
        ("evaluation-error.json", 0o644),
        ("JOB-ERROR.json", 0o644),
        ("RESULT.json", 0o644),
    ):
        source = work / name
        target = run_dir / name
        if source.is_file() and not target.exists():
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            fd = os.open(target, flags, mode)
            try:
                os.write(fd, source.read_bytes())
                os.fsync(fd)
            finally:
                os.close(fd)


construction_service._promote = _promote_with_progress
v11.MdArcSolverAgent = LiveConstructionAgent


def apply_live_argv(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--constructor-transport")
    parser.add_argument("--constructor-fixture")
    parser.add_argument("--constructor-url")
    parser.add_argument("--constructor-wall-seconds", type=int)
    parser.add_argument("--constructor-hard-seconds", type=int)
    parser.add_argument("--constructor-max-requests", type=int)
    args, rest = parser.parse_known_args(argv)
    if args.constructor_transport:
        os.environ["ARC_CONSTRUCTOR_TRANSPORT"] = args.constructor_transport
    if args.constructor_fixture:
        os.environ["ARC_CONSTRUCTOR_FIXTURE"] = args.constructor_fixture
    if args.constructor_url:
        os.environ["ARC_CONSTRUCTOR_URL"] = args.constructor_url
    os.environ.setdefault("ARC_CONSTRUCTOR_WALL_SECONDS", "45")
    os.environ.setdefault("ARC_CONSTRUCTOR_HARD_SECONDS", "50")
    os.environ.setdefault("ARC_CONSTRUCTOR_MAX_REQUESTS", "2")
    if args.constructor_wall_seconds is not None:
        os.environ["ARC_CONSTRUCTOR_WALL_SECONDS"] = str(args.constructor_wall_seconds)
    if args.constructor_hard_seconds is not None:
        if args.constructor_hard_seconds > 60:
            raise ValueError("constructor hard seconds exceed the incumbent 60s cell")
        os.environ["ARC_CONSTRUCTOR_HARD_SECONDS"] = str(args.constructor_hard_seconds)
    if args.constructor_max_requests is not None:
        os.environ["ARC_CONSTRUCTOR_MAX_REQUESTS"] = str(args.constructor_max_requests)
    return rest


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *apply_live_argv(sys.argv[1:])]
    asyncio.run(v11.run())

