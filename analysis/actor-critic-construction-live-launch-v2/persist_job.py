#!/usr/bin/env python3
"""Isolated constructor job that keeps tool-events. Overlay construction_job.py is not edited."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import sys
from pathlib import Path

OVERLAY = Path(__file__).resolve().parents[1] / "actor-critic-construction-integrated"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(OVERLAY))

from construction_service import _limits, load_transport  # noqa: E402
from persist import persist_episode  # noqa: E402
from prefix import project_transitions  # noqa: E402


def main() -> int:
    libc = ctypes.CDLL(None)
    if libc.prctl(1, signal.SIGKILL) != 0:
        raise OSError("could not set parent-death signal")
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGKILL)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipc", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    try:
        case = project_transitions(Path(args.ipc) / "transitions.jsonl")
        persist_episode(case, load_transport(), out, limits=_limits())
    except Exception as error:
        out.mkdir(parents=True, exist_ok=True)
        (out / "JOB-ERROR.json").write_text(json.dumps({"error": f"{type(error).__name__}: {error}"}) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

