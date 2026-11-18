#!/usr/bin/env python3
"""Isolated constructor job. Writes only to --out. Sealed runner unchanged."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from host import run_episode  # noqa: E402  # identity path
from prefix import project_transitions  # noqa: E402
from construction_service import _limits, load_transport  # noqa: E402


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
    try:
        case = project_transitions(Path(args.ipc) / "transitions.jsonl")
        run_episode(case, load_transport(), Path(args.out), limits=_limits())
    except Exception as error:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "JOB-ERROR.json").write_text(
            json.dumps({"error": f"{type(error).__name__}: {error}"}) + "\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
