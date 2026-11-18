#!/usr/bin/env python3
"""Double-fork q2d so run_solver's PR_SET_PDEATHSIG parent is PID 1.

setsid+nohup is not enough: run_solver.py calls prctl(PR_SET_PDEATHSIG, SIGTERM),
so the kernel kills it when the launch shell exits. The 16:05 attempt died that way.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROBE = Path(__file__).resolve().parent
LAUNCH = PROBE / "launch_q2d.sh"
PIDFILE = PROBE / "q2d.pid"
OUT = PROBE / "q2d.out"


def main() -> None:
    if os.fork() > 0:
        return
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    os.chdir(PROBE)
    os.umask(0o022)
    devnull = os.open("/dev/null", os.O_RDWR)
    out = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(devnull, 0)
    os.dup2(out, 1)
    os.dup2(out, 2)
    if devnull > 2:
        os.close(devnull)
    if out > 2:
        os.close(out)
    PIDFILE.write_text(str(os.getpid()))
    os.execv("/bin/bash", ["bash", str(LAUNCH)])


if __name__ == "__main__":
    main()
    # First-parent path: the grandchild pid is not known yet. Poll briefly.
    for _ in range(50):
        if PIDFILE.exists() and PIDFILE.read_text().strip().isdigit():
            pid = int(PIDFILE.read_text().strip())
            try:
                os.kill(pid, 0)
            except OSError:
                sys.exit(1)
            print(pid)
            sys.exit(0)
        import time

        time.sleep(0.05)
    sys.exit(1)
