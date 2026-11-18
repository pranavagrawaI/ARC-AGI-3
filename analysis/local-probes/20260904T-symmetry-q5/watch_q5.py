#!/usr/bin/env python3
"""Monitor q5. Print only DONE / FAILED / CANCELLED / ACTION_REQUIRED.

Diagnostics go to watch_q5.log keyed on the pidfile pid.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
import urllib.request
from pathlib import Path

PROBE = Path(__file__).resolve().parent
PIDFILE = PROBE / "q5.pid"
OUT = PROBE / "q5.out"
CAP = 2.50
WALL = 2 * 3600


def remaining(key: str) -> float:
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        d = json.load(r).get("data", {})
    return float(d["limit_remaining"])


def run_pid() -> int | None:
    if not PIDFILE.exists():
        return None
    try:
        pid = int(PIDFILE.read_text().strip())
    except ValueError:
        return None
    try:
        os.kill(pid, 0)
        return pid
    except OSError:
        return None


def run_dir() -> Path | None:
    hits = sorted((PROBE / "results" / "nemo_solver").glob("*_ls20_mdfiles_q5"))
    return hits[-1] if hits else None


def log(msg: str, pid: int | None) -> None:
    path = PROBE / f"watch_q5_{pid or 'nopid'}.log"
    with path.open("a") as f:
        f.write(time.strftime("%H:%M:%S") + " " + msg + "\n")


def kill(pid: int | None) -> None:
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def reasoning_tokens(rd: Path | None) -> int:
    if rd is None:
        return 0
    n = 0
    for p in rd.glob("traces/*.jsonl"):
        for line in p.read_text(errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += int(o.get("reasoning_tokens") or 0)
            usage = o.get("usage") or {}
            n += int(usage.get("reasoning_tokens") or 0)
    return n


def main() -> int:
    start_rem = float(sys.argv[1])
    key = os.environ["ARC_LLM_API_KEY"]
    started = time.time()
    deadline = started + WALL
    while time.time() < deadline:
        pid = run_pid()
        rd = run_dir()
        try:
            rem = remaining(key)
            spent = start_rem - rem
        except Exception as e:
            rem, spent = None, 0.0
            log(f"credit_poll_failed {type(e).__name__}", pid)
        if spent >= CAP:
            kill(pid)
            log(f"spend {spent:.2f} >= {CAP}", pid)
            print("FAILED: spend cap", flush=True)
            return 2
        if rd and (rd / "ipc" / "ABORT").exists():
            log("ABORT sentinel", pid)
            print("FAILED: refusal deadlock", flush=True)
            return 3
        if reasoning_tokens(rd) > 0:
            kill(pid)
            log("reasoning_tokens > 0", pid)
            print("FAILED: thinking-on", flush=True)
            return 4
        n_empty = 0
        if rd and (rd / "launcher.err").is_file():
            n_empty = (rd / "launcher.err").read_text(errors="ignore").count("EmptyContentError")
        actions = 0
        if rd and (rd / "ipc" / "actions.jsonl").is_file():
            actions = sum(
                1 for line in (rd / "ipc" / "actions.jsonl").read_text().splitlines() if line.strip()
            )
        if n_empty >= 8 and actions == 0:
            kill(pid)
            log(f"EmptyContentError={n_empty} actions=0", pid)
            print("FAILED: hatch", flush=True)
            return 6
        if time.time() - started > 180 and rd is None:
            kill(pid)
            log("no run dir after 180s", pid)
            print("FAILED: no run dir", flush=True)
            return 7
        if pid is None:
            if rd and (rd / "gameplay.json").is_file():
                print("DONE", flush=True)
                return 0
            tail = OUT.read_text(errors="ignore")[-500:] if OUT.exists() else ""
            log(f"pid dead without gameplay: {tail!r}", pid)
            print("FAILED: orchestrator died", flush=True)
            return 1
        time.sleep(30)
    kill(run_pid())
    print("CANCELLED: wall", flush=True)
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
