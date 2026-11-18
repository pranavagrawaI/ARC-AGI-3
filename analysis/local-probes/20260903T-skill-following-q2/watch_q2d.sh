#!/usr/bin/env bash
# Kill q2d if OpenRouter spend from launch exceeds $6, or if EmptyContentError
# dominates the first minutes (hatch failed). Print only DONE/FAILED/CANCELLED.
set -euo pipefail

PROBE="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$PROBE/q2d.pid"
START_REMAINING="${1:?start remaining dollars}"
CAP="${2:-6}"
RUN_GLOB="$PROBE/results/nemo_solver/"*"_ls20_mdfiles_q2d"

python3 - "$PIDFILE" "$START_REMAINING" "$CAP" "$PROBE" <<'PY'
import json, os, sys, time, urllib.request, glob, signal
from pathlib import Path

pidfile, start_rem, cap, probe = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
# credentials already in this process env via the parent
key = os.environ["ARC_LLM_API_KEY"]

def remaining():
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        d = json.load(r).get("data", {})
    return float(d["limit_remaining"])

def run_pid():
    p = Path(pidfile)
    if not p.exists():
        return None
    try:
        pid = int(p.read_text().strip())
    except ValueError:
        return None
    try:
        os.kill(pid, 0)
        return pid
    except OSError:
        return None

def run_dir():
    hits = sorted(glob.glob(str(probe / "results/nemo_solver" / "*_ls20_mdfiles_q2d")))
    return Path(hits[-1]) if hits else None

def empty_content_count(rd: Path | None) -> int:
    if rd is None:
        return 0
    err = rd / "launcher.err"
    if not err.is_file():
        return 0
    return err.read_text(errors="ignore").count("EmptyContentError")

deadline = time.time() + 4 * 3600  # 4h wall; 40 turns * 300s timeout is the real bound
empty_streak_start = None
while time.time() < deadline:
    pid = run_pid()
    rd = run_dir()
    try:
        rem = remaining()
    except Exception:
        rem = None
    spent = (start_rem - rem) if rem is not None else 0.0
    if spent >= cap:
        if pid:
            os.kill(pid, signal.SIGTERM)
        print("FAILED", flush=True)
        sys.exit(2)
    n_empty = empty_content_count(rd)
    # hatch failed if we see many EmptyContentErrors early with no actions
    actions = 0
    if rd and (rd / "ipc/actions.jsonl").is_file():
        actions = sum(1 for line in (rd / "ipc/actions.jsonl").read_text().splitlines() if line.strip())
    if n_empty >= 8 and actions == 0:
        if empty_streak_start is None:
            empty_streak_start = time.time()
        elif time.time() - empty_streak_start > 180:
            if pid:
                os.kill(pid, signal.SIGTERM)
            print("FAILED", flush=True)
            sys.exit(3)
    else:
        empty_streak_start = None
    if pid is None:
        out = (probe / "q2d.out").read_text(errors="ignore") if (probe / "q2d.out").is_file() else ""
        if "signal 15" in out:
            print("FAILED", flush=True)
            sys.exit(1)
        if rd and (rd / "gameplay.json").is_file():
            print("DONE", flush=True)
            sys.exit(0)
        print("FAILED", flush=True)
        sys.exit(1)
    time.sleep(20)

pid = run_pid()
if pid:
    os.kill(pid, signal.SIGTERM)
print("CANCELLED", flush=True)
sys.exit(4)
PY
