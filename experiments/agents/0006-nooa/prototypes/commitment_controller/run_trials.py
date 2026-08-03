"""PROTOTYPE: execute the prepared controller trials, one at a time.

    set -a && . ./.env && set +a
    uv run --python 3.14 python \
      experiments/agents/0006-nooa/prototypes/commitment_controller/run_trials.py

This spends. Each trial runs only against its own already-prepared approval
request, using the confirmation digest recorded in that file, so it cannot
silently run a configuration that was never approved. A consumed approval is
single-use, so a re-invocation skips trials that already ran.

Each trial is a subprocess, not an in-process call: `run_live_probe` re-executes
itself with `os.execve` to enter the approved snapshot, which replaces the
calling process. Driving it in-process therefore runs exactly one trial and then
silently exits, which is how the first two batches each lost their remaining
trials.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

TREATMENT_ROOT = Path(__file__).resolve().parents[2]
REPO = TREATMENT_ROOT.parents[2]
if str(TREATMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(TREATMENT_ROOT))

from prepare_approvals import BUDGETS

# The pinned ModelRun endpoint answers 429 under sustained load, and an uncertain
# transport is deliberately never retried -- it may already have been served. So
# pacing is the only lever this side of the boundary, and a 429 costs a whole
# trial, which is far more expensive than waiting.
COOLDOWN_SECONDS = 240
APPROVALS = TREATMENT_ROOT / "artifacts" / "approvals"
RUNS = TREATMENT_ROOT / "artifacts" / "runs"
RUNNER = TREATMENT_ROOT / "run_live_probe.py"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not os.environ.get("ARC_LLM_API_KEY"):
        raise SystemExit(
            "ARC_LLM_API_KEY absent. It is defined in .env as ${OPENROUTER_API_KEY}, "
            "which only expands when the file is sourced by a shell: "
            "set -a && . ./.env && set +a"
        )
    only = set(argv)
    results: list[tuple[str, int, str]] = []
    for approval in sorted(APPROVALS.glob("controller-probe-*.json")):
        stem = approval.stem
        if only and stem not in only:
            continue
        request = json.loads(approval.read_text())
        condition = request["authorization"]["knowledge_condition"]
        run_root = RUNS / stem
        if run_root.exists():
            print(f"skip {stem}: already run", flush=True)
            continue
        if results:
            print(f"pausing {COOLDOWN_SECONDS}s for the provider rate limit", flush=True)
            time.sleep(COOLDOWN_SECONDS)
        print(f"\n=== {stem} ({condition})", flush=True)
        completed = subprocess.run(
            [
                sys.executable, str(RUNNER),
                *BUDGETS,
                "--knowledge-condition", condition,
                "--run-root", str(run_root),
                "--approval-request", str(approval),
                "--confirm-live-run", request["required_confirmation"],
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        note = "no summary"
        summary = run_root / "summary.json"
        if summary.is_file():
            record = json.loads(summary.read_text())
            note = (
                f"actions={record.get('actions')} "
                f"levels={record.get('levels_completed')} "
                f"stop={record.get('stop_reason')}"
            )
        else:
            print(completed.stderr[-2000:], flush=True)
        print(f"--- {stem}: {note} exit={completed.returncode}", flush=True)
        results.append((stem, completed.returncode, note))
    print("\n=== trials complete")
    for stem, code, note in results:
        print(f"  {stem}: exit {code} | {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
