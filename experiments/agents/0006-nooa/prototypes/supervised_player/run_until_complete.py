"""PROTOTYPE: run the supervised 44-action condition until one attempt completes.

    python .../supervised_player/run_until_complete.py prepare [N] [FIRST]

    set -a && . ./.env && set +a
    uv run --python 3.14 python .../supervised_player/run_until_complete.py run [COOLDOWN]

`prepare` spends nothing. `run` spends, one attempt at a time, and stops at the
first attempt that reaches its action budget or clears level 1.

Why this exists. The pinned ModelRun pool for `google/gemma-4-31b-it` answers
429 in bursts, and a 429 is classified UNCERTAIN by the frozen transport
(`nooa-runtime/llm.py`), which never retries -- the request may already have
been served. `--transport-max-retries` covers connect errors only and never
engages. So a single 429 anywhere in a ~100-crossing run ends it, and no
in-run setting can change that.

What the recorded runs show is that the pool has good windows and bad ones,
not a uniform failure rate:

    2026-08-03 08:06-11:24   4 runs, 50/98/67/111 Gemma calls,  0 rate limits
    2026-08-03 14:36-15:03   7 runs, every one killed by a 429
    2026-08-04 08:30-08:37   2 runs, both killed at the third call

111 consecutive calls were served in one good window, which is more than a
44-action supervised run needs. The lever is therefore patience, not budget:
wait out a bad window rather than hammering it. Seven minutes was not enough
this morning, so the default cooldown is twenty.

Each failed attempt is cheap -- the two attempts on 2026-08-04 spent about
three cents between them, because a 429 arrives before most of the work. A
completed attempt should cost roughly $0.6-1.0. Budget the whole loop at about
a dollar plus a few cents per failure.
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

import run_live_probe

APPROVALS = TREATMENT_ROOT / "artifacts" / "approvals"
RUNS = TREATMENT_ROOT / "artifacts" / "runs"
RUNNER = TREATMENT_ROOT / "run_live_probe.py"
CONDITION = "rules_supervised"
STEM = "supervised-probe-rules_supervised-l1-44-v{trial}"
ATTEMPTS = 6
# Twenty minutes. Two attempts seven minutes apart both died in the same bad
# window, and the good windows in the record last hours, so a short pause only
# buys another dead artifact.
COOLDOWN_SECONDS = 1200
# A run that reached one of these ends the loop. Everything else is an abort.
COMPLETED = {"action_budget", "level_limit"}
# Identical to the two attempts already recorded, so a completed run is
# comparable to them and to the pinned 44-action runs.
BUDGETS = [
    "--game", "ls20",
    "--mode", "offline",
    "--environment-seed", "0",
    "--model", "openai/google/gemma-4-31b-it",
    "--wire-model", "google/gemma-4-31b-it",
    "--api-base", "https://openrouter.ai/api/v1",
    "--max-output-tokens", "16384",
    "--reasoning-effort", "medium",
    "--request-timeout-seconds", "180",
    "--max-actions", "44",
    "--max-model-calls", "400",
    "--max-reported-tokens", "8388608",
    "--max-model-rounds-per-action", "8",
    "--max-decision-repairs", "2",
    "--transport-max-retries", "2",
    "--workspace-timeout-seconds", "20",
    "--arm", "treatment",
    "--provider-route", "pinned",
    "--knowledge-condition", CONDITION,
]


def prepare(attempts: int, first: int) -> int:
    prepared: list[str] = []
    for trial in range(first, first + attempts):
        stem = STEM.format(trial=trial)
        approval = APPROVALS / f"{stem}.json"
        run_root = RUNS / stem
        if approval.exists() or run_root.exists():
            print(f"skip {stem}: already prepared or run")
            continue
        code = run_live_probe.main(
            [*BUDGETS, "--run-root", str(run_root),
             "--prepare-approval-request", str(approval)]
        )
        if code:
            return code
        prepared.append(stem)
    print(f"\nprepared {len(prepared)} approval request(s):")
    for stem in prepared:
        print(f"  {stem}")
    return 0


def run(cooldown: int) -> int:
    if not os.environ.get("ARC_LLM_API_KEY"):
        raise SystemExit(
            "ARC_LLM_API_KEY absent. It is defined in .env as ${OPENROUTER_API_KEY}, "
            "which only expands when the file is sourced by a shell: "
            "set -a && . ./.env && set +a"
        )
    attempted = 0
    for approval in sorted(APPROVALS.glob("supervised-probe-rules_supervised-l1-44-v*.json")):
        stem = approval.stem
        run_root = RUNS / stem
        if run_root.exists():
            print(f"skip {stem}: already run", flush=True)
            continue
        if attempted:
            print(f"pausing {cooldown}s to clear the provider window", flush=True)
            time.sleep(cooldown)
        request = json.loads(approval.read_text())
        print(f"\n=== {stem}", flush=True)
        completed = subprocess.run(
            [
                sys.executable, str(RUNNER),
                *BUDGETS,
                "--run-root", str(run_root),
                "--approval-request", str(approval),
                "--confirm-live-run", request["required_confirmation"],
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        attempted += 1
        summary_path = run_root / "summary.json"
        if not summary_path.is_file():
            print(completed.stderr[-2000:], flush=True)
            print(f"--- {stem}: no summary, exit={completed.returncode}", flush=True)
            continue
        record = json.loads(summary_path.read_text())
        stop = record.get("stop_reason")
        print(
            f"--- {stem}: actions={record.get('actions')} "
            f"levels={record.get('levels_completed')} stop={stop} "
            f"exit={completed.returncode}",
            flush=True,
        )
        if stop in COMPLETED:
            print(f"\n=== {stem} completed on {stop}; stopping the loop")
            return 0
        for error in record.get("agent_errors") or []:
            print(f"    {error.get('type')}: {error.get('message')}")
    print(f"\n=== no attempt completed after {attempted} run(s)")
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in {"prepare", "run"}:
        raise SystemExit("usage: run_until_complete.py {prepare [N] [FIRST] | run [COOLDOWN]}")
    if argv[0] == "prepare":
        attempts = int(argv[1]) if len(argv) > 1 else ATTEMPTS
        first = int(argv[2]) if len(argv) > 2 else 3
        return prepare(attempts, first)
    return run(int(argv[1]) if len(argv) > 1 else COOLDOWN_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
