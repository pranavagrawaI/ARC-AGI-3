"""PROTOTYPE: prepare the three no-spend known-world approval requests."""

from __future__ import annotations

import sys
from pathlib import Path

TREATMENT_ROOT = Path(__file__).resolve().parents[2]
if str(TREATMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(TREATMENT_ROOT))

import run_live_probe


def main() -> int:
    for condition in ("rules", "source", "executable"):
        stem = f"known-world-{condition}-l1-44-v1"
        code = run_live_probe.main(
            [
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
                "--run-root",
                str(TREATMENT_ROOT / "artifacts" / "runs" / stem),
                "--prepare-approval-request",
                str(TREATMENT_ROOT / "artifacts" / "approvals" / f"{stem}.json"),
                "--arm", "control",
                "--provider-route", "pinned",
                "--knowledge-condition", condition,
            ]
        )
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
