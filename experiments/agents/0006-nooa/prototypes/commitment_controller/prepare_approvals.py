"""PROTOTYPE: prepare the no-spend approval requests for the controller trials.

    python experiments/agents/0006-nooa/prototypes/commitment_controller/prepare_approvals.py [N]

Prepares 2N requests: N `rules_controller` trials and N matched `rules`
baselines, alternating so a provider-side bad patch cannot land entirely on one
condition. Both are the treatment arm, same tutorial, same budgets, same
harness, same seed. The only difference is the controller.

The matched baseline is prepared fresh rather than reused because there is not
currently a usable one: of the five recorded Gemma+rules treatment attempts,
three died on provider transport with zero actions and one exhausted its
deliberation rounds before acting. One trajectory is not a baseline.

This spends nothing. Each request still needs its own digest-bound confirmation
and an inherited ARC_LLM_API_KEY before it can run.
"""

from __future__ import annotations

import sys
from pathlib import Path

TREATMENT_ROOT = Path(__file__).resolve().parents[2]
if str(TREATMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(TREATMENT_ROOT))

import run_live_probe

TRIALS = 3
# Matched to the recorded Gemma+rules runs so the comparison is like-for-like.
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
]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    trials = int(argv[0]) if argv else TRIALS
    first = int(argv[1]) if len(argv) > 1 else 1
    prepared: list[str] = []
    for trial in range(first, first + trials):
        for condition in ("rules_controller", "rules"):
            stem = f"controller-probe-{condition}-l1-44-v{trial}"
            approval = TREATMENT_ROOT / "artifacts" / "approvals" / f"{stem}.json"
            run_root = TREATMENT_ROOT / "artifacts" / "runs" / stem
            if approval.exists() or run_root.exists():
                print(f"skip {stem}: already prepared or run")
                continue
            code = run_live_probe.main(
                [
                    *BUDGETS,
                    "--knowledge-condition", condition,
                    "--run-root", str(run_root),
                    "--prepare-approval-request", str(approval),
                ]
            )
            if code:
                return code
            prepared.append(stem)
    print(f"\nprepared {len(prepared)} approval request(s):")
    for stem in prepared:
        print(f"  {stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
