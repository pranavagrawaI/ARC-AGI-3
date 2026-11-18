#!/usr/bin/env python3
"""Re-run final, network-free checks for the closed public run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    audit = json.loads((HERE / "AUDIT.json").read_text())
    pre = json.loads((HERE / "PRELAUNCH.json").read_text())
    quota = json.loads((HERE / "postrun-quota.json").read_text())
    gpu = next(row for row in quota if row["resource"] == "GPU")
    gpu_delta = float(gpu["used"].removesuffix("h")) - 1.10
    env_files = [str(path) for path in (HERE / "raw").rglob(".env")]
    paths = {
        "card_sha256": HERE / "EXPERIMENT-CARD.md",
        "notebook_sha256": ROOT / "deployment/kaggle/submission-20260912-v15/submission.ipynb",
        "metadata_sha256": ROOT / "deployment/kaggle/submission-20260912-v15/kernel-metadata.json",
        "build_sha256": ROOT / "deployment/kaggle/submission-20260912-v15/build.py",
        "play_sha256": ROOT / "deployment/kaggle/submission-20260912-v15/play.py",
    }
    checks = {
        "kernel_complete": "KernelWorkerStatus.COMPLETE" in (HERE / "final-status.txt").read_text(),
        "offline_audit_valid": audit.get("valid") is True,
        "routine_reasoning_zero": audit.get("routine_reasoning_tokens") == 0,
        "no_sparse_call_observed": audit.get("deep_investigations") == [],
        "at_least_five_executed_actions": int(audit.get("launcher_executed_actions") or 0) >= 5,
        "zero_failed_requests": audit.get("request_lifecycle", {}).get("failed") == 0,
        "gpu_quota_within_cap": gpu_delta <= float(pre["cap_gpu_accounted_hours"]),
        "no_retained_env": not env_files,
        "prelaunch_hashes_still_match": all(digest(path) == pre[key] for key, path in paths.items()),
        "one_push_no_retry": pre.get("push_cap") == 1 and pre.get("retry_cap") == 0,
    }
    output = {
        "schema": "ls20-sparse-reasoning-public-v1-verification",
        "checks": checks,
        "passed": all(checks.values()),
        "gpu_delta_hours": round(gpu_delta, 2),
        "env_files": env_files,
    }
    (HERE / "VERIFIED.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
