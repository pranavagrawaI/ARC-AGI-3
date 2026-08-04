"""Run the corrected control through the ordinary offline artifact path."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[5]
for path in (EXPERIMENT_ROOT, REPO / "experiments" / "agents" / "0002-llm-repl-agent"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from uniform_action_control import corrected_control_spec
from p9.freeze import verify as verify_frozen_control
from p9.run_arm import build_parser, run_one, spec_from_args


def main() -> int:
    before = verify_frozen_control()
    if before["drift"]:
        raise RuntimeError(f"old frozen control drifted before verification: {before['drift']}")

    args = build_parser().parse_args(["--offline", "--max-actions", "2"])
    spec = corrected_control_spec(spec_from_args(args))
    with TemporaryDirectory(prefix="0004-uniform-actions-") as scratch:
        run_dir = Path(scratch) / "run"
        manifest = run_one(spec, run_dir, args)
        required_artifacts = {
            "manifest.json",
            "process.jsonl",
            "trace.jsonl",
            "friction.json",
            "process-annotation.json",
            "workspace",
        }
        missing = sorted(
            name for name in required_artifacts if not (run_dir / name).exists()
        )
        result = {
            "schema": "0004-uniform-action-interface-offline-verification-v1",
            "status": manifest["status"],
            "arm_id": manifest["arm_id"],
            "actions": manifest["actions"],
            "stop_reason": manifest["stop_reason"],
            "model": manifest["environment"]["model"],
            "treatment_capabilities": manifest["treatment_capabilities"],
            "missing_artifacts": missing,
            "old_control_drift_before": before["drift"],
            "old_control_drift_after": verify_frozen_control()["drift"],
        }

    expected = {
        "status": "complete",
        "arm_id": "0006-uniform-actions-v1",
        "actions": 2,
        "stop_reason": "terminal",
        "model": "fixture-task-policy",
        "treatment_capabilities": [],
        "missing_artifacts": [],
        "old_control_drift_before": [],
        "old_control_drift_after": [],
    }
    for key, value in expected.items():
        if result[key] != value:
            raise AssertionError(f"{key}: expected {value!r}, got {result[key]!r}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
