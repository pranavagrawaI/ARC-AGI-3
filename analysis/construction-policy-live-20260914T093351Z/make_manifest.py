#!/usr/bin/env python3
"""Write the immutable r5-derived review manifest."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PACKAGE = REPO / "deployment/kaggle" / HERE.name
ANALYSIS_FILES = [
    "AUDIT-PLAN.md", "EVIDENCE-ALLOWLIST.json", "EXPERIMENT-CARD.md", "R5-REVIEW-PREREQUISITE.json", "SERVER-REFERENCE.json",
    "accelerator.py", "audit_run.py", "case.json", "collect_readonly.py", "constructor.py", "containment_base.py", "inspection_runner.py",
    "integration.py", "integration_boundary.py", "local_transport.py", "make_manifest.py", "model-metadata.json", "prelaunch_check.py",
    "PROVIDER-SELECTION.json", "qualified_runner.py", "real_trace.py", "request_contract.py", "tests/test_live_controls.py",
    "tests/test_default_branch_lifecycle.py", "tests/test_integration.py", "tests/test_local_wire.py", "tests/notebook_local_mock_runner.py", "tests/run_notebook_mode.py",
]
PACKAGE_FILES = ["BUILD.json", "construction.ipynb", "kernel-metadata.json", "launch_once.py", "reconcile_build.py", "remote_entry.py"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files = {}
    for name in ANALYSIS_FILES:
        files["analysis/" + name] = sha(HERE / name)
    for name in PACKAGE_FILES:
        files["package/" + name] = sha(PACKAGE / name)
    manifest = {
        "schema": "construction-policy-live-manifest-v2",
        "identity": HERE.name,
        "reviewed_baseline": {
            "identity": "construction-policy-integration-20260913T221412Z-r5",
            "manifest_sha256": "493018986c25f14472fe6c10bd9c2c888cde4cafa8388d7425495687eb68d440",
            "notebook_sha256": "50d345ec00f2a5d089e19b4ee74412576f54a3902ba67859bd4bf9258d02dbd6",
            "build_sha256": "7135749d2a22bd89299a0306b7c6fb94a08efdba5eac1e6c5ba6e1dd66f2cf4d",
        },
        "files": files,
        "mutable_exclusions": ["AUTHORIZATION.json", "FINAL-REVIEW-REQUEST.json", "REVIEW-RESOLVED.json", "evidence/**", "LAUNCH-CONSUMED.json"],
    }
    (HERE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha(HERE / "MANIFEST.json"), "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
