#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from constructor import construction_messages
from real_trace import load_case
from sandbox_eval import evaluate_candidate

HERE = Path(__file__).resolve().parent
REFERENCE = Path("/home/pranav/projects/arc-agi-3-reference/model_construction_probe")


def run(command: list[str], cwd: Path) -> dict:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> None:
    checks = []
    checks.append({"name": "reference_sha256", **run(["sha256sum", "-c", "SHA256SUMS"], REFERENCE)})
    checks.append({
        "name": "reference_tests",
        **run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], REFERENCE),
    })
    checks.append({
        "name": "integration_tests",
        **run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], HERE),
    })
    checks.append({"name": "dry_run", **run([sys.executable, "run.py"], HERE)})

    case = load_case()
    incumbent = evaluate_candidate(case.legacy_incumbent_source(), case, include_suffix=True)
    witness_source = (HERE / "tests/witness_candidate.py").read_text()
    witness = evaluate_candidate(witness_source, case, include_suffix=True)
    prompt = json.dumps(construction_messages(case, None), sort_keys=True)
    semantic = {
        "incumbent_visible_passed": incumbent["visible"]["status"] == "passed",
        "incumbent_suffix_compatible_depth": incumbent["suffix"]["compatible_depth"] == 2,
        "incumbent_suffix_first_contradiction": incumbent["suffix"].get("first_contradiction_step") == 8,
        "witness_visible_passed": witness["visible"]["status"] == "passed",
        "witness_suffix_passed": witness["suffix"]["status"] == "passed",
        "prompt_excludes_suffix_hash": case.sealed_suffix_sha256 not in prompt,
        "prompt_excludes_suffix_steps": all(
            token not in prompt for token in ('"step": 6', '"step":6', '"step": 7', '"step":7', '"step": 8', '"step":8')
        ),
        "incumbent_digest_reconciles": hashlib.sha256(case.incumbent_source.encode()).hexdigest() == case.incumbent_sha256,
    }
    passed = all(check["exit_code"] == 0 for check in checks) and all(semantic.values())
    record = {
        "schema": "model-construction-prelaunch-v1",
        "passed": passed,
        "reference_path": str(REFERENCE),
        "reference_manifest_sha256": hashlib.sha256((REFERENCE / "SHA256SUMS").read_bytes()).hexdigest(),
        "case": {
            "frontier": case.frontier,
            "incumbent_sha256": case.incumbent_sha256,
            "sealed_suffix_sha256": case.sealed_suffix_sha256,
        },
        "semantic_checks": semantic,
        "process_checks": checks,
    }
    (HERE / "PRELAUNCH.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "checks": len(checks) + len(semantic)}, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
