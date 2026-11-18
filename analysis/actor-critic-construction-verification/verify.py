"""Run an integration check and bind its real output to source hashes.

Usage: python3 verify.py -- <test command and arguments>
Writes only into this coordinator-owned verification directory.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANDIDATE = ROOT / "analysis/actor-critic-construction-integrated"


def hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".md", ".toml"}
        and not any(part in {"__pycache__", ".pytest_cache", "evidence", "runs"} for part in path.relative_to(root).parts)
    }


def main() -> int:
    command = sys.argv[1:]
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("provide a command after --")
    before = hashes(CANDIDATE)
    if not before:
        raise SystemExit("candidate has no source artifacts")
    donor_before = hashes(ROOT / "analysis/construction-policy-live-20260914T093351Z")
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    print(result.stdout, end="")
    after = hashes(CANDIDATE)
    baseline = json.loads((HERE / "incumbent-baseline-sha256.json").read_text())
    changed_incumbent = [name for name, expected in baseline.items()
                         if not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected]
    report = {
        "command": command,
        "returncode": result.returncode,
        "seconds": round(time.monotonic() - started, 3),
        "candidate_sources": before,
        "construction_donor_sources": donor_before,
        "construction_donor_changed_during_check": donor_before != hashes(ROOT / "analysis/construction-policy-live-20260914T093351Z"),
        "candidate_changed_during_check": before != after,
        "incumbent_changed_from_baseline": changed_incumbent,
        "scope": "Local execution only; inspect test transport before making model or gameplay claims.",
    }
    (HERE / "integration-check.stdout").write_text(result.stdout)
    (HERE / "integration-check.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"candidate_sources", "construction_donor_sources"}}, indent=2))
    return result.returncode or int(before != after or bool(changed_incumbent) or report["construction_donor_changed_during_check"])


if __name__ == "__main__":
    raise SystemExit(main())
