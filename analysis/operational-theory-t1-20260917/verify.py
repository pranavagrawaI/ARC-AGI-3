"""Deterministic local verification; loopback fixtures only, no provider credentials."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from bootstrap import HERE, ROOT, RUNTIME
from identity import files, sha, verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_before = {str(p.relative_to(ROOT)): sha(p) for p in files()}
    manifest_sha = verify() if (HERE / "MANIFEST.json").exists() else None
    python = ROOT / "deployment/kaggle/.venv/bin/python"
    ruff = ROOT / "deployment/kaggle/.venv/bin/ruff"
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        ARC_DISABLE_DOTENV="1",
    )
    for name in list(env):
        if "API_KEY" in name or "TOKEN" in name or name.startswith("ARC_LLM_"):
            env.pop(name)
    prefix = [
        "unshare",
        "--user",
        "--map-auto",
        "--map-root-user",
        str(python),
        "-B",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    prior = ROOT / "analysis/openrouter-constructor-fixed-20260916T214433Z"
    commands = {
        "new": prefix
        + [str(HERE / "test_operational.py"), "--basetemp=" + str(output / "new-tmp")],
        "transport-baseline": prefix
        + [
            str(prior / "test_pair.py"),
            str(prior / "test_work_item.py"),
            "--basetemp=" + str(output / "transport-tmp"),
        ],
        "actor-baseline": prefix
        + [
            str(ROOT / "analysis/actor-critic-construction-integrated/tests"),
            "--basetemp=" + str(output / "actor-tmp"),
        ],
        "current-runnable": prefix
        + [
            str(RUNTIME),
            "--ignore=" + str(RUNTIME / "test_assessment.py"),
            "--ignore=" + str(RUNTIME / "test_packaging.py"),
            "--basetemp=" + str(output / "current-tmp"),
        ],
        "lint": [str(ruff), "check", *map(str, sorted(HERE.glob("*.py")))],
        "format": [
            str(ruff),
            "format",
            "--check",
            *map(str, sorted(HERE.glob("*.py"))),
        ],
    }
    results = {}
    for label, command in commands.items():
        with (output / (label + ".txt")).open("wb") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=180,
            )
        results[label] = {
            "command": command,
            "exit_code": result.returncode,
            "log_sha256": sha(output / (label + ".txt")),
        }
        print(label, result.returncode, flush=True)
    if manifest_sha and verify() != manifest_sha:
        raise RuntimeError("identity changed during checks")
    if source_before != {str(p.relative_to(ROOT)): sha(p) for p in files()}:
        raise RuntimeError("source changed during checks")
    record = {
        "source_files": source_before,
        "manifest_sha256": manifest_sha,
        "provider_calls": 0,
        "scope": "local loopback and namespace fixtures only",
        "checks": results,
        "known_baseline_collection_errors": [
            "runtime/test_assessment.py: missing prepare_candidate",
            "runtime/test_packaging.py: missing prepare_candidate",
        ],
    }
    (output / "CHECKS.json").write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n"
    )
    raise SystemExit(0 if all(r["exit_code"] == 0 for r in results.values()) else 1)


if __name__ == "__main__":
    main()
