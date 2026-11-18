"""Regenerate the unlaunched q6 candidate, or verify it without changing bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
Q6 = ROOT / "analysis/local-probes/20260905T-stall-authority-q6"
SHARED = (
    "solver_agent.py",
    "host_contract.py",
    "stall_contract.py",
    "integrity_guard.py",
)
FILES = (
    "nooa_worker.py",
    "headless.py",
    "local_solver.py",
    "stub_agent.py",
    "nooa_public_probe.py",
    "chassis/arc_llm.py",
    "chassis/skills/interactive-game-solver/SKILL.md",
    *(f"chassis/{name}" for name in SHARED),
)
SOURCES = (
    # The outer runtime is embedded beside the join rather than inside its zip,
    # so it is hashed here: without it the contract would not cover the
    # reconciler the run depends on.
    "deployment/kaggle/q6-nooa-candidate/exact_request_outer_runtime.py",
    "deployment/kaggle/q6-nooa-candidate/build_notebook.py",
    "deployment/kaggle/q6-nooa-candidate/freeze_candidate.py",
    "deployment/kaggle/agent/my_agent.py",
    "deployment/kaggle/runtime/local_solver.py",
    "deployment/kaggle/runtime/run_official_agent.py",
    "analysis/local-probes/20260905T-stall-authority-q6/score_q6.py",
    "analysis/local-probes/20260905T-stall-authority-q6/EXPERIMENT-CARD.md",
    "analysis/local-probes/20260905T-stall-authority-q6/patched-source/after/harness.py",
    "analysis/local-probes/20260905T-stall-authority-q6/patched-source/after/launcher.py",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest():
    files = {name: digest(HERE / name) for name in FILES}
    sources = {name: digest(ROOT / name) for name in SOURCES}
    identity = hashlib.sha256(
        json.dumps({"files": files, "sources": sources}, sort_keys=True).encode()
    ).hexdigest()
    return {
        "schema": "q6-nooa-candidate-v1",
        "identity": "q6-stall-authority",
        "base": "q3c-mdfiles-after",
        "contract_sha256": identity,
        "files": files,
        "sources": sources,
    }


def verify():
    actual = manifest()
    expected = json.loads((HERE / "CHASSIS.json").read_text())
    if expected != actual:
        raise RuntimeError(
            "candidate drift: regenerate an unlaunched candidate with freeze_candidate.py"
        )
    for name in SHARED:
        if (HERE / "chassis" / name).read_bytes() != (
            Q6 / "patched-source/after" / name
        ).read_bytes():
            raise RuntimeError(f"candidate differs from shared source: {name}")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        for name in SHARED:
            (HERE / "chassis" / name).write_bytes(
                (Q6 / "patched-source/after" / name).read_bytes()
            )
        (HERE / "CHASSIS.json").write_text(json.dumps(manifest(), indent=2) + "\n")
    print(verify()["contract_sha256"])


if __name__ == "__main__":
    main()
