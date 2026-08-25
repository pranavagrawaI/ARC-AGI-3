"""Freeze the zero-provider remote-history architecture proof."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path

from tycho_nooa.remote_history_stress import prove_remote_history_stress

repo = Path(__file__).resolve().parents[2]
base = Path(__file__).resolve().parent
proof_root = base / "proof"


async def main() -> None:
    if proof_root.exists():
        shutil.rmtree(proof_root)
    proof = await prove_remote_history_stress(
        repository_root=repo, destination=proof_root
    )
    if not proof.exact_source_disclosures_identical:
        raise SystemExit("remote-history source evidence diverged")
    if not all(arm.answer_correct and arm.facts_verified for arm in proof.arms):
        raise SystemExit("a remote-history golden path failed")
    _write_manifest()


def _write_manifest() -> None:
    inputs = (
        "analysis/matched-actor-v1-pilot/sample.json",
        "analysis/matched-actor-v5-pilot/checkpoints.json",
        "analysis/matched-actor-contract-repair-v1/manifest.json",
        "analysis/remote-history-stress-v1/EXPERIMENT-CARD.md",
        "analysis/remote-history-stress-v1/PROTOCOL.md",
        "analysis/remote-history-stress-v1/run.py",
        "solvers/tycho_nooa/src/tycho_nooa/context_evidence_v3.py",
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_context_v3.py",
        "solvers/tycho_nooa/src/tycho_nooa/remote_history_stress.py",
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": "remote-history-stress-v1",
        "provider_calls": 0,
        "arc_actions": 0,
        "inputs": [_commitment(repo / path) for path in inputs],
        "proof_files": [
            _commitment(path)
            for path in sorted(proof_root.rglob("*"))
            if path.is_file()
        ],
    }
    (base / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _commitment(path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(repo)),
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    asyncio.run(main())
