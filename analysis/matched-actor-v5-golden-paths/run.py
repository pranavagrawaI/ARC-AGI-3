"""Execute the zero-provider golden paths for the five incomplete V5 packets."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path

from tycho_nooa.matched_actor_golden_paths import prove_golden_paths
from tycho_nooa.repaired_checkpoint import load_repaired_pilot_sample

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PACKAGE_ROOT / "proof"
MANIFEST_PATH = PACKAGE_ROOT / "manifest.json"


async def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    _, packets = load_repaired_pilot_sample(
        REPOSITORY_ROOT / "analysis/matched-actor-v1-pilot/sample.json",
        REPOSITORY_ROOT / "analysis/matched-actor-v5-pilot/checkpoints.json",
        repository_root=REPOSITORY_ROOT,
    )
    proof = await prove_golden_paths(
        packets,
        repository_root=REPOSITORY_ROOT,
        destination=OUTPUT_ROOT,
    )
    if not proof.all_runtime_paths_correct:
        raise SystemExit("a runtime golden path failed")
    _write_manifest()


def _write_manifest() -> None:
    inputs = (
        "analysis/matched-actor-v1-pilot/sample.json",
        "analysis/matched-actor-v5-pilot/checkpoints.json",
        (
            "analysis/matched-actor-v5-pilot/runs/"
            "20260824T215146Z-deepseek-v4-flash-deepinfra/pilot-report.json"
        ),
        (
            "analysis/matched-actor-v5-pilot/runs/"
            "20260824T215146Z-deepseek-v4-flash-deepinfra/"
            "authorization-events.jsonl"
        ),
        "analysis/matched-actor-v5-golden-paths/run.py",
        "solvers/tycho_nooa/src/tycho_nooa/matched_actor_golden_paths.py",
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": "matched-actor-v5-golden-paths",
        "provider_calls": 0,
        "arc_actions": 0,
        "inputs": [_commitment(REPOSITORY_ROOT / path) for path in inputs],
        "proof_files": [
            _commitment(path)
            for path in sorted(OUTPUT_ROOT.rglob("*"))
            if path.is_file()
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _commitment(path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(REPOSITORY_ROOT)),
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    asyncio.run(main())
