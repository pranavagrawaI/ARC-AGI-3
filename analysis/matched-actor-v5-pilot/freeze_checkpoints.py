"""Snapshot the three pilot workspaces without interpreter-generated files."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from tycho_nooa.repaired_checkpoint import semantic_tree_sha256

repo = Path(__file__).resolve().parents[2]
base = Path(__file__).resolve().parent
snapshot_root = base / "checkpoints"
sources = {
    "symmetry-uptake-ar25": (
        "runs/tycho-nooa-symmetry-uptake-ar25-20260820T101248Z/workspace/ar25"
    ),
    "candidate-retention-turn18-ar25": (
        "analysis/candidate-retention/20260822-turn18/workspace/ar25"
    ),
    "level-start-ar25": (
        "runs/tycho-nooa-level-start-ar25-20260818T175557Z/workspace/ar25"
    ),
}


def main() -> None:
    snapshot_root.mkdir(exist_ok=True)
    records = []
    for name, source_text in sources.items():
        source = repo / source_text
        destination = snapshot_root / name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        records.append(
            {
                "source_checkpoint_root": source_text,
                "frozen_checkpoint_root": str(destination.relative_to(repo)),
                "semantic_sha256": semantic_tree_sha256(destination),
            }
        )
    value = {
        "schema_version": 1,
        "hash_policy": "exclude_python_bytecode_v1",
        "supersedes_field": "task.checkpoint_sha256",
        "checkpoints": records,
    }
    (base / "checkpoints.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
