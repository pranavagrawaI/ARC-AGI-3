from __future__ import annotations

import json
import shutil
from pathlib import Path

from tycho_nooa.repaired_checkpoint import (
    load_repaired_pilot_sample,
    semantic_tree_sha256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SAMPLE = REPOSITORY_ROOT / "analysis/matched-actor-v1-pilot/sample.json"
CHECKPOINTS = REPOSITORY_ROOT / "analysis/matched-actor-v5-pilot/checkpoints.json"


def test_repaired_sample_loads_all_six_packets() -> None:
    sample, packets = load_repaired_pilot_sample(
        SAMPLE,
        CHECKPOINTS,
        repository_root=REPOSITORY_ROOT,
    )

    assert len(sample.packets) == len(packets) == 6
    assert [item.packet_id for item in sample.packets] == [
        packet.task.packet_id for packet in packets
    ]
    assert all("analysis/matched-actor-v5-pilot/checkpoints/" in packet.checkpoint_root for packet in packets)


def test_semantic_hash_survives_clean_checkout_without_python_cache(
    tmp_path: Path,
) -> None:
    index = json.loads(CHECKPOINTS.read_text())
    for position, record in enumerate(index["checkpoints"]):
        source = REPOSITORY_ROOT / record["frozen_checkpoint_root"]
        destination = tmp_path / f"checkpoint-{position}"
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )

        assert semantic_tree_sha256(source) == record["semantic_sha256"]
        assert semantic_tree_sha256(destination) == record["semantic_sha256"]
