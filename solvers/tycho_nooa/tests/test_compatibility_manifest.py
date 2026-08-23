from __future__ import annotations

import json
from pathlib import Path

import pytest

from tycho_nooa.compatibility_manifest import load_gate_manifest
from tycho_nooa.parity_compatibility import CompatibilityEvidenceCapability

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
GATE_MANIFEST = REPOSITORY_ROOT / "analysis/parity-compat-v2/gate-manifest.json"


def test_frozen_gate_manifest_loads_every_validated_ladder_role() -> None:
    manifest, ladder = load_gate_manifest(
        GATE_MANIFEST, repository_root=REPOSITORY_ROOT
    )

    assert manifest.ladder_id == "parity-compat-v2"
    assert [gate.task.gate_id for gate in ladder.ordered_gates()] == [
        "g0-termination",
        "g1-metadata-query",
        "g2-representation-rle",
        "g2-representation-raw",
        "g3-easy-positive-rle",
        "g3-easy-positive-raw",
    ]
    assert ladder.termination.treatment_payload().keys() == {
        "task",
        "evidence_contract",
    }


def test_metadata_gate_exact_query_excludes_grids() -> None:
    _, ladder = load_gate_manifest(
        GATE_MANIFEST, repository_root=REPOSITORY_ROOT
    )
    capability = CompatibilityEvidenceCapability.from_gate(
        ladder.metadata_query, repository_root=REPOSITORY_ROOT
    )

    assert capability is not None
    result = capability.query(ladder.metadata_query.evidence.query_value())

    assert result["records"][2] == {
        "type": "transition_metadata",
        "level": 0,
        "turn": 3,
        "action": "LEFT",
        "row": None,
        "col": None,
    }
    assert all("prev" not in record and "next" not in record for record in result["records"])


def test_representation_alternatives_name_the_same_cell_values() -> None:
    _, ladder = load_gate_manifest(
        GATE_MANIFEST, repository_root=REPOSITORY_ROOT
    )
    raw_capability = CompatibilityEvidenceCapability.from_gate(
        ladder.representation_raw, repository_root=REPOSITORY_ROOT
    )
    rle_capability = CompatibilityEvidenceCapability.from_gate(
        ladder.representation_rle, repository_root=REPOSITORY_ROOT
    )

    assert raw_capability is not None
    assert rle_capability is not None
    raw = raw_capability.query(ladder.representation_raw.evidence.query_value())
    rle = rle_capability.query(ladder.representation_rle.evidence.query_value())

    assert raw["records"][0]["prev"][0][63] == 11
    assert raw["records"][0]["next"][0][63] == 5
    assert rle["records"][0]["prev"][0].endswith("1:b")
    assert rle["records"][0]["next"][0].endswith("1:5")


def test_gate_manifest_detects_packet_commitment_drift(tmp_path: Path) -> None:
    raw = json.loads(GATE_MANIFEST.read_text())
    raw["gates"][0]["file"]["sha256"] = "sha256:" + "0" * 64
    forged = tmp_path / "gate-manifest.json"
    forged.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="file commitment mismatch"):
        load_gate_manifest(forged, repository_root=REPOSITORY_ROOT)
