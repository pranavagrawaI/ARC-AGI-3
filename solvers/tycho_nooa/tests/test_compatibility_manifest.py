from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tycho_nooa.cognitive_parity import cognitive_return_schema_sha256
from tycho_nooa.compatibility_manifest import (
    CompatibilityExperimentManifest,
    load_compatibility_experiment_manifest,
    load_gate_manifest,
)
from tycho_nooa.compatibility_return import retrieval_return_schema_sha256
from tycho_nooa.parity_compatibility import CompatibilityEvidenceCapability

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
GATE_MANIFEST = REPOSITORY_ROOT / "analysis/parity-compat-v9/gate-manifest.json"
EXPERIMENT = REPOSITORY_ROOT / "analysis/parity-compat-v9/experiment.json"
LEGACY_EXPERIMENT = REPOSITORY_ROOT / "analysis/parity-compat-v2/experiment.json"


def test_frozen_gate_manifest_loads_every_validated_ladder_role() -> None:
    manifest, ladder = load_gate_manifest(
        GATE_MANIFEST, repository_root=REPOSITORY_ROOT
    )

    assert manifest.ladder_id == "parity-compat-v9"
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


def test_experiment_is_fully_committed_but_paid_execution_is_disabled() -> None:
    manifest, ladder = load_compatibility_experiment_manifest(
        EXPERIMENT, repository_root=REPOSITORY_ROOT
    )

    assert manifest.runtime_commit == "2913039"
    assert manifest.package_commit == "2913039"
    assert manifest.caps.calls_per_arm_gate == 3
    assert manifest.caps.calls_total == 30
    assert manifest.caps.input_tokens_per_call == 65536
    assert manifest.return_contract == "retrieval_flat"
    assert manifest.model_route.model is None
    assert ladder.termination.task.gate_id == "g0-termination"
    with pytest.raises(RuntimeError, match="execution is disabled"):
        manifest.require_execution_authorized()


def test_authorization_requires_complete_route_and_cost_cap() -> None:
    raw = json.loads(EXPERIMENT.read_text())
    raw["execution_authorized"] = True

    with pytest.raises(ValidationError, match="complete model route"):
        CompatibilityExperimentManifest.model_validate(raw)


def test_experiment_detects_runner_commitment_drift(tmp_path: Path) -> None:
    raw = json.loads(EXPERIMENT.read_text())
    raw["ladder_runner"]["sha256"] = "sha256:" + "0" * 64
    forged = tmp_path / "experiment.json"
    forged.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="file commitment mismatch"):
        load_compatibility_experiment_manifest(
            forged, repository_root=REPOSITORY_ROOT
        )


def test_schema_v2_requires_and_checks_the_shared_return_contract() -> None:
    raw = json.loads(LEGACY_EXPERIMENT.read_text())
    raw["schema_version"] = 2

    with pytest.raises(ValidationError, match="bind both controllers"):
        CompatibilityExperimentManifest.model_validate(raw)

    raw.update(
        native_controller=raw["gate_executor"],
        codeact_controller=raw["gate_executor"],
        return_schema_sha256=cognitive_return_schema_sha256(),
    )
    manifest = CompatibilityExperimentManifest.model_validate(raw)

    assert manifest.return_schema_sha256 == cognitive_return_schema_sha256()


def test_schema_v3_requires_projection_and_executor_commitments() -> None:
    raw = json.loads(LEGACY_EXPERIMENT.read_text())
    raw["schema_version"] = 3

    with pytest.raises(ValidationError, match="projection, and executor"):
        CompatibilityExperimentManifest.model_validate(raw)


def test_schema_v4_requires_the_exec_worker_commitment() -> None:
    raw = json.loads(LEGACY_EXPERIMENT.read_text())
    raw.update(
        schema_version=4,
        native_controller=raw["gate_executor"],
        codeact_controller=raw["gate_executor"],
        return_schema_sha256=cognitive_return_schema_sha256(),
        execution_projection=raw["gate_executor"],
        sterile_executor=raw["gate_executor"],
    )

    with pytest.raises(ValidationError, match="sterile worker"):
        CompatibilityExperimentManifest.model_validate(raw)

    raw["sterile_worker"] = {
        "path": "solvers/tycho_nooa/src/tycho_nooa/_sterile_worker.py",
        "sha256": "sha256:" + "0" * 64,
    }
    manifest = CompatibilityExperimentManifest.model_validate(raw)

    assert manifest.sterile_worker is not None


def test_schema_v5_requires_and_checks_flat_retrieval_contract() -> None:
    raw = json.loads(LEGACY_EXPERIMENT.read_text())
    raw.update(
        schema_version=5,
        native_controller=raw["gate_executor"],
        codeact_controller=raw["gate_executor"],
        return_schema_sha256=retrieval_return_schema_sha256(),
        execution_projection=raw["gate_executor"],
        sterile_executor=raw["gate_executor"],
        sterile_worker=raw["gate_executor"],
    )

    with pytest.raises(ValidationError, match="flat retrieval"):
        CompatibilityExperimentManifest.model_validate(raw)

    raw["return_contract"] = "retrieval_flat"

    with pytest.raises(ValidationError, match="contract and module"):
        CompatibilityExperimentManifest.model_validate(raw)

    raw["return_contract_module"] = raw["gate_executor"]
    manifest = CompatibilityExperimentManifest.model_validate(raw)

    assert manifest.return_contract == "retrieval_flat"
    assert manifest.return_schema_sha256 == retrieval_return_schema_sha256()
