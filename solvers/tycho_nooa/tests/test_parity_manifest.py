from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tycho_nooa.parity_manifest import (
    ParityExperimentManifest,
    load_experiment_manifest,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPOSITORY_ROOT / "analysis/codeact-cognitive-parity/v1/experiment.json"


def test_v1_manifest_is_committed_but_paid_execution_is_disabled() -> None:
    manifest = load_experiment_manifest(MANIFEST, repository_root=REPOSITORY_ROOT)

    assert manifest.package_commit == "34ec8b4"
    assert manifest.caps.sessions == 24
    assert manifest.caps.calls_total == 192
    assert manifest.model_route.model is None
    assert manifest.model_route.reported_model is None
    with pytest.raises(RuntimeError, match="execution is disabled"):
        manifest.require_execution_authorized()


def test_authorization_requires_route_and_cost_before_execution() -> None:
    raw = json.loads(MANIFEST.read_text())
    raw["execution_authorized"] = True

    with pytest.raises(ValidationError, match="complete model route"):
        ParityExperimentManifest.model_validate(raw)


def test_file_commitment_detects_protocol_drift(tmp_path: Path) -> None:
    raw = json.loads(MANIFEST.read_text())
    raw["protocol"]["sha256"] = "sha256:" + "0" * 64
    forged = tmp_path / "experiment.json"
    forged.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="file commitment mismatch"):
        load_experiment_manifest(forged, repository_root=REPOSITORY_ROOT)
