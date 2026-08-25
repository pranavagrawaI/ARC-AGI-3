from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tycho_nooa.remote_history_stress import prove_remote_history_stress

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE = REPOSITORY_ROOT / "analysis/remote-history-stress-v1"


@pytest.mark.asyncio
async def test_three_remote_history_paths_share_evidence_and_differ_in_control(
    tmp_path: Path,
) -> None:
    proof = await prove_remote_history_stress(
        repository_root=REPOSITORY_ROOT,
        destination=tmp_path / "proof",
    )

    assert proof.provider_calls == 0
    assert proof.arc_actions == 0
    assert proof.exact_source_disclosures_identical is True
    assert proof.packet_specific_helper_present is False
    assert {
        arm.arm: (
            arm.provider_calls,
            arm.model_evidence_acquisitions,
            arm.source_pages,
        )
        for arm in proof.arms
    } == {
        "native_paged": (32, 30, 30),
        "codeact": (2, 30, 30),
        "native_acquire": (3, 1, 30),
    }
    assert all(arm.answer_correct and arm.facts_verified for arm in proof.arms)


def test_remote_history_manifest_binds_inputs_and_proof() -> None:
    manifest = json.loads((PACKAGE / "manifest.json").read_text())

    assert manifest["provider_calls"] == 0
    assert manifest["arc_actions"] == 0
    for record in (*manifest["inputs"], *manifest["proof_files"]):
        path = REPOSITORY_ROOT / record["path"]
        assert record["sha256"] == "sha256:" + hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
