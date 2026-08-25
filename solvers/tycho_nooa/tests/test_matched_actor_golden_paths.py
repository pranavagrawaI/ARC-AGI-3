from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tycho_nooa.matched_actor_golden_paths import prove_golden_paths
from tycho_nooa.repaired_checkpoint import load_repaired_pilot_sample

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.asyncio
async def test_five_answer_key_independent_golden_paths(tmp_path: Path) -> None:
    _, packets = load_repaired_pilot_sample(
        REPOSITORY_ROOT / "analysis/matched-actor-v1-pilot/sample.json",
        REPOSITORY_ROOT / "analysis/matched-actor-v5-pilot/checkpoints.json",
        repository_root=REPOSITORY_ROOT,
    )

    proof = await prove_golden_paths(
        packets,
        repository_root=REPOSITORY_ROOT,
        destination=tmp_path / "proof",
    )

    assert proof.provider_calls == 0
    assert proof.arc_actions == 0
    assert proof.answer_key_used_for_planning is False
    assert proof.all_runtime_paths_correct is True
    assert {
        packet.packet_id: (
            packet.native.provider_calls,
            packet.codeact.provider_calls,
        )
        for packet in proof.packets
    } == {
        "action-success-grounded": (2, 2),
        "history-success-level1-noop": (3, 2),
        "objective-success-level-boundary": (4, 2),
        "cross-level-success-action-response": (6, 2),
        "remote-failure-composition": (32, 2),
    }
    assert {
        packet.packet_id: packet.task_inherently_exceeds_six_calls
        for packet in proof.packets
    } == {
        "action-success-grounded": (),
        "history-success-level1-noop": (),
        "objective-success-level-boundary": (),
        "cross-level-success-action-response": (),
        "remote-failure-composition": ("native",),
    }


def test_report_names_the_observed_contract_gaps() -> None:
    report = json.loads(
        (
            REPOSITORY_ROOT / "analysis/matched-actor-v5-golden-paths/proof/report.json"
        ).read_text()
    )

    assert {finding["finding_id"] for finding in report["interface_findings"]} == {
        "codeact_evidence_query_contract_missing",
        "native_limit_constraint_missing",
        "after_cursor_shape_under_specified",
        "receipt_omits_next_after",
        "native_single_call_policy_not_prompted",
    }


def test_manifest_binds_every_input_and_proof_file() -> None:
    package = REPOSITORY_ROOT / "analysis/matched-actor-v5-golden-paths"
    manifest = json.loads((package / "manifest.json").read_text())

    assert manifest["provider_calls"] == 0
    assert manifest["arc_actions"] == 0
    for record in (*manifest["inputs"], *manifest["proof_files"]):
        path = REPOSITORY_ROOT / record["path"]
        assert (
            record["sha256"]
            == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        )
    assert {record["path"] for record in manifest["proof_files"]} == {
        str(path.relative_to(REPOSITORY_ROOT))
        for path in (package / "proof").rglob("*")
        if path.is_file()
    }
