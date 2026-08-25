from __future__ import annotations

from tycho_nooa.context_computational_evidence_v3 import EvidenceReceiptView
from tycho_nooa.context_execution_v3 import (
    ContextComputationalSterileExecutor,
    ContextExecutionManifest,
)
from tycho_nooa.native_parity_v2 import NativeEvidenceArguments


def _receipt() -> dict:
    return {
        "schema_version": "evidence_receipt.v2",
        "handle": "ev_fixed",
        "content_hash": "sha256:" + "a" * 64,
        "query_hash": "sha256:" + "b" * 64,
        "refs": ["L0:T1", "L0:T2"],
        "record_count": 2,
        "record_fields": ["action", "ref"],
        "encoding": "rle",
        "frontier": {"level": 1, "turn": 2},
        "next_after": None,
        "truncated": False,
        "truncation_reason": None,
        "serialized_bytes": 512,
    }


def _query_contract() -> dict:
    return {
        "schema_version": "evidence_query.v1",
        "parameters": NativeEvidenceArguments.model_json_schema(),
        "example": {
            "kind": "transitions",
            "level": 0,
            "limit": 16,
            "projection": "metadata",
        },
        "result_schema": "evidence_receipt.v2",
    }


def test_receipt_repr_and_discovery_name_only_real_attributes() -> None:
    receipt = EvidenceReceiptView(_receipt())

    rendered = repr(receipt)
    assert "record_count=2" in rendered
    assert "records=2" not in rendered
    assert "evidence_cache.load('ev_fixed')" in rendered
    assert "record_count" in dir(receipt)
    assert "records" not in dir(receipt)
    assert receipt.record_count == 2


async def test_sterile_worker_exposes_the_congruent_receipt_contract() -> None:
    manifest = ContextExecutionManifest(
        invocation_id="receipt-v3",
        role="actor",
        capabilities=("evidence.query", "evidence.read"),
        capability_contracts={"evidence.query": _query_contract()},
    )

    async with ContextComputationalSterileExecutor(
        manifest,
        return_type=dict,
        capability_handlers={
            "evidence.query": lambda _: _receipt(),
            "evidence.read": lambda _: {
                "records": [{"ref": "L0:T1"}, {"ref": "L0:T2"}],
                "frontier": {"level": 1, "turn": 2},
            },
        },
    ) as executor:
        result = await executor.run_cell(
            "receipt=evidence.query(kind='transitions', level=0, "
            "limit=16, projection='metadata')\n"
            "print(repr(receipt))\n"
            "print('record_count' in dir(receipt), 'records' in dir(receipt))\n"
            "page=evidence_cache.load(receipt.handle)\n"
            "print(page.records[0].ref)"
        )

    assert result.status == "complete"
    assert "record_count=2" in result.stdout
    assert "records=2" not in result.stdout
    assert "evidence_cache.load('ev_fixed')" in result.stdout
    assert result.stdout.endswith("True False\nL0:T1\n")
