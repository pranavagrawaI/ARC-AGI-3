from __future__ import annotations

import pytest

import tycho_nooa._context_computational_worker as worker
from tycho_nooa.context_computational_evidence import (
    EvidenceBoundedView,
    EvidencePageView,
    EvidenceReceiptView,
)


class _Client:
    def __init__(self, result: dict):
        self.result = result
        self.calls: list[tuple[str, dict]] = []

    def call(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return self.result


def _manifest(*, capabilities: object, contracts: object | None = None) -> dict:
    return {
        "invocation_id": "context-worker-test",
        "capabilities": capabilities,
        "capability_contracts": {} if contracts is None else contracts,
    }


def test_context_worker_namespace_exposes_only_granted_surfaces() -> None:
    contract = {
        "kind": "transitions",
        "level": 1,
        "limit": 8,
        "projection": "full",
    }
    values = worker._capability_namespace(
        _manifest(
            capabilities=["evidence.query", "evidence.read", "evidence.view"],
            contracts={"evidence.query": contract},
        )
    )

    assert set(values) == {"rle", "evidence", "evidence_cache"}
    assert "query" in repr(values["evidence"])
    assert "load" in repr(values["evidence_cache"])
    assert worker._has_query_example(contract)
    assert not worker._has_query_example({"kind": "transitions"})
    assert not worker._has_query_example(None)


def test_context_worker_read_only_namespace_has_no_query_capability() -> None:
    values = worker._capability_namespace(_manifest(capabilities=["evidence.read"]))

    assert set(values) == {"rle", "evidence_cache"}


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (_manifest(capabilities="evidence.read"), "must be a list"),
        (_manifest(capabilities=["workspace.read"]), "unknown sterile"),
        (_manifest(capabilities=["evidence.view"]), "requires the evidence query"),
        (
            _manifest(capabilities=[], contracts=[]),
            "capability_contracts must be an object",
        ),
        (
            _manifest(
                capabilities=["evidence.query"],
                contracts={"evidence.query": "opaque"},
            ),
            "query contract must be an object",
        ),
    ],
)
def test_context_worker_rejects_namespace_drift(manifest: dict, message: str) -> None:
    with pytest.raises((TypeError, RuntimeError), match=message):
        worker._capability_namespace(manifest)


def test_context_worker_value_views_are_readable_but_not_authoritative() -> None:
    page = EvidencePageView(
        {
            "records": [{"ref": "L1:T1", "action": "DOWN"}],
            "frontier": {"level": 1, "turn": 2},
        }
    )
    receipt = EvidenceReceiptView(
        {
            "schema_version": "evidence_receipt.v1",
            "handle": "ev_one",
            "record_count": 1,
            "refs": ["L1:T1"],
            "encoding": "rle",
        }
    )
    bounded = EvidenceBoundedView({"handle": "ev_one", "records": [{"ref": "L1:T1"}]})

    assert page.records[0].action == "DOWN"
    assert receipt.handle == "ev_one"
    assert receipt["record_count"] == 1
    assert tuple(receipt) == tuple(receipt._value)
    assert len(receipt) == len(receipt._value)
    assert "ev_one" in repr(receipt)
    assert bounded.records[0].ref == "L1:T1"
    assert bounded["records"] == bounded.records
    assert bounded.handle == "ev_one"
    assert "ev_one" in repr(bounded)
    with pytest.raises(AttributeError):
        _ = receipt.unavailable
    with pytest.raises(AttributeError):
        _ = bounded.unavailable
    with pytest.raises(TypeError, match="unsupported"):
        EvidencePageView({})
    with pytest.raises(ValueError, match="unsupported"):
        EvidenceReceiptView({})


def test_context_worker_proxies_return_typed_value_views() -> None:
    view_client = _Client({"handle": "ev_one", "records": [{"ref": "L1:T1"}]})
    view = worker._EvidenceView(view_client, EvidenceBoundedView)
    assert view(handle="ev_one").records[0].ref == "L1:T1"
    assert "evidence.view" in repr(view)
    assert view_client.calls == [("evidence.view", {"handle": "ev_one"})]

    read_client = _Client({"records": [{"ref": "L1:T1"}]})
    read = worker._EvidenceRead(read_client, EvidencePageView)
    assert read("ev_one").records[0].ref == "L1:T1"
    assert "evidence_cache.load" in repr(read)
    assert read_client.calls == [("evidence.read", {"handle": "ev_one"})]


def test_context_worker_output_governor_preserves_digest_and_bound() -> None:
    output = worker._BoundedText(128)

    assert output.write("small") == 5
    assert output.getvalue() == "small"
    output.write("x" * 1_000)

    value = output.getvalue()
    assert len(value.encode()) <= 128
    assert "output truncated: 1005 bytes; sha256=" in value
