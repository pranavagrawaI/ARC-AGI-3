from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from nooa.unifiedllm import FakeLLMClient
from pydantic import ValidationError

import tycho_nooa._context_computational_worker_v2 as worker
from tycho_nooa.context_evidence_v2 import (
    ContextEvidencePlane,
    EvidenceAccessContext,
    EvidenceAdmissionAuthority,
    EvidenceCache,
    EvidenceGrantRegistry,
    EvidenceReadBroker,
)
from tycho_nooa.context_projection import ContextProjector
from tycho_nooa.execution_events import ExecutionProvenanceLog, InvocationProvenance
from tycho_nooa.matched_actor_context_v2 import (
    CONTEXT_NATIVE_SYSTEM_PROMPT,
    ContextCodeActController,
)
from tycho_nooa.matched_actor_corpus import load_matched_actor_corpus
from tycho_nooa.native_parity_v2 import NativeEvidenceArguments

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CORPUS = REPOSITORY_ROOT / "analysis/matched-actor-v1/corpus.json"
FRONTIER = {"level": 1, "turn": 2}


class _Source:
    def query(self, raw_query: dict) -> dict:
        return {
            "kind": "transitions",
            "frontier": FRONTIER,
            "records": [
                {
                    "type": "transition",
                    "level": 0,
                    "turn": 1,
                    "ref": "L0:T1",
                    "action": "LEFT",
                }
            ],
            "next_after": {"level": 0, "turn": 1},
            "truncated": True,
            "truncation_reason": "record_limit",
            "grid_encoding": "rle",
            "projection": "metadata",
        }


def _packet():
    _, packets = load_matched_actor_corpus(CORPUS, repository_root=REPOSITORY_ROOT)
    return next(
        item for item in packets if item.task.packet_id == "remote-success-old-up"
    )


def _plane(*, invocation_id: str) -> ContextEvidencePlane:
    cache = EvidenceCache()
    grants = EvidenceGrantRegistry()
    clock = lambda: datetime(2026, 8, 25, tzinfo=UTC)
    context = EvidenceAccessContext(
        session_id="contract-repair",
        invocation_id=invocation_id,
        role="actor",
        frontier=FRONTIER,
    )
    return ContextEvidencePlane(
        _Source(),
        EvidenceAdmissionAuthority(
            cache,
            grants,
            clock=clock,
            handle_factory=lambda: "ev_contract_repair",
        ),
        EvidenceReadBroker(cache, grants, clock=clock),
        context,
    )


def test_native_query_schema_matches_the_authoritative_bounds() -> None:
    schema = NativeEvidenceArguments.model_json_schema()

    assert schema["properties"]["limit"]["minimum"] == 1
    assert schema["properties"]["limit"]["maximum"] == 32
    with pytest.raises(ValidationError):
        NativeEvidenceArguments.model_validate(
            {"kind": "transitions", "limit": 33}
        )


def test_native_cursor_requires_the_complete_evidence_reference() -> None:
    with pytest.raises(ValidationError):
        NativeEvidenceArguments.model_validate(
            {"kind": "transitions", "after": {"turn": 32}}
        )

    query = NativeEvidenceArguments.model_validate(
        {"kind": "transitions", "after": {"level": 0, "turn": 32}}
    )
    assert query.after is not None
    assert query.after.level == 0
    assert query.after.turn == 32


def test_receipt_preserves_structural_pagination_metadata() -> None:
    plane = _plane(invocation_id="receipt")

    receipt = plane.query(
        {
            "kind": "transitions",
            "level": 0,
            "limit": 1,
            "projection": "metadata",
        }
    )

    assert receipt["next_after"] == {"level": 0, "turn": 1}
    assert receipt["truncated"] is True
    assert receipt["truncation_reason"] == "record_limit"


def test_codeact_manifest_makes_the_general_query_contract_discoverable(
    tmp_path: Path,
) -> None:
    packet = _packet()
    invocation_id = "contract-manifest"
    controller = ContextCodeActController(
        llm=FakeLLMClient(scripted_responses=[]),
        packet=packet,
        evidence_plane=_plane(invocation_id=invocation_id),
        projector=ContextProjector(recent_exchanges=2),
        provenance=InvocationProvenance(
            ExecutionProvenanceLog(
                tmp_path / "provenance.jsonl", actor_session_id="contract-repair"
            ),
            invocation_id=invocation_id,
        ),
        max_llm_calls=1,
        max_output_tokens=64,
        cell_timeout_seconds=1,
    )

    manifest = controller._execution_manifest().model_dump(mode="json")
    contract = manifest["capability_contracts"]["evidence.query"]
    namespace = worker._capability_namespace(manifest)
    rendered = repr(namespace["evidence"])

    assert contract["parameters"] == NativeEvidenceArguments.model_json_schema()
    assert contract["result_schema"] == "evidence_receipt.v2"
    assert contract["example"] == {
        "kind": "transitions",
        "level": 0,
        "limit": 16,
        "projection": "metadata",
    }
    assert "after" in rendered
    assert "frontier" in rendered
    assert "limit: int = 16" in rendered
    assert "maximum 32" in rendered


def test_native_prompt_states_the_controller_single_call_rule() -> None:
    assert "Call exactly one tool per provider turn" in CONTEXT_NATIVE_SYSTEM_PROMPT


def test_repair_checkpoint_binds_all_inputs_and_proofs() -> None:
    package = REPOSITORY_ROOT / "analysis/matched-actor-contract-repair-v1"
    manifest = json.loads((package / "manifest.json").read_text())

    assert manifest["provider_calls"] == 0
    assert manifest["arc_actions"] == 0
    for record in (*manifest["inputs"], *manifest["proof_files"]):
        path = REPOSITORY_ROOT / record["path"]
        assert record["sha256"] == "sha256:" + hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
