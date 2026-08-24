from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from tycho_nooa.context_evidence import (
    ContextEvidencePlane,
    EvidenceAccessContext,
    EvidenceAdmissionAuthority,
    EvidenceCache,
    EvidenceGrantRegistry,
    EvidenceReadBroker,
)
from tycho_nooa.context_projection import ContextProjector
from tycho_nooa.evidence_v2 import WorkspaceEvidenceCapability
from tycho_nooa.execution_events import ExecutionProvenanceLog, InvocationProvenance
from tycho_nooa.matched_actor_context import (
    ContextCodeActController,
    ContextNativeActorController,
)
from tycho_nooa.matched_actor_corpus import load_matched_actor_corpus

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CORPUS = REPOSITORY_ROOT / "analysis/matched-actor-v1/corpus.json"


def _response(name: str, arguments: dict, call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
        usage={"prompt_tokens": 2, "completion_tokens": 1, "cost": 0.01},
    )


def _packet():
    _, packets = load_matched_actor_corpus(CORPUS, repository_root=REPOSITORY_ROOT)
    return next(
        item for item in packets if item.task.packet_id == "remote-success-old-up"
    )


class _Source:
    def __init__(self, frontier: dict):
        self.queries: list[dict] = []
        self.value = {
            "kind": "transitions",
            "frontier": frontier,
            "records": [
                {
                    "type": "transition",
                    "level": 0,
                    "turn": 1,
                    "ref": "L0:T1",
                    "action": "UP",
                    "prev": ["b:0"],
                    "next": ["a:0,1:5"],
                }
            ],
            "next_after": None,
            "truncated": False,
            "grid_encoding": "rle",
            "projection": "full",
        }

    def query(self, raw_query: dict) -> dict:
        self.queries.append(raw_query)
        return self.value


def _plane(packet, *, invocation_id: str):
    source = _Source(packet.task.evidence_frontier.model_dump(mode="json"))
    context = EvidenceAccessContext(
        session_id="matched-session",
        invocation_id=invocation_id,
        role="actor",
        frontier=packet.task.evidence_frontier.model_dump(mode="json"),
    )
    cache = EvidenceCache()
    grants = EvidenceGrantRegistry()
    clock = lambda: datetime(2026, 8, 24, tzinfo=UTC)
    admission = EvidenceAdmissionAuthority(
        cache,
        grants,
        clock=clock,
        handle_factory=lambda: "ev_fixed",
    )
    broker = EvidenceReadBroker(cache, grants, clock=clock)
    return ContextEvidencePlane(source, admission, broker, context), source


def _final() -> dict:
    return {
        "conclusion": "The earliest recorded UP transition is L0:T1.",
        "evidence_refs": [{"level": 0, "turn": 1}],
    }


def test_real_workspace_query_becomes_a_receipt_with_exact_brokered_value() -> None:
    packet = _packet()
    root = REPOSITORY_ROOT / packet.checkpoint_root
    source = WorkspaceEvidenceCapability.from_root(root)
    context = EvidenceAccessContext(
        session_id="real-workspace",
        invocation_id="actor-turn",
        role="actor",
        frontier=packet.task.evidence_frontier.model_dump(mode="json"),
    )
    cache = EvidenceCache()
    grants = EvidenceGrantRegistry()
    plane = ContextEvidencePlane(
        source,
        EvidenceAdmissionAuthority(
            cache, grants, handle_factory=lambda: "ev_real_workspace"
        ),
        EvidenceReadBroker(cache, grants),
        context,
    )
    query = {
        "kind": "transitions",
        "level": 0,
        "limit": 8,
        "projection": "full",
        "grid_encoding": "rle",
    }

    receipt = plane.query(query)
    exact = plane.read({"handle": receipt["handle"]})

    assert receipt["refs"] == [record["ref"] for record in exact["records"]]
    assert receipt["serialized_bytes"] > len(json.dumps(receipt))
    assert exact == source.query(query)


async def test_native_receipt_keeps_bulk_evidence_out_of_provider_history() -> None:
    packet = _packet()
    plane, source = _plane(packet, invocation_id="native-turn")
    llm = FakeLLMClient(
        scripted_responses=[
            _response(
                "evidence_query",
                {
                    "kind": "transitions",
                    "level": 0,
                    "limit": 8,
                    "projection": "full",
                    "grid_encoding": "rle",
                },
                "query",
            ),
            _response(
                "run_python",
                {
                    "code": (
                        "page=evidence_cache.load('ev_fixed')\n"
                        "print(page.records[0].ref, page.records[0].action)"
                    ),
                    "values": {},
                },
                "compute",
            ),
            _response("return_result", _final(), "return"),
        ]
    )
    controller = ContextNativeActorController(
        llm=llm,
        packet=packet,
        evidence_plane=plane,
        projector=ContextProjector(recent_exchanges=2),
        max_llm_calls=3,
        max_output_tokens=256,
        cell_timeout_seconds=2,
    )

    answer = await controller.run()

    assert answer.conclusion.startswith("The earliest")
    assert source.queries
    rendered = json.dumps(llm.last_messages)
    assert "ev_fixed" in rendered
    assert '"prev":["b:0"]' not in rendered
    assert "L0:T1 UP" in rendered


async def test_codeact_uses_the_same_receipt_and_read_semantics(tmp_path: Path) -> None:
    packet = _packet()
    invocation_id = "codeact-turn"
    plane, source = _plane(packet, invocation_id=invocation_id)
    llm = FakeLLMClient(
        scripted_responses=[
            _response(
                "execute_python",
                {
                    "code": (
                        "receipt=evidence.query(kind='transitions', level=0, "
                        "limit=8, projection='full', grid_encoding='rle')"
                    )
                },
                "query",
            ),
            _response(
                "execute_python",
                {
                    "code": (
                        "page=evidence_cache.load(receipt.handle)\n"
                        "print(page.records[0].ref, page.records[0].action)"
                    )
                },
                "compute",
            ),
            _response("return_result", _final(), "return"),
        ]
    )
    log = ExecutionProvenanceLog(
        tmp_path / "provenance.jsonl", actor_session_id="matched-session"
    )
    controller = ContextCodeActController(
        llm=llm,
        packet=packet,
        evidence_plane=plane,
        projector=ContextProjector(recent_exchanges=2),
        provenance=InvocationProvenance(log, invocation_id=invocation_id),
        max_llm_calls=3,
        max_output_tokens=256,
        cell_timeout_seconds=2,
    )

    answer = await controller.run()

    assert answer.conclusion.startswith("The earliest")
    assert source.queries
    assert any(
        event.event_type == "CapabilityReturned"
        and event.payload.get("result", {}).get("handle") == "ev_fixed"
        for event in log.events
    )
    assert "L0:T1 UP" in json.dumps(controller.provider_turns[-1].messages)
