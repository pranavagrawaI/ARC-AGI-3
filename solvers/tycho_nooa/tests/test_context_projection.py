from __future__ import annotations

import json
from pathlib import Path

from tycho_nooa.context_projection import ContextProjector
from tycho_nooa.execution_events import (
    ExecutionProvenanceLog,
    InvocationProvenance,
    NooaExecutionProjection,
    render_execution_context,
)


def _native_history(exchanges: int, payload_bytes: int = 2_000) -> list[dict]:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    for index in range(exchanges):
        call_id = f"call-{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "evidence_query",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": f"result-{index}:" + "x" * payload_bytes,
                },
            ]
        )
    return messages


def _codeact_history(path: Path, cells: int, output_bytes: int = 2_000):
    log = ExecutionProvenanceLog(path, actor_session_id="actor-session")
    recorder = InvocationProvenance(log, invocation_id="actor-turn")
    recorder.start({"role": "actor"})
    for cell in range(1, cells + 1):
        source = f"print('cell-{cell}')"
        recorder.cell_dispatched(cell, source)
        if cell == 1:
            recorder.capability_requested(
                cell, cell, "evidence.query", {"kind": "transitions"}
            )
            recorder.capability_resolved(
                cell,
                cell,
                result={"handle": "ev_first", "record_count": 1},
            )
        recorder.cell_finished(
            cell,
            status="complete",
            stdout=f"cell-{cell}:" + "x" * output_bytes,
            stderr="",
            locals_preview={},
            exception=None,
        )
    return log


def test_native_projection_preserves_recent_tool_topology_and_compact_receipts() -> (
    None
):
    messages = _native_history(8)
    receipts = tuple(
        {
            "handle": f"ev_{index}",
            "content_hash": f"sha256:{index}",
            "refs": [f"L0:T{index}"],
            "record_count": 1,
            "secret_interpretation": "must not enter context",
        }
        for index in range(3)
    )

    projected = ContextProjector(recent_exchanges=2).native(messages, receipts=receipts)

    assert len(json.dumps(projected)) < len(json.dumps(messages)) // 3
    assert projected[-4:] == messages[-4:]
    assert "CONTEXT_HISTORY_COLLAPSED" in projected[2]["content"]
    assert "ev_2" in projected[2]["content"]
    assert "secret_interpretation" not in projected[2]["content"]
    assert messages[2]["tool_calls"][0]["id"] == "call-0"


def test_codeact_projection_is_bounded_while_lossless_provenance_is_unchanged(
    tmp_path: Path,
) -> None:
    log = _codeact_history(tmp_path / "events.jsonl", cells=8)
    raw_before = log.path.read_bytes()
    full = render_execution_context(NooaExecutionProjection(log.events).event_manager())
    projected = ContextProjector(recent_exchanges=2).codeact(log.events)

    assert len(json.dumps(projected)) < len(json.dumps(full)) // 3
    assert "codeact_history_collapse.v1" in json.dumps(projected)
    assert "cell-1:" not in json.dumps(projected)
    assert "cell-7:" in json.dumps(projected)
    assert "cell-8:" in json.dumps(projected)
    assert "ev_first" in json.dumps(projected)
    assert log.path.read_bytes() == raw_before


def test_old_execution_growth_is_sublinear_after_collapse(tmp_path: Path) -> None:
    small = _codeact_history(tmp_path / "small.jsonl", cells=5)
    large = _codeact_history(tmp_path / "large.jsonl", cells=50)
    projector = ContextProjector(recent_exchanges=2)

    small_size = len(json.dumps(projector.codeact(small.events)))
    large_size = len(json.dumps(projector.codeact(large.events)))

    assert large_size < small_size * 1.2
