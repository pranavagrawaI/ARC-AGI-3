from __future__ import annotations

import json
from pathlib import Path

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from tycho_nooa.cognitive_corpus import load_corpus
from tycho_nooa.parity_experiment import execute_matched_packet

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CORPUS_MANIFEST = (
    REPOSITORY_ROOT / "analysis/codeact-cognitive-parity/v1/corpus-manifest.json"
)


def response(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(id="scripted", name=name, arguments=json.dumps(arguments))
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
        usage={"prompt_tokens": 3, "completion_tokens": 2},
    )


async def test_easy_positive_control_matches_with_no_additional_authority(
    tmp_path: Path,
) -> None:
    packet = _packet("remote-success-old-up")
    answer = {
        "task_kind": "retrieve",
        "evidence_refs": [{"level": 0, "turn": 1}],
        "conclusion": "The earliest UP is L0:T1.",
    }
    report = await execute_matched_packet(
        packet,
        repository_root=REPOSITORY_ROOT,
        destination=tmp_path / "easy",
        native_llm=FakeLLMClient(
            scripted_responses=[
                response("evidence_query", {"kind": "transitions"}),
                response("return_result", {"result": answer}),
            ]
        ),
        codeact_llm=FakeLLMClient(
            scripted_responses=[
                response(
                    "execute_python",
                    {
                        "code": (
                            "page = evidence.query(kind='transitions')\n"
                            "up = next(x for x in page['records'] if x['action'] == 'UP')\n"
                            "return_result({'task_kind': 'retrieve', "
                            "'evidence_refs': [{'level': up['level'], 'turn': up['turn']}], "
                            "'conclusion': 'The earliest UP is L0:T1.'})"
                        )
                    },
                )
            ]
        ),
        max_llm_calls=8,
        max_output_tokens=4096,
        session_timeout_seconds=120,
        cell_timeout_seconds=5,
    )

    assert report.native.evaluation.evidence_recall == 1
    assert report.codeact.evaluation.evidence_recall == 1
    assert report.native.metrics.llm_calls == 2
    assert report.codeact.metrics.llm_calls == 1
    assert report.native.metrics.evidence_queries == 1
    assert report.codeact.metrics.evidence_queries == 1
    assert report.codeact.metrics.python_cells == 1
    assert (tmp_path / "easy/codeact/provenance.jsonl").is_file()
    assert (tmp_path / "easy/report.json").is_file()


async def test_negative_control_preserves_insufficient_evidence_answer(
    tmp_path: Path,
) -> None:
    packet = _packet("objective-failure-insufficient")
    answer = {
        "task_kind": "discriminate",
        "supported": "H_insufficient",
        "falsified": [],
        "evidence_refs": [],
        "conclusion": "Ordinary evidence does not identify completion.",
    }
    report = await execute_matched_packet(
        packet,
        repository_root=REPOSITORY_ROOT,
        destination=tmp_path / "negative",
        native_llm=FakeLLMClient(
            scripted_responses=[response("return_result", {"result": answer})]
        ),
        codeact_llm=FakeLLMClient(
            scripted_responses=[response("return_result", {"result": answer})]
        ),
        max_llm_calls=8,
        max_output_tokens=4096,
        session_timeout_seconds=120,
        cell_timeout_seconds=5,
    )

    assert report.native.evaluation.verdict_correct is True
    assert report.codeact.evaluation.verdict_correct is True
    assert report.native.metrics.evidence_queries == 0
    assert report.codeact.metrics.evidence_queries == 0


async def test_composition_control_requires_two_pages_in_both_arms(
    tmp_path: Path,
) -> None:
    packet = _packet("remote-failure-composition")
    refs = [{"level": 0, "turn": turn} for turn in (1, 4, 39, 56)]
    answer = {
        "task_kind": "retrieve",
        "evidence_refs": refs,
        "conclusion": "Four recorded actions made no visible change.",
    }
    report = await execute_matched_packet(
        packet,
        repository_root=REPOSITORY_ROOT,
        destination=tmp_path / "composition",
        native_llm=FakeLLMClient(
            scripted_responses=[
                response("evidence_query", {"kind": "transitions", "limit": 32}),
                response(
                    "evidence_query",
                    {
                        "kind": "transitions",
                        "after": {"level": 0, "turn": 32},
                        "frontier": {"level": 0, "turn": 59},
                        "limit": 32,
                    },
                ),
                response("return_result", {"result": answer}),
            ]
        ),
        codeact_llm=FakeLLMClient(
            scripted_responses=[
                response(
                    "execute_python",
                    {
                        "code": (
                            "first = evidence.query(kind='transitions', limit=32)\n"
                            "second = evidence.query(kind='transitions', "
                            "after=first['next_after'], frontier=first['frontier'], limit=32)\n"
                            "rows = first['records'] + second['records']\n"
                            "refs = [{'level': x['level'], 'turn': x['turn']} "
                            "for x in rows if x['prev'] == x['next']]\n"
                            "return_result({'task_kind': 'retrieve', 'evidence_refs': refs, "
                            "'conclusion': 'Four recorded actions made no visible change.'})"
                        )
                    },
                )
            ]
        ),
        max_llm_calls=8,
        max_output_tokens=4096,
        session_timeout_seconds=120,
        cell_timeout_seconds=5,
    )

    assert report.native.evaluation.evidence_precision == 1
    assert report.native.evaluation.evidence_recall == 1
    assert report.codeact.evaluation.evidence_precision == 1
    assert report.codeact.evaluation.evidence_recall == 1
    assert report.native.metrics.evidence_queries == 2
    assert report.codeact.metrics.evidence_queries == 2


def _packet(packet_id: str):
    _, packets = load_corpus(CORPUS_MANIFEST, repository_root=REPOSITORY_ROOT)
    return next(packet for packet in packets if packet.task.packet_id == packet_id)
