from __future__ import annotations

import json
from pathlib import Path

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from tycho_nooa.evidence import EvidenceQuery as LegacyEvidenceQuery
from tycho_nooa.matched_actor_corpus import load_matched_actor_corpus
from tycho_nooa.matched_actor_experiment_v3 import _recording_evidence
from tycho_nooa.matched_actor_pilot_run_v3 import execute_pilot_pair

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CORPUS = REPOSITORY_ROOT / "analysis/matched-actor-v1/corpus.json"


def _response(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=name, name=name, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
        usage={"prompt_tokens": 2, "completion_tokens": 1, "cost": 0.01},
    )


def _packet():
    _, packets = load_matched_actor_corpus(CORPUS, repository_root=REPOSITORY_ROOT)
    return next(
        item for item in packets if item.task.packet_id == "remote-success-old-up"
    )


def test_v3_adapts_legacy_and_codeact_queries_to_identical_values() -> None:
    packet = _packet()
    root = REPOSITORY_ROOT / packet.checkpoint_root
    query = {
        "kind": "transitions",
        "level": 0,
        "limit": 32,
        "projection": "metadata",
        "grid_encoding": "rle",
    }
    native = _recording_evidence(root)
    codeact = _recording_evidence(root)

    native_value = native.query(LegacyEvidenceQuery.model_validate(query))
    codeact_value = codeact.query(query)

    assert native_value == codeact_value
    assert native.disclosures == codeact.disclosures == [native_value]


async def test_v3_pilot_pair_discloses_target_to_both_real_controllers(
    tmp_path: Path,
) -> None:
    packet = _packet()
    query = {
        "kind": "transitions",
        "level": 0,
        "limit": 32,
        "projection": "metadata",
        "grid_encoding": "rle",
    }
    final = {
        "conclusion": "The only and earliest UP is L0:T1.",
        "evidence_refs": [{"level": 0, "turn": 1}],
    }
    native = FakeLLMClient(
        scripted_responses=[
            _response("evidence_query", query),
            _response("return_result", final),
        ]
    )
    codeact = FakeLLMClient(
        scripted_responses=[
            _response(
                "execute_python",
                {
                    "code": (
                        "page = evidence.query(kind='transitions', level=0, limit=32, "
                        "projection='metadata', grid_encoding='rle')\n"
                        "print(page.records[0].ref)"
                    )
                },
            ),
            _response("return_result", final),
        ]
    )

    report = await execute_pilot_pair(
        packet,
        stratum="composition",
        arm_order=("native", "codeact"),
        repository_root=REPOSITORY_ROOT,
        destination=tmp_path / "pair",
        native_llm=native,
        codeact_llm=codeact,
        max_provider_calls=2,
        max_output_tokens=256,
        session_timeout_seconds=10,
        cell_timeout_seconds=2,
    )

    assert report.complete is True
    for result in report.arms:
        assert result.observation is not None
        assert result.observation.resources.evidence_disclosures == 1
        assert result.observation.milestones.target_record_disclosed == 1
        assert result.observation.evaluation.evidence_recall == 1
