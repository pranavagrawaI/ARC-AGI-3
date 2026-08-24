"""Zero-provider replay of a V2-failed packet through both V3 controllers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from tycho_nooa.computational_evidence import describe_evidence_page
from tycho_nooa.evidence_v2 import WorkspaceEvidenceCapability
from tycho_nooa.matched_actor_audit import _flat_expected_answer
from tycho_nooa.matched_actor_corpus import load_matched_actor_corpus
from tycho_nooa.matched_actor_pilot_run_v3 import execute_pilot_pair

REPO = Path(__file__).resolve().parents[2]
BASE = Path(__file__).parent
CORPUS = REPO / "analysis/matched-actor-v1/corpus.json"
PACKET_ID = "history-success-level1-noop"
QUERY = {
    "kind": "transitions",
    "level": 1,
    "limit": 32,
    "projection": "full",
    "grid_encoding": "rle",
}
NATIVE_COMPUTE = (
    "record = page['records'][0]\n"
    "same = [rle.decode_row(row) for row in record['prev']] == "
    "[rle.decode_row(row) for row in record['next']]\n"
    "print(record['ref'], same)"
)
CODEACT_COMPUTE = (
    "page = evidence.query(kind='transitions', level=1, limit=32, "
    "projection='full', grid_encoding='rle')\n"
    "record = page.records[0]\n"
    "same = [rle.decode_row(row) for row in record.prev_rows] == "
    "[rle.decode_row(row) for row in record.next_rows]\n"
    "print(record.ref, same)"
)


def _response(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(
                id=f"scripted-{name}",
                name=name,
                arguments=json.dumps(arguments, separators=(",", ":")),
            )
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
    )


async def main() -> None:
    _, packets = load_matched_actor_corpus(CORPUS, repository_root=REPO)
    packet = next(item for item in packets if item.task.packet_id == PACKET_ID)
    root = REPO / packet.checkpoint_root
    page = WorkspaceEvidenceCapability.from_root(root).query(QUERY)
    final = _flat_expected_answer(packet)
    native = FakeLLMClient(
        scripted_responses=[
            _response("evidence_query", QUERY),
            _response(
                "run_python",
                {"code": NATIVE_COMPUTE, "values": {"page": describe_evidence_page(page)}},
            ),
            _response("return_result", final),
        ]
    )
    codeact = FakeLLMClient(
        scripted_responses=[
            _response("execute_python", {"code": CODEACT_COMPUTE}),
            _response("return_result", final),
        ]
    )
    destination = BASE / "offline-replay" / PACKET_ID
    if destination.exists():
        raise FileExistsError(destination)
    report = await execute_pilot_pair(
        packet,
        stratum="computation_useful",
        arm_order=("native", "codeact"),
        repository_root=REPO,
        destination=destination,
        native_llm=native,
        codeact_llm=codeact,
        max_provider_calls=6,
        max_output_tokens=4096,
        session_timeout_seconds=120,
        cell_timeout_seconds=5,
    )
    observations = [result.observation for result in report.arms]
    valid = report.complete and all(
        observation is not None
        and observation.evaluation.verdict_correct is True
        and observation.evaluation.falsification_correct is True
        and observation.evaluation.evidence_recall == 1
        and observation.resources.evidence_disclosures == 1
        and observation.resources.invalid_interactions == 0
        and observation.milestones.target_record_disclosed == 1
        and observation.milestones.representation_decoded is not None
        and observation.milestones.terminal_result_accepted is not None
        for observation in observations
    )
    summary = {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "external_provider_calls": 0,
        "scripted_model_turns": native.call_count + codeact.call_count,
        "pair_complete": report.complete,
        "adapter_repair_valid": valid,
        "arms": [
            {
                "arm": observation.arm,
                "provider_calls": observation.resources.provider_calls,
                "evidence_disclosures": observation.resources.evidence_disclosures,
                "invalid_interactions": observation.resources.invalid_interactions,
                "target_record_disclosed": observation.milestones.target_record_disclosed,
                "representation_decoded": observation.milestones.representation_decoded,
                "terminal_result_accepted": observation.milestones.terminal_result_accepted,
                "evidence_recall": observation.evaluation.evidence_recall,
                "verdict_correct": observation.evaluation.verdict_correct,
                "falsification_correct": observation.evaluation.falsification_correct,
            }
            for observation in observations
            if observation is not None
        ],
    }
    (BASE / "offline-replay.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if not valid:
        raise RuntimeError("V3 value-adapter replay failed")


if __name__ == "__main__":
    asyncio.run(main())
