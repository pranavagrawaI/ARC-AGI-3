from __future__ import annotations

import json
from pathlib import Path

import pytest

from tycho_nooa.matched_actor_context_audit import audit_context_actor_packet
from tycho_nooa.matched_actor_corpus import load_matched_actor_corpus
from tycho_nooa.matched_actor_pilot import load_matched_actor_pilot_sample

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CORPUS = REPOSITORY_ROOT / "analysis/matched-actor-v1/corpus.json"
SAMPLE = REPOSITORY_ROOT / "analysis/matched-actor-v1-pilot/sample.json"


@pytest.mark.parametrize(
    "packet_id",
    [record["packet_id"] for record in json.loads(SAMPLE.read_text())["packets"]],
)
async def test_context_plane_reaches_same_six_packet_answers(
    packet_id: str, tmp_path: Path
) -> None:
    sample, sampled_packets = load_matched_actor_pilot_sample(
        SAMPLE, repository_root=REPOSITORY_ROOT
    )
    _, corpus_packets = load_matched_actor_corpus(
        CORPUS, repository_root=REPOSITORY_ROOT
    )
    packets = {packet.task.packet_id: packet for packet in corpus_packets}
    scheduled = next(item for item in sample.packets if item.packet_id == packet_id)

    audit = await audit_context_actor_packet(
        packets[packet_id],
        stratum=scheduled.stratum,
        repository_root=REPOSITORY_ROOT,
        destination=tmp_path / packet_id,
    )

    assert len(sampled_packets) == 6
    assert audit.evidence_disclosures_identical
    assert audit.computation_stdout_identical
    assert audit.native_answer_correct
    assert audit.codeact_answer_correct
    assert audit.bulk_evidence_absent_from_provider_context
    assert audit.immutable_handle_semantics_shared
    assert not audit.native_python_has_evidence_acquisition_authority
    assert not audit.canonical_write_authority
    assert not audit.arc_action_authority
