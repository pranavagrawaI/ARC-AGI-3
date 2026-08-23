from __future__ import annotations

from pathlib import Path

from tycho_nooa.cognitive_corpus import load_corpus
from tycho_nooa.evidence import WorkspaceEvidenceCapability

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPOSITORY_ROOT / "analysis/codeact-cognitive-parity/v1/corpus-manifest.json"


def test_v1_corpus_is_committed_crossed_and_ordinary_evidence_complete() -> None:
    manifest, packets = load_corpus(MANIFEST, repository_root=REPOSITORY_ROOT)

    assert manifest.packet_count == 12
    assert len(packets) == 12
    assert {packet.task.selection_stratum for packet in packets} == {
        "remote_retrieval",
        "hypothesis_discrimination",
        "cross_level_invariant",
        "history_dependence",
        "objective_inference",
        "next_action",
    }
    assert all("expected" not in packet.treatment_payload() for packet in packets)


def test_composition_key_requires_second_page_and_exact_noop_refs() -> None:
    _, packets = load_corpus(MANIFEST, repository_root=REPOSITORY_ROOT)
    packet = next(
        item for item in packets if item.task.packet_id == "remote-failure-composition"
    )
    capability = WorkspaceEvidenceCapability.from_root(
        REPOSITORY_ROOT / packet.checkpoint_root
    )
    first = capability.query({"kind": "transitions", "limit": 32})
    second = capability.query(
        {
            "kind": "transitions",
            "after": first["next_after"],
            "frontier": first["frontier"],
            "limit": 32,
        }
    )
    transitions = [*first["records"], *second["records"]]
    noops = {
        (item["level"], item["turn"])
        for item in transitions
        if item["prev"] == item["next"]
    }

    assert first["truncated"] is True
    assert second["truncated"] is False
    assert noops == {(0, 1), (0, 4), (0, 39), (0, 56)}


def test_cross_level_atomic_key_matches_independent_grid_diffs() -> None:
    _, packets = load_corpus(MANIFEST, repository_root=REPOSITORY_ROOT)
    packet = next(
        item
        for item in packets
        if item.task.packet_id == "cross-level-success-action-response"
    )
    capability = WorkspaceEvidenceCapability.from_root(
        REPOSITORY_ROOT / packet.checkpoint_root
    )
    transitions = capability.query({"kind": "transitions", "limit": 32})[
        "records"
    ]
    by_ref = {(item["level"], item["turn"]): item for item in transitions}

    assert _changed_cells(by_ref[(0, 16)]) == 108
    assert _changed_cells(by_ref[(1, 2)]) == 487
    assert packet.expected.invariant_atoms == {
        "same_changed_cell_count": False,
        "level0_changed_cells": 108,
        "level1_changed_cells": 487,
    }


def _changed_cells(transition: dict) -> int:
    return sum(
        before != after
        for before_row, after_row in zip(transition["prev"], transition["next"])
        for before, after in zip(before_row, after_row)
    )
