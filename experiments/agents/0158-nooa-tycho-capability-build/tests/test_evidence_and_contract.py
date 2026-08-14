from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from host.contract import (
    MODEL_CONTRACT,
    MODEL_CONTRACT_SHA256,
    contract_word_count,
    validate_model_contract,
)
from host.evidence import EvidenceStore, diff_text


def test_contract_is_fixed_short_and_rejects_role_or_mechanics_disclosures() -> None:
    assert validate_model_contract(deepcopy(MODEL_CONTRACT)) == MODEL_CONTRACT
    assert contract_word_count() <= 400
    assert len(MODEL_CONTRACT_SHA256) == 64

    leaked = deepcopy(MODEL_CONTRACT)
    leaked["objective"] = "The player begins at row 61."
    with pytest.raises(ValueError, match="prohibited"):
        validate_model_contract(leaked)

    extra = deepcopy(MODEL_CONTRACT)
    extra["mechanics"] = "ACTION1 changes something"
    with pytest.raises(ValueError, match="non-allowlisted"):
        validate_model_contract(extra)


def test_content_addressed_frames_are_immutable_and_deduplicated(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "evidence")
    one = store.record_frame([[0, 1], [2, 3]], level=0, turn=0, attempt_id="a", available_actions=[1])
    two = store.record_frame([[0, 1], [2, 3]], level=0, turn=0, attempt_id="a", available_actions=[1])
    assert one == two
    assert len((store.indexes / "frame.jsonl").read_text().splitlines()) == 1
    stored = store.objects / f"{one['evidence_id']}.json"
    original = stored.read_bytes()
    assert json.loads(original)["grid"] == [[0, 1], [2, 3]]
    assert stored.stat().st_mode & 0o222 == 0


def test_transitions_cannot_cross_attempt_or_level_and_events_stay_separate(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "evidence")
    before = store.record_frame([[1]], level=0, turn=0, attempt_id="a", available_actions=[1])
    other_attempt = store.record_frame([[2]], level=0, turn=1, attempt_id="b", available_actions=[1])
    with pytest.raises(ValueError, match="attempt or level"):
        store.record_transition(before["evidence_id"], other_attempt["evidence_id"], action=1, data=None, commitment_id="c")
    event = store.record_event(
        "death", level=0, attempt_id="a", before_id=before["evidence_id"], grids=[[[9]]], action=1
    )
    assert event["kind"] == "death"
    assert store.transitions() == []
    assert len(store.events()["death"]) == 1


def test_long_run_view_bounds_full_grids_but_preserves_history_and_probe_results(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "evidence")
    frames = [
        store.record_frame([[turn]], level=0, turn=turn, attempt_id="a", available_actions=[1])
        for turn in range(4)
    ]
    store.record_transition(
        frames[0]["evidence_id"], frames[1]["evidence_id"],
        action=1, data=None, commitment_id="commitment-a",
    )
    store.record_probe_result(
        before_id=frames[0]["evidence_id"], after_id=frames[1]["evidence_id"],
            grounding_claim={"claim_id": "claim-a"},
            probe_commitment={"commitment_id": "commitment-a"},
            verifier_verdict={"claim_id": "claim-a", "commitment_id": "commitment-a", "status": "contradicted"},
            change_accounting={
                "schema": "0158-change-accounting-v1",
                "transition_id": "transition-a",
                "changed_cells_total": 1,
                "changed_cells_inside_primary": 1,
                "changed_cells_outside_primary": 0,
                "primary_region_changed": True,
                "outside_primary_changed": False,
            },
        )
    view = store.model_view(frame_window=2)
    assert [frame["turn"] for frame in view["frames"]] == [2, 3]
    assert [frame["turn"] for frame in view["frame_summaries"]] == [0, 1, 2, 3]
    assert all("grid" not in frame for frame in view["frame_summaries"])
    assert view["probe_results"][0]["verifier_verdict"]["status"] == "contradicted"


def test_diff_text_matches_pinned_tycho_golden_fixture() -> None:
    before = [[3, 3, 3], [3, 9, 9], [3, 9, 9]]
    after = [[3, 12, 12], [3, 3, 3], [3, 9, 9]]
    assert diff_text(before, after) == (
        "4 cells changed; region [rows 0-1, cols 1-2]\n"
        "  3->12  row 0, cols 1-2\n"
        "  9->3  row 1, cols 1-2"
    )


def test_historical_shadow_contains_only_cropped_public_records() -> None:
    fixture = json.loads(
        (Path(__file__).resolve().parents[1] / "fixtures/historical-shadow/ls20-public-crop.json").read_text()
    )
    assert fixture["public_only"] is True
    assert fixture["sanitization"]["roles_removed"] is True
    assert len(fixture["frames"]) == 4
    assert all(len(frame["grid"]) == 16 and len(frame["grid"][0]) == 20 for frame in fixture["frames"])
    lowered = json.dumps(fixture).lower()
    assert all(term not in lowered for term in ("player", "goal", "wall", "controllable"))
