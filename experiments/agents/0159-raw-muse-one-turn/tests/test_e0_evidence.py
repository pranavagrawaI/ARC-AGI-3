from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "e0_evidence.py"
SPEC = importlib.util.spec_from_file_location("e0_evidence", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
TAXONOMY_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/e0-event-taxonomy-v1.json").read_text()
)


def observation(
    value: int,
    *,
    state: str = "NOT_FINISHED",
    levels_completed: int = 0,
    full_reset: bool = False,
    frames: tuple[tuple[tuple[int, ...], ...], ...] | None = None,
):
    settled = ((value, 0), (0, 0))
    return MODULE.PublicObservationInput(
        frames=frames or (settled,),
        public_state=state,
        levels_completed=levels_completed,
        full_reset=full_reset,
        legal_actions=("RESET", "UP", "DOWN", "LEFT", "RIGHT"),
    )


@pytest.mark.parametrize(
    "case", TAXONOMY_FIXTURE["cases"], ids=lambda case: case["name"]
)
def test_frozen_host_event_taxonomy(case: dict, tmp_path: Path) -> None:
    ledger = MODULE.HostEvidenceLedger(tmp_path / "evidence", run_id="run-a")
    ledger.record_initial_reset(observation(0))
    after = case["after"]

    result = ledger.record_action_result(
        global_turn=0,
        action=MODULE.Action(case["action"]),
        successor=observation(
            after["value"],
            state=after["state"],
            levels_completed=after["levels"],
            full_reset=after["full_reset"],
        ),
    )

    transition = result.snapshot.transitions[0]
    assert transition.boundary == case["boundary"]
    assert transition.outcome == case["outcome"]
    assert transition.attempt_after == case["attempt_after"]


def test_ordinary_transition_stays_in_one_attempt(tmp_path: Path) -> None:
    ledger = MODULE.HostEvidenceLedger(tmp_path / "evidence", run_id="run-a")
    initial = ledger.record_initial_reset(observation(0))

    result = ledger.record_action_result(
        global_turn=0,
        action=MODULE.Action("UP"),
        successor=observation(1),
    )

    transition = result.snapshot.transitions[0]
    assert initial.attempt_id == "attempt-000000"
    assert transition.boundary == "none"
    assert transition.outcome == "ongoing"
    assert transition.attempt_before == initial.attempt_id
    assert transition.attempt_after == initial.attempt_id


@pytest.mark.parametrize(
    ("action", "full_reset", "expected_boundary"),
    [
        ("RESET", False, "explicit_reset"),
        ("UP", True, "api_full_reset"),
        ("RESET", True, "explicit_reset"),
    ],
)
def test_reset_boundary_allocates_exactly_one_successor_attempt(
    tmp_path: Path,
    action: str,
    full_reset: bool,
    expected_boundary: str,
) -> None:
    ledger = MODULE.HostEvidenceLedger(tmp_path / "evidence", run_id="run-a")
    initial = ledger.record_initial_reset(observation(0))

    result = ledger.record_action_result(
        global_turn=0,
        action=MODULE.Action(action),
        successor=observation(0, full_reset=full_reset),
    )

    transition = result.snapshot.transitions[0]
    assert transition.boundary == expected_boundary
    assert transition.attempt_before == initial.attempt_id
    assert transition.attempt_after == "attempt-000001"
    assert ledger.attempt_count == 2


def test_visually_reset_like_ordinary_transition_is_not_a_boundary(
    tmp_path: Path,
) -> None:
    ledger = MODULE.HostEvidenceLedger(tmp_path / "evidence", run_id="run-a")
    initial = ledger.record_initial_reset(observation(0))
    ledger.record_action_result(
        global_turn=0,
        action=MODULE.Action("UP"),
        successor=observation(7),
    )

    result = ledger.record_action_result(
        global_turn=1,
        action=MODULE.Action("DOWN"),
        successor=observation(0),
    )

    transition = result.snapshot.transitions[-1]
    assert transition.boundary == "none"
    assert transition.attempt_before == initial.attempt_id
    assert transition.attempt_after == initial.attempt_id
    assert ledger.attempt_count == 1


def test_game_over_closes_attempt_until_host_actually_resets(tmp_path: Path) -> None:
    ledger = MODULE.HostEvidenceLedger(tmp_path / "evidence", run_id="run-a")
    initial = ledger.record_initial_reset(observation(0))

    terminal = ledger.record_action_result(
        global_turn=0,
        action=MODULE.Action("UP"),
        successor=observation(9, state="GAME_OVER"),
    )

    transition = terminal.snapshot.transitions[0]
    assert transition.outcome == "game_over"
    assert transition.attempt_before == initial.attempt_id
    assert transition.attempt_after is None
    assert terminal.snapshot.terminal_events[0].attempt_id == initial.attempt_id
    with pytest.raises(MODULE.EvidenceIntegrityError, match="active decision"):
        ledger.record_action_result(
            global_turn=1,
            action=MODULE.Action("UP"),
            successor=observation(1),
        )

    reset = ledger.record_host_reset(observation(0), global_turn=1)
    assert reset.attempt_id == "attempt-000001"
    assert ledger.attempt_count == 2


def test_level_completion_keeps_terminal_successor_in_closing_attempt(
    tmp_path: Path,
) -> None:
    ledger = MODULE.HostEvidenceLedger(tmp_path / "evidence", run_id="run-a")
    initial = ledger.record_initial_reset(observation(0))

    completed = ledger.record_action_result(
        global_turn=0,
        action=MODULE.Action("RIGHT"),
        successor=observation(8, state="WIN", levels_completed=1),
    )

    transition = completed.snapshot.transitions[0]
    assert transition.outcome == "level_complete"
    assert transition.attempt_before == initial.attempt_id
    assert transition.attempt_after == initial.attempt_id
    assert completed.observation.attempt_id == initial.attempt_id
    next_decision = ledger.record_decision_observation(
        observation(8, state="NOT_FINISHED", levels_completed=1), global_turn=1
    )
    assert next_decision.attempt_id == "attempt-000001"


def test_animation_is_a_side_channel_not_an_ordinary_transition(
    tmp_path: Path,
) -> None:
    ledger = MODULE.HostEvidenceLedger(tmp_path / "evidence", run_id="run-a")
    ledger.record_initial_reset(observation(0))
    frames = (
        ((0, 0), (0, 0)),
        ((1, 0), (0, 0)),
        ((1, 2), (0, 0)),
    )

    result = ledger.record_action_result(
        global_turn=0,
        action=MODULE.Action("UP"),
        successor=observation(1, frames=frames),
    )

    assert len(result.snapshot.transitions) == 1
    assert len(result.snapshot.animation_events) == 1
    animation = result.snapshot.animation_events[0]
    assert animation.frame_sha256[-1] == result.observation.grid_sha256
    assert animation.settled_frame_index == 2
    assert result.snapshot.transitions[0].animation_event_id == (
        animation.animation_event_id
    )


def test_snapshot_commits_input_prefix_and_derived_manifest_independently(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    ledger = MODULE.HostEvidenceLedger(root, run_id="run-a")
    ledger.record_initial_reset(observation(0))
    result = ledger.record_action_result(
        global_turn=0,
        action=MODULE.Action("UP"),
        successor=observation(1),
    )

    assert MODULE.verify_snapshot(result.snapshot)
    persisted = json.loads(
        (root / "snapshots" / f"{result.snapshot.snapshot_id}.json").read_text()
    )
    assert persisted["occurrence_prefix_sha256"] == (
        result.snapshot.occurrence_prefix_sha256
    )
    assert persisted["transition_records_sha256"] == (
        result.snapshot.transition_records_sha256
    )
    assert result.snapshot.occurrence_prefix_sha256 != (
        result.snapshot.transition_records_sha256
    )
    broken_input = replace(result.snapshot, occurrence_prefix_sha256="0" * 64)
    with pytest.raises(MODULE.EvidenceIntegrityError, match="occurrence-prefix"):
        MODULE.verify_snapshot(broken_input)
    broken_transition = replace(
        result.snapshot.transitions[0], public_state_after="GAME_OVER"
    )
    broken_manifest = replace(result.snapshot, transitions=(broken_transition,))
    with pytest.raises(MODULE.EvidenceIntegrityError, match="derived-manifest"):
        MODULE.verify_snapshot(broken_manifest)
    with pytest.raises(FrozenInstanceError):
        result.snapshot.through_host_sequence = 99
    blob_paths = list((root / "blobs").glob("grid-*.json"))
    assert blob_paths
    assert all(path.stat().st_mode & 0o222 == 0 for path in blob_paths)


def test_frontier_accepts_idempotence_and_strict_append_only_successors(
    tmp_path: Path,
) -> None:
    ledger = MODULE.HostEvidenceLedger(tmp_path / "evidence", run_id="run-a")
    ledger.record_initial_reset(observation(0))
    first = ledger.snapshot()
    second = ledger.record_action_result(
        global_turn=0,
        action=MODULE.Action("UP"),
        successor=observation(1),
    ).snapshot
    frontier = MODULE.EvidenceSnapshotFrontier()

    assert frontier.install(first).status == "installed"
    assert frontier.install(first).status == "idempotent"
    assert frontier.install(second).status == "installed"
    with pytest.raises(MODULE.EvidenceIntegrityError, match="older"):
        frontier.install(first)


def test_frontier_rejects_divergence_at_same_or_later_frontier(
    tmp_path: Path,
) -> None:
    ledger = MODULE.HostEvidenceLedger(tmp_path / "evidence", run_id="run-a")
    ledger.record_initial_reset(observation(0))
    first = ledger.snapshot()
    second = ledger.record_action_result(
        global_turn=0,
        action=MODULE.Action("UP"),
        successor=observation(1),
    ).snapshot
    frontier = MODULE.EvidenceSnapshotFrontier()
    frontier.install(first)

    divergent_same = replace(first, occurrence_prefix_sha256="0" * 64)
    with pytest.raises(MODULE.EvidenceIntegrityError, match="commitment"):
        frontier.install(divergent_same)
    divergent_later = replace(
        second,
        source_event_ids=("different:event:00000000",) + second.source_event_ids[1:],
    )
    with pytest.raises(MODULE.EvidenceIntegrityError, match="prefix"):
        frontier.install(divergent_later)


def test_reopened_ledger_reconstructs_the_same_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    ledger = MODULE.HostEvidenceLedger(root, run_id="run-a")
    ledger.record_initial_reset(observation(0))
    expected = ledger.record_action_result(
        global_turn=0,
        action=MODULE.Action("RIGHT"),
        successor=observation(1),
    ).snapshot

    reopened = MODULE.HostEvidenceLedger(root, run_id="run-a")

    assert reopened.snapshot() == expected
    assert reopened.attempt_count == 1


def test_reopen_rejects_a_tampered_authoritative_prefix(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    ledger = MODULE.HostEvidenceLedger(root, run_id="run-a")
    ledger.record_initial_reset(observation(0))
    ledger_path = root / "ledger.jsonl"
    record = json.loads(ledger_path.read_text())
    record["observation"]["public_state"] = "GAME_OVER"
    ledger_path.write_text(json.dumps(record) + "\n")

    with pytest.raises(MODULE.EvidenceIntegrityError, match="diverges"):
        MODULE.HostEvidenceLedger(root, run_id="run-a")
