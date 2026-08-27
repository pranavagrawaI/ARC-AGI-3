from __future__ import annotations

import json
from pathlib import Path

import pytest

from tycho_nooa.epistemic import (
    CandidateGoalStatus,
    GoalLedger,
    GoalToolSurface,
    ObjectiveAssessment,
    RequestBuilderArguments,
)
from tycho_nooa.models import GameTurn


def test_candidate_goal_is_durable_but_never_canonicalizes_itself(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "game"
    evidence = workspace / "level_0" / "turn_003.txt"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("host-authored frame")
    world_model = workspace / "world_model.py"
    world_model.write_text('def outcome(state):\n    return "ongoing"\n')
    before = world_model.read_bytes()

    ledger = GoalLedger(workspace)
    ledger.register_authoritative_evidence("level_0/turn_003.txt")
    proposed = ledger.propose_goal(
        predicate="body colour 4 overlaps target colour 11",
        evidence_refs=["level_0/turn_003.txt"],
        falsifier="overlap occurs without levels_completed increasing",
        level=0,
        turn_in_level=3,
    )
    selected = ledger.update_goal(
        proposed.id,
        status=CandidateGoalStatus.SELECTED,
        evidence_refs=["level_0/turn_003.txt"],
        note="worth asking the builder to encode and test",
        level=0,
        turn_in_level=3,
    )

    assert selected.status is CandidateGoalStatus.SELECTED
    assert world_model.read_bytes() == before
    assert GoalLedger(workspace).get(proposed.id) == selected
    events = [
        json.loads(line)
        for line in (workspace / "epistemic" / "candidate-goal-events.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [event["event"] for event in events] == ["goal_proposed", "goal_updated"]


def test_candidate_goal_rejects_model_authored_or_missing_evidence(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "game"
    notes = workspace / "notes" / "actor_beliefs.md"
    notes.parent.mkdir(parents=True)
    notes.write_text("I think blue is a target")
    ledger = GoalLedger(workspace)
    with pytest.raises(ValueError, match="authoritative public evidence"):
        ledger.register_authoritative_evidence("notes/actor_beliefs.md")

    for ref in ["notes/actor_beliefs.md", "level_0/turn_999.txt", "../outside"]:
        with pytest.raises(ValueError, match="authoritative public evidence"):
            ledger.propose_goal(
                predicate="reach blue",
                evidence_refs=[ref],
                falsifier="reaching blue does not complete the level",
                level=0,
                turn_in_level=0,
            )


def test_authoritative_evidence_rejects_content_changed_after_registration(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "game"
    evidence = workspace / "level_0" / "turn_000.txt"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("host-authored frame")
    ledger = GoalLedger(workspace)
    ledger.register_authoritative_evidence("level_0/turn_000.txt")

    evidence.write_text("replaced after registration")

    with pytest.raises(ValueError, match="changed after registration"):
        ledger.propose_goal(
            predicate="reach the target",
            evidence_refs=["level_0/turn_000.txt"],
            falsifier="target contact is nonterminal",
            level=0,
            turn_in_level=0,
        )


def test_goal_update_revalidates_existing_evidence_when_no_refs_are_added(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "game"
    evidence = workspace / "level_0" / "turn_000.txt"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("host-authored frame")
    ledger = GoalLedger(workspace)
    ledger.register_authoritative_evidence("level_0/turn_000.txt")
    goal = ledger.propose_goal(
        predicate="reach the target",
        evidence_refs=["level_0/turn_000.txt"],
        falsifier="target contact is nonterminal",
        level=0,
        turn_in_level=0,
    )
    evidence.write_text("changed before no-op evidence update")

    with pytest.raises(ValueError, match="changed after registration"):
        ledger.update_goal(
            goal.id,
            status=CandidateGoalStatus.PROPOSED,
            evidence_refs=[],
            note="retain without adding evidence",
            level=0,
            turn_in_level=0,
        )


def test_action_admission_revalidates_independent_review_evidence(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "game"
    goal_evidence = workspace / "level_0" / "turn_000.txt"
    review_evidence = workspace / "level_0" / "turn_001.txt"
    goal_evidence.parent.mkdir(parents=True)
    goal_evidence.write_text("host-authored proposal frame")
    review_evidence.write_text("host-authored review frame")
    ledger = GoalLedger(workspace)
    ledger.register_authoritative_evidence("level_0/turn_000.txt")
    ledger.register_authoritative_evidence("level_0/turn_001.txt")
    goal = ledger.propose_goal(
        predicate="reach the target",
        evidence_refs=["level_0/turn_000.txt"],
        falsifier="target contact is nonterminal",
        level=0,
        turn_in_level=0,
    )
    review = ledger.record_builder_review(
        candidate_goal_id=goal.id,
        evidence_refs=["level_0/turn_001.txt"],
        report="candidate_goal_assessment: inconclusive",
        level=0,
        turn_in_level=0,
    )
    ledger.update_goal(
        goal.id,
        status=CandidateGoalStatus.SELECTED,
        evidence_refs=[],
        note="select after review",
        level=0,
        turn_in_level=0,
        resolved_review_id=review.id,
    )
    review_evidence.write_text("changed after independent review")

    required = GoalToolSurface(ledger).required_before_action()

    assert required is not None
    assert "stale evidence" in required
    assert "changed after registration" in required


def test_rejected_goal_does_not_retain_action_authority(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "game"
    evidence = workspace / "level_0" / "turn_000.txt"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("host-authored frame")
    ledger = GoalLedger(workspace)
    ledger.register_authoritative_evidence("level_0/turn_000.txt")
    goal = ledger.propose_goal(
        predicate="reach the target",
        evidence_refs=["level_0/turn_000.txt"],
        falsifier="target contact is nonterminal",
        level=0,
        turn_in_level=0,
    )
    ledger.update_goal(
        goal.id,
        status=CandidateGoalStatus.REJECTED,
        evidence_refs=[],
        note="discarded",
        level=0,
        turn_in_level=0,
    )
    evidence.write_text("changed after rejection")

    assert GoalToolSurface(ledger).required_before_action() is None


def test_authoritative_evidence_commitment_survives_restart(tmp_path: Path) -> None:
    workspace = tmp_path / "game"
    evidence = workspace / "level_0" / "turn_000.txt"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("host-authored frame")
    GoalLedger(workspace).register_authoritative_evidence("level_0/turn_000.txt")

    payload = json.loads(
        (workspace / "epistemic" / "authoritative-evidence.json").read_text()
    )
    commitment = payload["files"]["level_0/turn_000.txt"]
    assert payload["schema_version"] == 1
    assert len(commitment["sha256"]) == 64
    assert commitment["size_bytes"] == len("host-authored frame")

    restarted = GoalLedger(workspace)
    evidence.write_text("changed after restart")
    with pytest.raises(ValueError, match="changed after registration"):
        restarted.validate_evidence_refs(["level_0/turn_000.txt"])


def test_legacy_path_only_evidence_index_fails_closed(tmp_path: Path) -> None:
    workspace = tmp_path / "game"
    index = workspace / "epistemic" / "authoritative-evidence.json"
    index.parent.mkdir(parents=True)
    index.write_text('["level_0/turn_000.txt"]\n')

    with pytest.raises(TypeError, match="legacy path-only evidence index"):
        GoalLedger(workspace)


def test_objective_builder_request_requires_a_candidate_goal(tmp_path: Path) -> None:
    workspace = tmp_path / "game"
    evidence = workspace / "level_0" / "turn_000.txt"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("host-authored frame")
    ledger = GoalLedger(workspace)
    ledger.register_authoritative_evidence("level_0/turn_000.txt")
    surface = GoalToolSurface(ledger)

    with pytest.raises(ValueError, match="candidate_goal_id"):
        surface.builder_request(
            RequestBuilderArguments(
                focus="objective",
                evidence_refs=["level_0/turn_000.txt"],
            )
        )


def test_objective_review_closes_the_required_goal_resolution_loop(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "game"
    evidence = workspace / "level_0" / "turn_000.txt"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("host-authored frame")
    ledger = GoalLedger(workspace)
    ledger.register_authoritative_evidence("level_0/turn_000.txt")
    surface = GoalToolSurface(ledger)
    surface.begin_turn(
        GameTurn(
            grid=[[0]],
            available_actions=["UP"],
            level=0,
            turn_in_level=0,
        )
    )
    proposed = json.loads(
        surface.propose_goal(
            predicate="colour 4 must overlap colour 11",
            evidence_refs=["level_0/turn_000.txt"],
            falsifier="overlap occurs without completing the level",
        )
    )

    assert "request_builder" in surface.required_before_action()
    with pytest.raises(ValueError, match="request_builder"):
        surface.update_goal(
            proposed["id"],
            CandidateGoalStatus.SELECTED,
            ["level_0/turn_000.txt"],
            "premature selection",
        )

    request = RequestBuilderArguments(
        focus="objective",
        evidence_refs=["level_0/turn_000.txt"],
        candidate_goal_id=proposed["id"],
    )
    surface.builder_request(request)
    review = surface.record_builder_review(
        request,
        report=(
            "candidate_goal_assessment: supported\n"
            "candidate_goal_reason: shapes are congruent\n"
            "candidate_goal_route: DOWN x11, LEFT x6\n"
            "recommended_action: DOWN\n"
        ),
    )

    assert review.assessment is ObjectiveAssessment.SUPPORTED
    assert review.recommended_route == "DOWN x11, LEFT x6"
    assert "update_goal" in surface.required_before_action()

    selected = json.loads(
        surface.update_goal(
            proposed["id"],
            CandidateGoalStatus.SELECTED,
            ["level_0/turn_000.txt"],
            "builder evidence supports testing this objective",
        )
    )
    assert selected["resolved_review_id"] == review.id
    assert surface.required_before_action() is None
    assert GoalLedger(workspace).get(proposed["id"]).status is (
        CandidateGoalStatus.SELECTED
    )


def test_unavailable_builder_review_can_be_acknowledged_without_false_selection(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "game"
    evidence = workspace / "level_0" / "turn_000.txt"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("host-authored frame")
    ledger = GoalLedger(workspace)
    ledger.register_authoritative_evidence("level_0/turn_000.txt")
    surface = GoalToolSurface(ledger)
    surface.begin_turn(
        GameTurn(
            grid=[[0]],
            available_actions=["UP"],
            level=0,
            turn_in_level=0,
        )
    )
    proposed = json.loads(
        surface.propose_goal(
            predicate="reach the target",
            evidence_refs=["level_0/turn_000.txt"],
            falsifier="target contact is nonterminal",
        )
    )
    request = RequestBuilderArguments(
        focus="objective",
        evidence_refs=["level_0/turn_000.txt"],
        candidate_goal_id=proposed["id"],
    )

    review = surface.record_builder_review(
        request,
        report="BuilderDeadlineExceeded: no objective allocation remained",
        unavailable=True,
    )
    surface.update_goal(
        proposed["id"],
        CandidateGoalStatus.PROPOSED,
        ["level_0/turn_000.txt"],
        "review unavailable; retain as an unselected hypothesis",
    )

    assert review.assessment is ObjectiveAssessment.UNAVAILABLE
    assert surface.required_before_action() is None


def test_update_goal_may_keep_existing_evidence_refs(tmp_path: Path) -> None:
    workspace = tmp_path / "game"
    evidence = workspace / "level_0" / "turn_000.txt"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("host-authored frame")
    ledger = GoalLedger(workspace)
    ledger.register_authoritative_evidence("level_0/turn_000.txt")
    surface = GoalToolSurface(ledger)
    surface.begin_turn(
        GameTurn(grid=[[0]], available_actions=["UP"], level=0, turn_in_level=0)
    )
    proposed = json.loads(
        surface.propose_goal(
            predicate="reach the target",
            evidence_refs=["level_0/turn_000.txt"],
            falsifier="target contact is nonterminal",
        )
    )
    request = RequestBuilderArguments(
        focus="objective",
        evidence_refs=["level_0/turn_000.txt"],
        candidate_goal_id=proposed["id"],
    )
    surface.builder_request(request)
    surface.record_builder_review(
        request,
        report="candidate_goal_assessment: inconclusive\nrecommended_action: UP\n",
    )

    updated = json.loads(
        surface.update_goal(
            proposed["id"],
            CandidateGoalStatus.PROPOSED,
            [],
            "retain without citing new frames",
        )
    )

    assert updated["evidence_refs"] == ["level_0/turn_000.txt"]
    assert surface.required_before_action() is None


def test_observation_advisory_carries_required_step_and_selected_review(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "game"
    evidence = workspace / "level_0" / "turn_000.txt"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("host-authored frame")
    ledger = GoalLedger(workspace)
    ledger.register_authoritative_evidence("level_0/turn_000.txt")
    surface = GoalToolSurface(ledger)
    surface.begin_turn(
        GameTurn(grid=[[0]], available_actions=["UP"], level=0, turn_in_level=0)
    )
    assert surface.observation_advisory() == []

    proposed = json.loads(
        surface.propose_goal(
            predicate="colour 4 must overlap colour 11",
            evidence_refs=["level_0/turn_000.txt"],
            falsifier="overlap occurs without completing the level",
        )
    )
    pending = "\n".join(surface.observation_advisory())
    assert "request_builder" in pending
    assert proposed["id"] in pending

    request = RequestBuilderArguments(
        focus="objective",
        evidence_refs=["level_0/turn_000.txt"],
        candidate_goal_id=proposed["id"],
    )
    surface.builder_request(request)
    surface.record_builder_review(
        request,
        report=(
            "candidate_goal_assessment: supported\n"
            "candidate_goal_reason: shapes are congruent\n"
            "candidate_goal_route: DOWN x2\n"
            "recommended_action: DOWN\n"
        ),
    )
    surface.update_goal(
        proposed["id"],
        CandidateGoalStatus.SELECTED,
        ["level_0/turn_000.txt"],
        "builder evidence supports testing this objective",
    )

    selected = "\n".join(surface.observation_advisory())
    assert "required_next_step" not in selected
    assert proposed["id"] in selected
    assert "selected" in selected
    assert "colour 4 must overlap colour 11" in selected
    assert "supported" in selected
    assert "DOWN" in selected
    assert "DOWN x2" in selected
