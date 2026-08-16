from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from theory_workbench import (
    TheoryArtifactSnapshot,
    TheoryWorkbenchView,
    TheoryWorkbenchViewBuilder,
    TransitionEvaluationFact,
    theory_snapshot_from_lane_view,
)

EXPERIMENT = Path(__file__).resolve().parents[2]
E0_PATH = EXPERIMENT / "e0_evidence.py"
E0_SPEC = importlib.util.spec_from_file_location("wm2_test_e0", E0_PATH)
assert E0_SPEC is not None and E0_SPEC.loader is not None
E0 = importlib.util.module_from_spec(E0_SPEC)
sys.modules[E0_SPEC.name] = E0
E0_SPEC.loader.exec_module(E0)


def observation(
    grid: tuple[tuple[int, ...], ...],
    *,
    full_reset: bool = False,
    public_state: str = "NOT_FINISHED",
    levels_completed: int = 0,
) -> object:
    return E0.PublicObservationInput(
        frames=(grid,),
        public_state=public_state,
        levels_completed=levels_completed,
        full_reset=full_reset,
        legal_actions=("UP", "DOWN", "LEFT", "RIGHT"),
    )


def empty_theory(evidence_snapshot_id: str) -> TheoryArtifactSnapshot:
    return TheoryArtifactSnapshot.create(
        installed_evidence_snapshot_id=evidence_snapshot_id,
        current_model_version_id=None,
        contract_kind=None,
        evaluation_evidence_snapshot_id=None,
        evaluation_stale=False,
        candidate_status=None,
        operational_status=None,
        prediction_coverage=None,
        transition_results=(),
        first_divergence_event_id=None,
        latest_edit_status=None,
    )


def builder(root: Path) -> TheoryWorkbenchViewBuilder:
    def read_grid(digest: str) -> tuple[tuple[int, ...], ...]:
        value = json.loads((root / "blobs" / f"grid-{digest}.json").read_text())
        return tuple(tuple(row) for row in value)

    return TheoryWorkbenchViewBuilder(read_grid=read_grid)


def checked_view(
    root: Path, evidence: object, theory: TheoryArtifactSnapshot
) -> TheoryWorkbenchView:
    workbench = builder(root)
    one = workbench.build(evidence, theory)
    two = workbench.build(evidence, theory)
    assert asdict(one) == asdict(two)
    assert one.evidence_snapshot_id == evidence.snapshot_id
    assert one.theory_snapshot_id == theory.snapshot_id
    rendered = one.render()
    assert len(rendered.encode("utf-8")) <= 6_144
    for card in one.cards:
        assert card.evidence_snapshot_id == evidence.snapshot_id
        assert card.theory_snapshot_id == theory.snapshot_id
        assert card.does_not_imply
        assert any(
            word in " ".join(card.possible_responses)
            for word in ("retain", "keep", "unresolved")
        )
        assert card.code not in rendered
    for forbidden in (
        "object identity",
        "correspondence",
        "displacement",
        "action recommendation",
    ):
        assert forbidden not in rendered.lower()
    return one


def test_turn_zero_explains_absence_without_an_internal_status_label(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    ledger = E0.HostEvidenceLedger(root, run_id="workbench-zero")
    ledger.record_initial_reset(observation(((0, 0), (0, 0))))
    evidence = ledger.snapshot()

    view = checked_view(root, evidence, empty_theory(evidence.snapshot_id))
    rendered = view.render()

    assert view.condition_codes == (
        "no_transition_evidence",
        "nonordinary_boundary",
    )
    assert "No ordinary action result is available yet" in rendered
    assert "no_transition_evidence" not in rendered
    assert "Nothing here requires an edit" in rendered
    assert evidence.snapshot_id in rendered
    assert len(rendered.encode("utf-8")) <= 6_144


def test_repeated_action_facts_are_exact_without_claiming_a_shared_rule(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    ledger = E0.HostEvidenceLedger(root, run_id="workbench-repeat")
    ledger.record_initial_reset(observation(((0, 0, 0), (0, 0, 0))))
    ledger.record_action_result(
        global_turn=0,
        action=E0.Action("RIGHT"),
        successor=observation(((0, 1, 0), (0, 0, 0))),
    )
    evidence = ledger.record_action_result(
        global_turn=1,
        action=E0.Action("RIGHT"),
        successor=observation(((0, 0, 1), (0, 0, 0))),
    ).snapshot

    view = checked_view(root, evidence, empty_theory(evidence.snapshot_id))
    rendered = view.render()

    assert view.action_counts == (("RIGHT", 2),)
    assert view.condition_codes == (
        "evidence_without_commitment",
        "repeated_action_evidence",
    )
    assert [fact.changed_cell_count for fact in view.recent_transitions] == [1, 2]
    assert view.recent_transitions[-1].changed_bounds == (1, 0, 3, 1)
    assert "It does not imply: the transitions follow the same rule" in rendered
    assert "RIGHT: 2" in rendered
    assert "model the movement" not in rendered


def test_reset_boundary_is_not_stitched_into_the_new_evidence_segment(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    ledger = E0.HostEvidenceLedger(root, run_id="workbench-reset")
    ledger.record_initial_reset(observation(((0, 0), (0, 0))))
    ledger.record_action_result(
        global_turn=0,
        action=E0.Action("RIGHT"),
        successor=observation(((0, 1), (0, 0))),
    )
    evidence = ledger.record_action_result(
        global_turn=1,
        action=E0.Action("RESET"),
        successor=observation(((0, 0), (0, 0)), full_reset=True),
    ).snapshot

    view = checked_view(root, evidence, empty_theory(evidence.snapshot_id))

    assert view.ordinary_transition_count == 0
    assert view.recent_transitions == ()
    assert "nonordinary_boundary" in view.condition_codes
    assert "must not be replayed as one ordinary transition segment" in view.render()
    assert "attempt-" not in view.render()


@pytest.mark.parametrize(
    ("public_state", "levels_completed", "expected_outcome"),
    [("WIN", 1, "level_complete"), ("GAME_OVER", 0, "game_over")],
)
def test_terminal_boundaries_are_not_replayed_as_ordinary_transitions(
    tmp_path: Path,
    public_state: str,
    levels_completed: int,
    expected_outcome: str,
) -> None:
    root = tmp_path / "evidence"
    ledger = E0.HostEvidenceLedger(root, run_id=f"workbench-{expected_outcome}")
    ledger.record_initial_reset(observation(((0, 0), (0, 0))))
    evidence = ledger.record_action_result(
        global_turn=0,
        action=E0.Action("UP"),
        successor=observation(
            ((1, 0), (0, 0)),
            public_state=public_state,
            levels_completed=levels_completed,
        ),
    ).snapshot

    view = checked_view(root, evidence, empty_theory(evidence.snapshot_id))

    assert evidence.transitions[-1].outcome == expected_outcome
    assert view.ordinary_transition_count == 0
    assert view.recent_transitions == ()
    assert "nonordinary_boundary" in view.condition_codes


def test_prior_segment_abstention_does_not_create_a_current_opportunity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    ledger = E0.HostEvidenceLedger(root, run_id="workbench-old-abstain")
    ledger.record_initial_reset(observation(((0, 0), (0, 0))))
    first = ledger.record_action_result(
        global_turn=0,
        action=E0.Action("UP"),
        successor=observation(((0, 0), (0, 0))),
    ).snapshot.transitions[0]
    evidence = ledger.record_action_result(
        global_turn=1,
        action=E0.Action("RESET"),
        successor=observation(((0, 0), (0, 0)), full_reset=True),
    ).snapshot
    theory = TheoryArtifactSnapshot.create(
        installed_evidence_snapshot_id=evidence.snapshot_id,
        current_model_version_id="model-version-00000000",
        contract_kind="partial_predictor",
        evaluation_evidence_snapshot_id=evidence.snapshot_id,
        evaluation_stale=False,
        candidate_status="active_inconclusive",
        operational_status="valid",
        prediction_coverage=0.0,
        transition_results=(
            TransitionEvaluationFact(
                source_event_id=first.source_event_id,
                action="UP",
                claim="whole_action_abstention",
                known_cells=0,
                total_cells=4,
                matched=None,
            ),
        ),
        first_divergence_event_id=None,
        latest_edit_status="active_inconclusive",
    )

    view = checked_view(root, evidence, theory)

    assert "whole_action_abstention" not in view.condition_codes
    assert "predict(...) returned None" not in view.render()


def test_abstention_is_explained_in_plain_language_with_nonimplication(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    ledger = E0.HostEvidenceLedger(root, run_id="workbench-abstain")
    ledger.record_initial_reset(observation(((0, 0), (0, 0))))
    evidence = ledger.record_action_result(
        global_turn=0,
        action=E0.Action("UP"),
        successor=observation(((0, 0), (0, 0))),
    ).snapshot
    transition_id = evidence.transitions[0].source_event_id
    theory = TheoryArtifactSnapshot.create(
        installed_evidence_snapshot_id=evidence.snapshot_id,
        current_model_version_id="model-version-00000000",
        contract_kind="partial_predictor",
        evaluation_evidence_snapshot_id=evidence.snapshot_id,
        evaluation_stale=False,
        candidate_status="active_inconclusive",
        operational_status="valid",
        prediction_coverage=0.0,
        transition_results=(
            TransitionEvaluationFact(
                source_event_id=transition_id,
                action="UP",
                claim="whole_action_abstention",
                known_cells=0,
                total_cells=4,
                matched=None,
            ),
        ),
        first_divergence_event_id=None,
        latest_edit_status="active_inconclusive",
    )

    view = checked_view(root, evidence, theory)
    rendered = view.render()

    assert "whole_action_abstention" in view.condition_codes
    assert "predict(...) returned None for UP" in rendered
    assert "neither supported nor contradicted" in rendered
    assert "It does not imply: UP is unpredictable" in rendered
    assert "Claimed cells in evaluated current-segment transitions: 0 of 4" in rendered
    assert (
        "Whole-action abstentions in evaluated current-segment transitions: 1"
        in rendered
    )
    assert "boundary none" in rendered
    assert f"reference {evidence.snapshot_id}" in rendered
    assert "whole_action_abstention" not in rendered


def test_lane_view_adapter_freezes_only_workbench_relevant_state() -> None:
    lane_view = {
        "current_model_version_id": "model-version-00000003",
        "contract_kind": "partial_predictor",
        "installed_evidence_snapshot_id": "snapshot-new",
        "evaluation_evidence_snapshot_id": "snapshot-old",
        "evaluation_stale": True,
        "latest_evaluation": {
            "candidate_status": "active_supported",
            "operational": {"status": "valid"},
            "dynamics": {
                "prediction_coverage": 0.25,
                "transition_results": [
                    {
                        "source_event_id": "event-7",
                        "action": "LEFT",
                        "claim": "partial_cell_claim",
                        "known_cells": 1,
                        "total_cells": 4,
                        "matched": True,
                    }
                ],
            },
            "first_divergence": None,
        },
        "source": "must not enter the snapshot",
        "quick_theory_template": "must not enter the snapshot",
    }

    snapshot = theory_snapshot_from_lane_view(
        lane_view, latest_edit_status="rejected_operationally"
    )

    assert snapshot.current_model_version_id == "model-version-00000003"
    assert snapshot.evaluation_stale is True
    assert snapshot.prediction_coverage == 0.25
    assert snapshot.transition_results[0].source_event_id == "event-7"
    assert snapshot.latest_edit_status == "rejected_operationally"
    assert "must not enter" not in repr(snapshot)


def test_multiple_conditions_use_fixed_priority_plain_language_and_omission(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    ledger = E0.HostEvidenceLedger(root, run_id="workbench-priority")
    ledger.record_initial_reset(observation(((0, 0), (0, 0))))
    ledger.record_action_result(
        global_turn=0,
        action=E0.Action("LEFT"),
        successor=observation(((1, 0), (0, 0))),
    )
    evidence = ledger.record_action_result(
        global_turn=1,
        action=E0.Action("LEFT"),
        successor=observation(((0, 1), (0, 0))),
    ).snapshot
    first = evidence.transitions[0].source_event_id
    theory = TheoryArtifactSnapshot.create(
        installed_evidence_snapshot_id=evidence.snapshot_id,
        current_model_version_id="model-version-00000001",
        contract_kind="partial_predictor",
        evaluation_evidence_snapshot_id="snapshot-older",
        evaluation_stale=True,
        candidate_status="active_contradicted",
        operational_status="valid",
        prediction_coverage=0.25,
        transition_results=(
            TransitionEvaluationFact(
                source_event_id=first,
                action="LEFT",
                claim="partial_cell_claim",
                known_cells=1,
                total_cells=4,
                matched=False,
            ),
        ),
        first_divergence_event_id=first,
        latest_edit_status="rejected_operationally",
    )

    one = checked_view(root, evidence, theory)
    two = builder(root).build(evidence, theory)
    rendered = one.render()

    assert one.view_digest == two.view_digest
    assert [card.code for card in one.cards] == [
        "empirical_contradiction",
        "operational_rejection",
        "partial_cell_claim",
    ]
    assert one.omitted_card_count == 2
    assert one.omitted_card_codes == (
        "new_evidence_since_evaluation",
        "repeated_action_evidence",
    )
    assert "active_contradicted" not in rendered
    assert "Operationally valid; replay contradicted" in rendered
    assert "greater coverage is not necessarily better" in rendered
    assert "Claimed cells in evaluated current-segment transitions: 1 of 4" in rendered
    assert f"reference snapshot-older, {first}" in rendered
    assert "2 additional mechanical condition(s)" in rendered


def test_render_byte_bound_fails_closed_instead_of_silently_truncating(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    ledger = E0.HostEvidenceLedger(root, run_id="x" * 7_000)
    ledger.record_initial_reset(observation(((0, 0), (0, 0))))
    evidence = ledger.record_action_result(
        global_turn=0,
        action=E0.Action("RIGHT"),
        successor=observation(((0, 1), (0, 0))),
    ).snapshot

    with pytest.raises(ValueError, match="frozen byte bound"):
        builder(root).build(evidence, empty_theory(evidence.snapshot_id))
