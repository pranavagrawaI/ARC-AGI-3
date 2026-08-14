from __future__ import annotations

from helpers import claim
from host.gates import GateTracker
from host.protocol import GroundingVerdict


def verdict(claim_id: str, status: str = "supported", promotable: bool = True) -> GroundingVerdict:
    return GroundingVerdict(
        claim_id=claim_id,
        commitment_id="commitment-" + claim_id,
        status=status,  # type: ignore[arg-type]
        checks=("CHECK",),
        action_specificity="supported" if promotable else "inconclusive",
        contrast_commitment_ids=("prior",) if promotable else (),
        promotion_credit=1 if promotable else 0,
        promotable=promotable,
    )


def test_three_linked_informative_effects_reach_g2_and_stop_without_new_api() -> None:
    tracker = GateTracker(level=0)
    current = claim("frame-0")
    for index, expected in enumerate(("G1", "G1", "G2")):
        tracker.accept_claim(current, observation_exists=True, evidence_refs_exist=True)
        assert tracker.apply_verdict(verdict(current.claim_id)) == expected
        if index < 2:
            current = claim(f"frame-{index + 1}", parent=current)
    assert tracker.transition_capability_eligible is True
    assert tracker.paused_for_review is True
    assert tracker.supported_informative_effects == 3


def test_unchanged_does_not_promote_or_consume_the_alternative() -> None:
    tracker = GateTracker(level=0)
    current = claim()
    tracker.accept_claim(current, observation_exists=True, evidence_refs_exist=True)
    assert tracker.apply_verdict(verdict(current.claim_id, promotable=False)) == "G0"
    assert tracker.supported_informative_effects == 0
    assert tracker.alternative_used is False


def test_one_inconclusive_allows_one_alternative_and_second_stops() -> None:
    tracker = GateTracker(level=0)
    current = claim()
    tracker.accept_claim(current, observation_exists=True, evidence_refs_exist=True)
    assert tracker.apply_verdict(verdict(current.claim_id, status="inconclusive", promotable=False)) == "G0"
    assert tracker.alternative_used is True
    successor = claim("frame-b", parent=current)
    tracker.accept_claim(successor, observation_exists=True, evidence_refs_exist=True)
    assert tracker.apply_verdict(verdict(successor.claim_id, status="inconclusive", promotable=False)) == "STOPPED"
    assert tracker.paused_for_review


def test_contradiction_invalidates_and_new_level_resets_all_qualification() -> None:
    tracker = GateTracker(level=0)
    current = claim()
    tracker.accept_claim(current, observation_exists=True, evidence_refs_exist=True)
    assert tracker.apply_verdict(verdict(current.claim_id, status="contradicted", promotable=False)) == "INVALIDATED"
    tracker.reset_for_level(1)
    assert tracker.gate == "EMPTY"
    assert tracker.lineage_id is None
    assert tracker.transition_capability_eligible is False


def test_autonomous_cycle_reset_retains_audit_history_without_unlocking_capability() -> None:
    tracker = GateTracker(level=0)
    current = claim()
    tracker.accept_claim(current, observation_exists=True, evidence_refs_exist=True)
    assert tracker.apply_verdict(
        verdict(current.claim_id, status="contradicted", promotable=False)
    ) == "INVALIDATED"
    tracker.begin_new_grounding_cycle(reason="contradicted")
    assert tracker.gate == "EMPTY" and tracker.paused_for_review is False
    assert tracker.tip_claim_id is None and tracker.transition_capability_eligible is False
    assert tracker.history[-1] == {
        "event": "grounding_cycle_reset",
        "reason": "contradicted",
        "previous_gate": "INVALIDATED",
    }
