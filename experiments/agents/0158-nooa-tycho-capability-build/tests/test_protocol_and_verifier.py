from __future__ import annotations

import pytest
from helpers import claim, commitment_for, mask
from host.protocol import (
    ProtocolError,
    grounding_claim_from_draft,
    predicted_effect_from_record,
    probe_commitment_from_draft,
)
from host.verifier import account_visible_change, verify_effect


def heuristic(
    evidence_ref: str,
    *,
    name: str = "candidate_motion",
    role: str = "active",
) -> dict:
    return {
        "name": name,
        "belief": "The selected pattern may respond to the chosen action.",
        "scope": "Current level and tested transitions only.",
        "evidence_refs": [evidence_ref],
        "decision_role": role,
        "reconsider_when": "A prospective effect mismatches an observed transition.",
        "epistemic_status": "provisional",
    }


def heuristic_update(*, name: str = "candidate_motion", role: str = "active") -> dict:
    record = heuristic("host-owned", name=name, role=role)
    record.pop("evidence_refs")
    record.pop("epistemic_status")
    return record


def test_model_cannot_supply_claim_identity_or_an_action() -> None:
    draft = {
        "observation_id": "frame-a",
        "primary_region": mask(1, 1),
        "alternatives": [mask(0, 0)],
        "evidence_refs": ["frame-a"],
        "parent_claim_id": None,
        "claim_id": "model-chosen",
        "action": 1,
    }
    with pytest.raises(ProtocolError, match="extra"):
        grounding_claim_from_draft(draft)


def test_claim_may_declare_no_alternative_but_supplied_masks_must_be_distinct() -> None:
    draft = {
        "observation_id": "frame-a",
        "primary_region": mask(1, 1),
        "alternatives": [],
        "evidence_refs": ["frame-a"],
        "parent_claim_id": None,
    }
    claim = grounding_claim_from_draft(draft)
    assert claim.alternatives == ()

    draft["alternatives"] = [draft["primary_region"]]
    with pytest.raises(ProtocolError, match="must be distinct"):
        grounding_claim_from_draft(draft)


def test_translate_is_supported_only_for_exact_observable_primary_motion() -> None:
    candidate = claim(primary=mask(1, 1))
    commitment = commitment_for(
        candidate,
        {"kind": "translate", "source": "primary_region", "d_row": 0, "d_col": 1},
    )
    before = [[0, 0, 0], [0, 7, 0], [0, 0, 0]]
    after = [[0, 0, 0], [0, 0, 7], [0, 0, 0]]
    verdict = verify_effect(candidate, commitment, before, after)
    assert verdict.status == "supported"
    assert verdict.action_specificity == "inconclusive"
    assert verdict.promotable is False
    assert verdict.checks == ("PRIMARY_TRANSLATION_EXACT",)

    no_change = verify_effect(candidate, commitment, before, before)
    assert no_change.status == "contradicted"  # destination does not contain the source value
    assert all("ROW" not in check and "COL" not in check for check in no_change.checks)


def test_unchanged_is_supported_but_never_promotable() -> None:
    candidate = claim(primary=mask(1, 1))
    commitment = commitment_for(
        candidate,
        {"kind": "unchanged", "source": "primary_region"},
    )
    grid = [[0, 0, 0], [0, 7, 0], [0, 0, 0]]
    verdict = verify_effect(candidate, commitment, grid, grid)
    assert verdict.status == "supported"
    assert verdict.promotable is False


def test_replace_must_be_local_bounded_and_change_an_asserted_primary_cell() -> None:
    candidate = claim(primary=mask(1, 1, 2, 2))
    local = commitment_for(
        candidate,
        {
            "kind": "replace",
            "source": "primary_region",
            "expected_after": mask(1, 1, 2, 2),
            "assertions": [{"row": 1, "col": 1, "value": 4}],
        },
    )
    before = [[0] * 5 for _ in range(5)]
    before[1][1] = 7
    after = [row[:] for row in before]
    after[1][1] = 4
    verdict = verify_effect(candidate, local, before, after)
    assert verdict.status == "supported" and not verdict.promotable

    unrelated = commitment_for(
        candidate,
        {
            "kind": "replace",
            "source": "primary_region",
            "expected_after": mask(4, 4),
            "assertions": [{"row": 4, "col": 4, "value": 2}],
        },
    )
    unrelated_verdict = verify_effect(candidate, unrelated, before, after)
    assert unrelated_verdict.status == "contradicted"
    assert unrelated_verdict.checks == ("REPLACEMENT_NOT_LOCAL_TO_PRIMARY",)

    with pytest.raises(ProtocolError, match="64"):
        predicted_effect_from_record(
            {
                "kind": "replace",
                "source": "primary_region",
                "expected_after": mask(0, 0, 9, 8),
                "assertions": [{"row": 0, "col": 0, "value": 1}],
            }
        )


def test_replace_assertions_cannot_address_outside_the_expected_region() -> None:
    with pytest.raises(ProtocolError, match="inside"):
        predicted_effect_from_record(
            {
                "kind": "replace",
                "source": "primary_region",
                "expected_after": mask(1, 1),
                "assertions": [{"row": 0, "col": 0, "value": 1}],
            }
        )


def test_commitment_requires_a_distinct_action_effect_contrast() -> None:
    candidate = claim(primary=mask(1, 1))
    selected = {"kind": "translate", "source": "primary_region", "d_row": -1, "d_col": 0}
    base = {
        "claim_id": candidate.claim_id,
        "action": 1,
        "data": None,
        "predicted_effect": selected,
    }
    with pytest.raises(ProtocolError, match="1..3"):
        probe_commitment_from_draft({**base, "alternative_predictions": []}, claim=candidate)
    with pytest.raises(ProtocolError, match="identifiers must be distinct"):
        probe_commitment_from_draft(
            {
                **base,
                "alternative_predictions": [
                    {"action": 1, "data": None, "predicted_effect": {"kind": "unchanged", "source": "primary_region"}}
                ],
            },
            claim=candidate,
        )
    with pytest.raises(ProtocolError, match="different primary-region effect"):
        probe_commitment_from_draft(
            {
                **base,
                "alternative_predictions": [
                    {"action": 2, "data": None, "predicted_effect": selected}
                ],
            },
            claim=candidate,
        )


def test_shifted_local_replacements_have_the_same_position_invariant_signature() -> None:
    candidate = claim(primary=mask(61, 13, 2, 2))
    selected = {
        "kind": "replace",
        "source": "primary_region",
        "expected_after": mask(61, 13, 2, 1),
        "assertions": [{"row": 61, "col": 13, "value": 3}],
    }
    shifted = {
        "kind": "replace",
        "source": "primary_region",
        "expected_after": mask(61, 14, 2, 1),
        "assertions": [{"row": 61, "col": 14, "value": 3}],
    }
    with pytest.raises(ProtocolError, match="different primary-region effect"):
        probe_commitment_from_draft(
            {
                "claim_id": candidate.claim_id,
                "action": 1,
                "data": None,
                "predicted_effect": selected,
                "alternative_predictions": [
                    {"action": 2, "data": None, "predicted_effect": shifted}
                ],
            },
            claim=candidate,
        )


def test_change_accounting_preserves_residuals_without_selecting_their_meaning() -> None:
    candidate = claim(primary=mask(1, 1))
    before = [[0, 0, 0], [0, 7, 0], [0, 0, 0]]
    after = [[9, 0, 0], [0, 7, 0], [0, 0, 8]]
    accounting = account_visible_change(
        candidate,
        before,
        after,
        transition_id="transition-a",
    )
    assert accounting == {
        "schema": "0158-change-accounting-v1",
        "transition_id": "transition-a",
        "changed_cells_total": 2,
        "changed_cells_inside_primary": 0,
        "changed_cells_outside_primary": 2,
        "primary_region_changed": False,
        "outside_primary_changed": True,
    }
    assert not any(key in accounting for key in ("object", "causal_role", "correct_region"))


def test_change_accounting_preserves_residual_when_primary_effect_is_also_present() -> None:
    candidate = claim(primary=mask(1, 1))
    accounting = account_visible_change(
        candidate,
        [[0, 0, 0], [0, 7, 0], [0, 0, 0]],
        [[9, 0, 0], [0, 8, 0], [0, 0, 0]],
        transition_id="transition-b",
    )
    assert accounting["changed_cells_inside_primary"] == 1
    assert accounting["changed_cells_outside_primary"] == 1
    assert accounting["primary_region_changed"] is True
    assert accounting["outside_primary_changed"] is True


def test_claim_after_probe_must_author_an_evidence_bound_update() -> None:
    latest = {
        "evidence_id": "probe-result-a",
        "change_accounting": {"outside_primary_changed": True},
    }
    draft = {
        "observation_id": "frame-b",
        "primary_region": mask(2, 2),
        "alternatives": [],
        "evidence_refs": ["frame-b"],
        "parent_claim_id": None,
    }
    with pytest.raises(ProtocolError, match="requires a hypothesis update"):
        grounding_claim_from_draft(draft, latest_probe_result=latest)

    draft["hypothesis_update"] = {
        "prior_probe_result_id": "probe-result-a",
        "disposition": "abandon",
        "rationale": "The prior primary stayed fixed while visible change remained elsewhere.",
    }
    draft["heuristic_updates"] = [heuristic_update(role="watch")]
    revised = grounding_claim_from_draft(draft, latest_probe_result=latest)
    assert revised.hypothesis_update is not None
    assert revised.hypothesis_update.disposition == "abandon"
    assert revised.heuristic_register[0].epistemic_status == "provisional"
    assert revised.heuristic_register[0].decision_role == "watch"


def test_host_forbids_epistemic_closure_but_not_operational_deferral() -> None:
    latest = {
        "evidence_id": "probe-result-a",
        "change_accounting": {"outside_primary_changed": True},
    }
    base = {
        "observation_id": "frame-b",
        "primary_region": mask(1, 1),
        "alternatives": [],
        "evidence_refs": ["frame-b"],
        "parent_claim_id": None,
    }
    accepted = grounding_claim_from_draft(
        {
            **base,
            "hypothesis_update": {
                "prior_probe_result_id": "probe-result-a",
                "disposition": "retain",
                "rationale": "This interpretation remains prospectively testable.",
            },
            "heuristic_updates": [heuristic_update(role="defer")],
        },
        latest_probe_result=latest,
    )
    assert accepted.heuristic_register[0].decision_role == "defer"

    closed = heuristic_update()
    closed["epistemic_status"] = "confirmed"
    with pytest.raises(ProtocolError, match="extra"):
        grounding_claim_from_draft(
            {
                **base,
                "hypothesis_update": {
                    "prior_probe_result_id": "probe-result-a",
                    "disposition": "retain",
                    "rationale": "The interpretation is being carried forward.",
                },
                "heuristic_updates": [closed],
            },
            latest_probe_result=latest,
        )


def test_host_retains_omitted_heuristics_and_unions_evidence_for_updates() -> None:
    prior = heuristic("frame-a")
    latest = {
        "evidence_id": "probe-result-a",
        "grounding_claim": {"heuristic_register": [prior]},
    }
    base = {
        "observation_id": "frame-b",
        "primary_region": mask(1, 1),
        "alternatives": [],
        "evidence_refs": ["frame-b"],
        "parent_claim_id": None,
        "hypothesis_update": {
            "prior_probe_result_id": "probe-result-a",
            "disposition": "revise",
            "rationale": "The prior scope is being revised without deleting its evidence.",
        },
    }
    retained = grounding_claim_from_draft(base, latest_probe_result=latest)
    assert retained.heuristic_register[0].to_record() == prior

    accepted = grounding_claim_from_draft(
        {**base, "heuristic_updates": [heuristic_update(role="watch")]},
        latest_probe_result=latest,
    )
    assert accepted.heuristic_register[0].decision_role == "watch"
    assert accepted.heuristic_register[0].evidence_refs == (
        "frame-a",
        "frame-b",
        "probe-result-a",
    )

    with pytest.raises(ProtocolError, match="extra"):
        grounding_claim_from_draft(
            {**base, "heuristic_register": [prior]}, latest_probe_result=latest
        )
