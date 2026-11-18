"""Behaviour tests for the objection lifecycle.

Each test names a failure the 2026-09-12 public run demonstrated. Where a number
appears it comes from
``analysis/deployment-probes/20260912-epistemic-questioner-public-v1`` and is
reproduced by ``analysis/critic-closure-contract-20260912/evidence.py``.
"""
import importlib.util
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


def load():
    spec = importlib.util.spec_from_file_location(
        "objection_ledger_test", HERE / "chassis/objection_ledger.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bar_objection(ledger, *, review_id="critic:3", step=6, wall=300.0):
    """S5I5's real objection: is filling the bottom bar the level's objective?"""
    return ledger.open(review_id=review_id, step=step, wall_remaining=wall, question={
        "claim_ref": "actor-message-119-tool-0",
        "question": "If clicking any cell adds one bar cell, how does the grid become "
                    "the level-completion state rather than merely reaching 64/64?",
        "reason": "Answering forces a pivot from raw counter-filling to the win condition.",
    })


def test_an_open_objection_costs_no_second_model_review():
    """S5I5 spent 93 of 600 wall seconds restating one objection four more times."""
    ledger = load().ObjectionLedger()
    assert ledger.accepts_review()
    bar_objection(ledger)
    for expected in (1, 2, 3, 4):
        assert not ledger.accepts_review()
        assert ledger.restate()["reminders"] == expected
    ledger.close({"review_id": "critic:3", "disposition": "defend",
                  "evidence": "every observed click added exactly one bar cell"}, step=9)
    assert ledger.accepts_review()


def test_an_objection_survives_actor_calls_that_are_not_commitments():
    """Critic 3 was followed by inspection, then a submit that never answered it."""
    module = load()
    ledger = module.ObjectionLedger()
    bar_objection(ledger)
    for _ in range(5):          # last_transition, notes, grid reads: no commitment
        assert ledger.blocking()
    assert ledger.pending["review_id"] == "critic:3"
    assert "defend" in module.submit_guidance(ledger.pending)


def test_agreement_is_not_a_disposition():
    """The Actor agreed at turn 11 and resumed the challenged batch at turn 12."""
    module = load()
    ledger = module.ObjectionLedger()
    bar_objection(ledger)
    for bad in (
        {"review_id": "critic:3", "disposition": "agree", "evidence": "you are right"},
        {"review_id": "critic:3", "disposition": "acknowledged", "evidence": "noted"},
        {"review_id": "critic:3", "disposition": "withdraw", "evidence": ""},
        {"review_id": "critic:9", "disposition": "defend", "evidence": "wrong objection"},
        {"disposition": "defend", "evidence": "no review id"},
    ):
        with pytest.raises(module.ObjectionError):
            ledger.close(bad, step=13)
    assert ledger.blocking(), "a rejected disposition must leave the objection open"


def test_withdrawing_must_name_the_actions_being_abandoned():
    """An abandonment that retires nothing cannot be checked for recurrence."""
    module = load()
    ledger = module.ObjectionLedger()
    bar_objection(ledger)
    with pytest.raises(module.ObjectionError, match="retire_actions"):
        ledger.close({"review_id": "critic:3", "disposition": "withdraw",
                      "evidence": "the bar is a counter, not the goal"}, step=13)
    record = ledger.close({"review_id": "critic:3", "disposition": "withdraw",
                           "evidence": "the bar is a counter, not the goal",
                           "retire_actions": ["CLICK 32 32"]}, step=13)
    assert record["retire_actions"] == ["CLICK 32 32"]


def test_a_withdrawn_proxy_cannot_silently_return():
    """Turn 12 resumed a ten-click bar batch one turn after withdrawing the claim."""
    module = load()
    ledger = module.ObjectionLedger()
    bar_objection(ledger)
    ledger.close({"review_id": "critic:3", "disposition": "withdraw",
                  "evidence": "the bar is a counter, not the goal",
                  "retire_actions": ["CLICK 32 32"]}, step=13)
    assert ledger.recurrence(["CLICK 31 10"]) is None, "an unrelated probe is free"
    recurred = ledger.recurrence(["CLICK 32 32"] * 10)
    assert recurred is not None and recurred["matched_actions"] == 10
    assert "actions_budgeted" in module.recurrence_guidance(recurred)


def test_deliberate_re_entry_needs_a_declared_action_budget():
    """The turn-12 return was a real test. It should have to name its price."""
    module = load()
    ledger = module.ObjectionLedger()
    bar_objection(ledger)
    ledger.close({"review_id": "critic:3", "disposition": "withdraw",
                  "evidence": "the bar is a counter", "retire_actions": ["CLICK 32 32"]},
                 step=13)
    recurred = ledger.recurrence(["CLICK 32 32"] * 10)
    with pytest.raises(module.ObjectionError, match="probe"):
        ledger.reopen_for_probe(recurred, {
            "review_id": "critic:3", "disposition": "defend",
            "evidence": "I changed my mind"}, step=14)
    record = ledger.reopen_for_probe(recurred, {
        "review_id": "critic:3", "disposition": "probe",
        "evidence": "fill the bar to 64 and see whether the level completes",
        "actions_budgeted": 46}, step=14)
    assert record["probe_deadline_step"] == 60
    assert ledger.recurrence(["CLICK 32 32"] * 10) is None, "re-entry retires the retirement"


def test_a_budgeted_probe_is_accounted_for_once_it_is_spent():
    """A probe answers by acting; it is honest only if the budget is later audited."""
    module = load()
    ledger = module.ObjectionLedger()
    bar_objection(ledger)
    ledger.close({"review_id": "critic:3", "disposition": "probe",
                  "evidence": "fill the bar to 64 and see whether the level completes",
                  "actions_budgeted": 46}, step=14)
    assert ledger.accepts_review(), "probing closes the objection; play continues"
    assert ledger.probe_due(59) is None
    spent = ledger.probe_due(60)
    assert spent["actions_budgeted"] == 46
    assert "what those actions decided" in module.probe_guidance(spent)
    assert ledger.probe_due(80) is None, "accounting is asked for once"


def test_a_blocked_commitment_cannot_deadlock_the_game():
    """The critic is fallible, and wall clock ended every game in the run."""
    module = load()
    ledger = module.ObjectionLedger()
    bar_objection(ledger)
    assert [ledger.block_submit() for _ in range(2)] == [None, None]
    expired = ledger.block_submit()
    assert expired["disposition"] == "expired"
    assert not ledger.blocking(), "the commitment proceeds rather than stalling"
    assert ledger.accepts_review()


def test_the_ledger_reports_what_was_retired_and_what_was_answered():
    module = load()
    ledger = module.ObjectionLedger()
    bar_objection(ledger)
    ledger.close({"review_id": "critic:3", "disposition": "withdraw",
                  "evidence": "the bar is a counter", "retire_actions": ["CLICK 32 32"]},
                 step=13)
    bar_objection(ledger, review_id="critic:8", step=20)
    view = ledger.view()
    assert view["pending"]["review_id"] == "critic:8"
    assert view["retired_claims"][0]["retire_actions"] == ["CLICK 32 32"]
    assert view["closed"] == [{"claim_ref": "actor-message-119-tool-0",
                               "disposition": "withdraw", "closed_at_step": 13}]


def test_a_defended_claim_is_counted_so_a_later_review_can_see_it():
    """Repeating an objection the Actor already answered is what cost the run."""
    module = load()
    ledger = module.ObjectionLedger()
    bar_objection(ledger)
    ledger.close({"review_id": "critic:3", "disposition": "defend",
                  "evidence": "nine observed clicks each advanced the bar"}, step=9)
    reopened = bar_objection(ledger, review_id="critic:5", step=11)
    assert reopened["previously_defended"] == 1
