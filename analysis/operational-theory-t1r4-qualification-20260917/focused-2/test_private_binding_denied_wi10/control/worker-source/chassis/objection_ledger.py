"""Host-owned lifecycle for one epistemic objection at a time.

The 2026-09-12 public run showed the cost of having no owner for this state.
S5I5 received five model-generated variants of one objection; the last four cost
93 of the game's 600 wall seconds, and the Actor ended 32 seconds short of the
budget needed to settle the very question they restated. So the ledger exists to
make repetition free and disposition explicit, not to overrule the Actor.

Rules:

* One objection is pending at a time. While one is pending the critic is not
  called; the host restates the open objection locally at no model cost.
* An objection is disposed of, not obeyed. ``defend`` is always available, so a
  wrong question costs the Actor a sentence rather than a pivot.
* ``probe`` is a disposition: the Actor may answer by budgeting a bounded test
  instead of settling the belief first. The host then holds it to that budget.
* ``withdraw`` must name the actions being abandoned. An abandonment that names
  nothing retires nothing, and recurrence could not be detected mechanically.

The host never judges game facts. Every check here is arithmetic over action
strings and counters the Actor itself supplied.
"""
from __future__ import annotations

import json
from typing import Any

# A disposition the Actor may attach to a commitment.
DISPOSITIONS = ("withdraw", "narrow", "defend", "probe")
# Non-submitting Actor calls tolerated before the host stops restating and the
# next commitment must carry a disposition.
FORCE_AFTER_REMINDERS = 2
# Share of a submitted batch that must match a retired action for recurrence.
RECURRENCE_SHARE = 0.5
# Consecutive blocked commitments before the host expires the objection itself.
# A critic that could deadlock a game would cost more than the errors it catches,
# and every game in the 2026-09-12 run died of wall clock, not of bad reasoning.
EXPIRE_AFTER_BLOCKED_SUBMITS = 3


class ObjectionError(ValueError):
    """A malformed disposition. Reported to the Actor, never raised at it."""


def _text(value: Any, field: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ObjectionError(f"{field} must be nonempty text")
    text = value.strip()
    if len(text) > limit:
        raise ObjectionError(f"{field} must stay within {limit} characters")
    return text


def _actions(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > 20:
        raise ObjectionError(f"{field} must be one to twenty action strings")
    out = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ObjectionError(f"{field} must be one to twenty action strings")
        out.append(item.strip().upper())
    return out


class ObjectionLedger:
    """The pending objection, the retired claims, and their audit trail."""

    def __init__(self) -> None:
        self.pending: dict[str, Any] | None = None
        self.retired: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.reminders = 0
        self.defended_refs: dict[str, int] = {}
        self.probe: dict[str, Any] | None = None
        self.blocked_submits = 0

    # -- opening -----------------------------------------------------------

    def accepts_review(self) -> bool:
        """A new critic call is justified only with nothing already pending."""
        return self.pending is None

    def open(self, *, review_id: str, question: dict[str, str], step: int,
             wall_remaining: float | None = None) -> dict[str, Any]:
        if self.pending is not None:
            raise ObjectionError("an objection is already pending")
        self.pending = {
            "review_id": review_id, "claim_ref": question["claim_ref"],
            "question": question["question"], "reason": question["reason"],
            "opened_at_step": step, "opened_wall_remaining": wall_remaining,
            "previously_defended": self.defended_refs.get(question["claim_ref"], 0),
        }
        self.reminders = 0
        return self.pending

    # -- while pending -----------------------------------------------------

    def restate(self) -> dict[str, Any]:
        """Spend a review boundary on the open objection instead of a model call."""
        if self.pending is None:
            raise ObjectionError("nothing pending to restate")
        self.reminders += 1
        return {**self.pending, "reminders": self.reminders,
                "blocking": self.blocking()}

    def blocking(self) -> bool:
        """Whether the next commitment must carry a disposition.

        Blocking from the first commitment is deliberate. The Actor is asked to
        dispose of the objection, not to agree with it, and ``defend`` and
        ``probe`` both let the intended commitment proceed unchanged.
        """
        return self.pending is not None

    def block_submit(self) -> dict[str, Any] | None:
        """Count a commitment refused for want of a disposition.

        Returns an expiry record once the Actor has been refused
        ``EXPIRE_AFTER_BLOCKED_SUBMITS`` times, at which point the objection is
        dropped and the commitment proceeds. The critic may be wrong, and a
        stalemate spends the one budget that actually ends these games.
        """
        self.blocked_submits += 1
        if self.blocked_submits < EXPIRE_AFTER_BLOCKED_SUBMITS:
            return None
        record = {"review_id": self.pending["review_id"],
                  "claim_ref": self.pending["claim_ref"], "disposition": "expired",
                  "evidence": f"refused {self.blocked_submits} commitments without a disposition",
                  "reminders": self.reminders}
        self.history.append(record)
        self.pending = None
        self.reminders = 0
        self.blocked_submits = 0
        return record

    # -- closing -----------------------------------------------------------

    def close(self, closes: Any, *, step: int) -> dict[str, Any]:
        """Apply an Actor disposition. Returns the recorded closure."""
        if not isinstance(closes, dict):
            raise ObjectionError("closes must be an object")
        unknown = set(closes) - {"review_id", "disposition", "evidence",
                                 "retire_actions", "actions_budgeted", "narrowed_claim"}
        if unknown:
            raise ObjectionError(f"unknown closes fields: {sorted(unknown)}")
        if self.pending is None:
            raise ObjectionError("no objection is pending")
        review_id = closes.get("review_id")
        if review_id != self.pending["review_id"]:
            raise ObjectionError(
                f"closes.review_id {review_id!r} is not the pending "
                f"{self.pending['review_id']!r}")
        disposition = closes.get("disposition")
        if disposition not in DISPOSITIONS:
            raise ObjectionError(f"disposition must be one of {list(DISPOSITIONS)}")
        evidence = _text(closes.get("evidence"), "closes.evidence", 400)

        record = {"review_id": review_id, "claim_ref": self.pending["claim_ref"],
                  "disposition": disposition, "evidence": evidence,
                  "closed_at_step": step, "reminders": self.reminders}
        if disposition == "withdraw":
            record["retire_actions"] = _actions(
                closes.get("retire_actions"), "closes.retire_actions")
            self.retired.append({"claim_ref": self.pending["claim_ref"],
                                 "review_id": review_id, "evidence": evidence,
                                 "retire_actions": record["retire_actions"],
                                 "retired_at_step": step})
        elif disposition == "narrow":
            record["narrowed_claim"] = _text(
                closes.get("narrowed_claim"), "closes.narrowed_claim", 400)
        elif disposition == "probe":
            budget = closes.get("actions_budgeted")
            if type(budget) is not int or not 1 <= budget <= 200:
                raise ObjectionError(
                    "closes.actions_budgeted must be an integer from 1 to 200")
            record["actions_budgeted"] = budget
            record["probe_deadline_step"] = step + budget
            self.probe = dict(record)
        else:  # defend
            self.defended_refs[self.pending["claim_ref"]] = (
                self.defended_refs.get(self.pending["claim_ref"], 0) + 1)

        self.history.append(record)
        self.pending = None
        self.reminders = 0
        self.blocked_submits = 0
        return record

    def probe_due(self, step: int) -> dict[str, Any] | None:
        """Return a spent action budget once, so a probe cannot be an escape hatch.

        ``probe`` lets the Actor answer an objection by acting instead of
        believing. That is only honest if the host later asks what the budgeted
        actions decided, which costs nothing to check.
        """
        if self.probe is None or step < self.probe["probe_deadline_step"]:
            return None
        spent, self.probe = self.probe, None
        return spent

    # -- retention ---------------------------------------------------------

    def recurrence(self, actions: list[str]) -> dict[str, Any] | None:
        """Detect a silent return to actions the Actor itself retired.

        Mechanical only: a majority of the batch matches retired action strings.
        Deliberate re-entry stays available through a ``probe`` disposition.
        """
        if not self.retired or not actions:
            return None
        normalized = [a.strip().upper() for a in actions]
        for retirement in reversed(self.retired):
            retired = set(retirement["retire_actions"])
            hits = sum(action in retired for action in normalized)
            if hits / len(normalized) >= RECURRENCE_SHARE:
                return {**retirement, "matched_actions": hits,
                        "submitted_actions": len(normalized)}
        return None

    def reopen_for_probe(self, retirement: dict[str, Any], closes: Any, *,
                         step: int) -> dict[str, Any]:
        """Re-enter a retired claim deliberately, under a declared action budget."""
        self.pending = {
            "review_id": retirement["review_id"], "claim_ref": retirement["claim_ref"],
            "question": "Retired claim re-entered; settle it under a declared budget.",
            "reason": "The Actor retired these actions and has resumed them.",
            "opened_at_step": step, "opened_wall_remaining": None,
            "previously_defended": self.defended_refs.get(retirement["claim_ref"], 0),
        }
        self.reminders = 0
        record = self.close(closes, step=step)
        if record["disposition"] != "probe":
            raise ObjectionError(
                "re-entering a retired claim requires disposition 'probe' "
                "with actions_budgeted")
        self.retired = [r for r in self.retired
                        if r["review_id"] != retirement["review_id"]]
        return record

    # -- rendering ---------------------------------------------------------

    def view(self) -> dict[str, Any]:
        """The ledger as the critic and the audit trail see it."""
        return {
            "pending": self.pending, "reminders": self.reminders,
            "retired_claims": [
                {k: v for k, v in row.items() if k != "review_id"} for row in self.retired],
            "closed": [{k: v for k, v in row.items() if k in
                        ("claim_ref", "disposition", "closed_at_step")}
                       for row in self.history],
        }


def restatement_injection(pending: dict[str, Any]) -> dict[str, str]:
    """Restate an open objection at a review boundary without a model call.

    S5I5 spent 93 of its 600 wall seconds on four model-generated restatements of
    one objection, and ended 32 seconds short of settling it. Restating locally is
    what makes repetition free.
    """
    cost = ""
    if pending.get("opened_wall_remaining") is not None:
        cost = (f" It has been open since step {pending['opened_at_step']}, when "
                f"{pending['opened_wall_remaining']:.0f}s of wall budget remained.")
    return {"role": "user", "content": (
        f"HOST EPISTEMIC OBJECTION {pending['review_id']} about "
        f"{pending['claim_ref']} is STILL OPEN after {pending['reminders']} "
        f"reminder(s); no new review was spent.{cost}\n"
        f"{pending['question']}\n"
        "Dispose of it on your next submit_actions with closes={...}: defend, "
        "narrow, withdraw, or probe. Continuing to inspect neither answers it nor "
        "buys back the wall clock."
    )}


def submit_guidance(pending: dict[str, Any]) -> str:
    """The rejection text that turns an open objection into the next commitment."""
    return (
        f"REJECTED: epistemic objection {pending['review_id']} about "
        f"{pending['claim_ref']} is open and this commitment does not dispose of it.\n"
        f"Question: {pending['question']}\n"
        "Pass closes={...} to submit_actions with exactly one disposition:\n"
        '  {"review_id":"%s","disposition":"defend","evidence":"the observation that warrants the claim"}\n'
        '  {"review_id":"%s","disposition":"narrow","evidence":"why","narrowed_claim":"the smaller claim you still hold"}\n'
        '  {"review_id":"%s","disposition":"withdraw","evidence":"why","retire_actions":["ACTION you will stop repeating"]}\n'
        '  {"review_id":"%s","disposition":"probe","evidence":"what the test decides","actions_budgeted":N}\n'
        "Defending and probing both leave this commitment unchanged. Choose the "
        "one that is true; do not agree to get past this check."
        % ((pending["review_id"],) * 4)
    )


def recurrence_guidance(retirement: dict[str, Any]) -> str:
    return (
        "REJECTED: this batch resumes actions you retired at step "
        f"{retirement['retired_at_step']} when withdrawing {retirement['claim_ref']} "
        f"({retirement['evidence']}).\n"
        f"Retired actions: {retirement['retire_actions']}.\n"
        "Resuming them deliberately is allowed, under a declared budget. Pass "
        'closes={"review_id":"%s","disposition":"probe","evidence":"what filling '
        'this out decides","actions_budgeted":N} and the host will hold you to N '
        "actions. Otherwise submit a different commitment."
        % retirement["review_id"]
    )


def probe_guidance(spent: dict[str, Any]) -> str:
    """The host's free follow-up when a budgeted probe is exhausted."""
    return (
        f"HOST PROBE ACCOUNTING: you closed objection {spent['review_id']} about "
        f"{spent['claim_ref']} by budgeting {spent['actions_budgeted']} actions to "
        f"settle it ({spent['evidence']}). That budget is now spent.\n"
        "State in one line what those actions decided: is the claim settled, "
        "narrowed, or withdrawn? If it is still open, say what would settle it "
        "within the wall clock remaining, rather than extending the same test."
    )
