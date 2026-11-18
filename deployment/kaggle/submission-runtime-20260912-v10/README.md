# Objection-closure runtime v10

Successor to v9 after the 2026-09-12 objection-closure public run
(`analysis/deployment-probes/20260912-objection-closure-public-v1`), which ended
in ERROR on LS20 with two games unplayed. The Actor declared the game "stuck"
with 152s and 193 actions remaining and returned DONE; the worker fell silent and
the host's decision timeout killed the launcher.

Two changes, both removals.

**The Actor no longer owns "game over".** Its instructions no longer tell it to
call `return_result(RespondReason.DONE)` on a "stuck" game; every turn ends with
`submit_actions`. If it calls DONE anyway on a NOT_FINISHED game, `headless.py`
records the refusal in `done_refused.jsonl` and redelivers the same state with a
host note. No new game-end mechanism is added: if the model budget really is
spent, the existing admission gate ends the game through `budget_exhausted.json`.

**The critic packet no longer carries a projected action count.** In the public
run the critic restated `affordable_actions_at_this_rate: 2` as "only two
wall-clock actions remaining" and the Actor adopted it as a rule. The packet
keeps wall seconds remaining and seconds per executed action — facts — and
states that the host ends games.

Everything else is v9: compact commitment packet (three code items, five
commitment rationales, measured transitions as counts and regions, the objection
ledger), one pending objection at a time, dispositions on commitments,
recurrence and probe accounting, expiry after three refused commitments.

On the one game v9 ran, the contract behaved as specified and the critic cost
1.3% of session wall (v8: 26.0%) at 1.7-2.2k prompt tokens per call.

This candidate has local deterministic and loopback qualification only.
`test_done_refusal.py` reproduces the public-run failure on v9 (`worker timed
out waiting for decision`) and passes on v10.
