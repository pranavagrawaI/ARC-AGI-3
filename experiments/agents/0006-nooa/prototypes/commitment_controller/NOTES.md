# Prototype verdict

**Question.** Does a fixed eight-field commitment-preserving state block admit
competent LS20 play while refusing the drift in the recorded Gemma rounds?

**Verdict: yes, after one rule was demoted.** Offline, no spend.

- Luna's winning 28-action trajectory: **28/28 turns admitted**, 1 identity
  reversal (the licensed one, at the shape changer), route1 survived 16/27
  turns, 3 blocked moves repaired locally, 1 outcome misclassified.
- Six drift cases lifted from the Gemma transcript: **6/6 rejected by their
  named rule**.

## What the prototype changed about the design

1. **R6 "route1 survives" cannot be a hard rule.** Three of Luna's 28 winning
   turns replace `ROUTE1` outright after a move revealed the corridor's real
   shape. That is competent re-planning. Enforcing R6 would force the model
   either to declare a contradiction it does not believe or to walk a route it
   knows is wrong. Route replacement is now counted and reported, never
   rejected. What must survive an action is the binding and the objective; the
   action list is the cheap part.

2. **R1 subsumes the identity half of R2 and R3.** An unlicensed binding change
   is caught whatever the outcome was. R2 and R3 earn their place only on the
   objective. Stated separately anyway, because a rejection naming the outcome
   that forbade the change is more legible to the model than a bare R1.

2b. **R2 and R3 had to yield to a named contradiction.** The first draft granted
   the contradiction exemption in R1 but not in R2 or R3, so a contradiction
   *discovered by an ordinary move* — "I moved and the thing I called the player
   stayed put", which is the most natural way to catch a mis-binding in this
   game — could never be acted on. The model's only escape would have been to
   stop classifying its own transitions, which is the behaviour the controller
   exists to prevent. Found by a test, not by the replay: Luna never mis-bound,
   so no recorded trajectory exercises it.

3. **`moved` / `blocked` / `special_event` is mechanically separable, and the
   host must not do it.** In this trajectory the pixel-diff cell count is 52 for
   a move, 2 for a block, 58 for the changer — cleanly separated. The host could
   therefore hand the model a correct label for free. It must not: that mapping
   is the semantic step the probe exists to measure. The host shows the neutral
   `changed_cells` count the existing evidence operator already produces, the
   model names the outcome, and `classify()` is used only to score it.

## The finding that matters most for the live design

The Gemma failure is **not at the commitment gate**. In
`known-world-rules-treatment-open-l1-44-v1` Gemma burned all 8 deliberation
rounds on `execute_python`, finished every round with `tool_calls`, and never
called `return_result` — zero actions, zero commitments. A controller that
enforces only at commitment time would never have fired.

So the intervention has to reach *inside* the rounds of a single action point.
The frozen 0002 worker already has that channel and it is currently unused:
`NOTES.md` is echoed in full at the top of every model round
(`nooa-runtime/agent.py:508-544`), and the workspace doc tells the model it
survives dropped turns. In all three recorded runs `NOTES.md` was left at its
364-byte seed — neither model ever wrote to it.

That splits enforcement in two, and the split should be stated plainly in any
write-up:

- **Round level, instructed not enforced.** The seeded state block in
  `NOTES.md`. This is where Gemma actually drowns, and the frozen substrate
  gives no hook to reject a round.
- **Action level, host-enforced.** The same eight fields validated as
  `decision.working_state` in `CompetenceController.admit`, which already
  carries a 4000-char working state into the next packet.

A negative live result therefore has two readings that this design cannot
separate: the rules did not help, or the round-level half was never obeyed.
So the runner records `notes_written` on every admitted action; without it a
null is uninterpretable.

## As wired

`--knowledge-condition rules_controller`, treatment arm only. Same `RULES.md`,
same budgets, same harness as plain `rules`; the differences are exactly:

- the seeded `NOTES.md` is the protocol rather than a one-paragraph blurb, so
  it reaches the model every round instead of once;
- `working_state` is published to the model as the eight fields plus the rules
  plus its own last accepted block, instead of a free-form object;
- the host checks each block as a continuation before `competence.admit`, so a
  refusal costs no action, emits no competence event — the recorded stream still
  replays exactly — and returns the rule name through the existing repair round.

The seed is capped at the frozen worker's 4000-character `NOTES.md` echo budget
and fails loudly over it, because silent truncation would mean measuring a
protocol the model never saw whole. It currently sits at 3644.
