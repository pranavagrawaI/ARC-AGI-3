# tycho-nooa-goal-gate-ar25-20260819T121859Z — RESULT

VALID. 20 actions, stop_reason action_budget, levels_completed 0. No crash, no
trigger_error, no PROBE_ERROR. Both invalidators of 120904Z cleared.

## Verdict: uptake null

Result map branch: **gate fires + no predicate -> the affordance is not the
bottleneck.**

    PLANNER_STATUS values over 7 builder passes
      6  not_run_goal_unsatisfiable            <- gate 0, new
      6  not_run_current_level_dynamics_failed <- upstream planner_ready, pre-existing
      0  no_plan_within_budget
      0  planner_timeout
    PLANNER_CONTRACT audits: 6      PROBE_ERROR: 0
    planner-trace.jsonl: absent — plan.py was never invoked, by the gate or the actor.

`outcome()` at the end of the run, after 188 builder tool calls:

    def outcome(state: State) -> str:
        return "ongoing"

Zero wasted searches, which is what the gate was built to buy, and zero
predicates, which is what it was built to provoke. The first half held; the
second did not.

Contrast with 175557Z (same game, same seed, no gate): 6 searches, 0
predicates, 0 levels. The gate converted 6 searches into 6 diagnoses at no cost
in levels. Both runs end with the same `outcome()`.

## What the run established

**1. The gate fires where it should, and only there.** Six fires, all after
dynamics verification passed — i.e. at exactly the moment upstream's own
`planner_ready` would have released a search. The other six blocks were
upstream's, on models that failed verification. The two checks do not overlap;
gate 0 is not a reimplementation of a check that already existed.

**2. The disjunction fix is load-bearing on builder-authored models, not just
on the pinned template.** The builder rewrote `world_model.py` wholesale at
pass 5 and its final model contains exactly one occurrence of the goal literal:

    205:    return [lambda s: outcome(s) == "level_complete"]

in `subgoals()`, with `outcome()` an unconditional `"ongoing"`. The
module-wide-literal-only implementation — the gate's first version — would have
cleared all six of this run's blocks. Confirmed in vivo on a hand-written
model, not only on `seed_world_model.py.tmpl`.

**3. The uptake failure has a stated reason, and it is one the gate already
answers.** The builder's own notes, written in the same pass as a gate block:

    - No level_complete or GAME_OVER observed yet
    - Objective unknown: possibly align zeros, clear rectangles, reach target...
    - Need to probe with further moves; outcome() remains ongoing

The gate's second line addresses precisely this: *"It does not need terminal
evidence to be worth planning against: keep it explicitly a candidate, and let
the verifier falsify it when a level actually ends."* The builder is not
ignoring the message; it holds an epistemic policy — no predicate without
terminal evidence — that the message fails to move. That is a different and
more tractable problem than indifference, and it points at the seed
`outcome()` docstring, which instructs the model to return literals
"corresponding to" observed ARC evidence and so licenses the policy directly.

**4. The operator audit found a state-abstraction defect no reading of the
model would have surfaced.** The builder added a `move_count` field to
reproduce a UI column. Against the real `wmlib` planner key:

    UP         forward             no key repeat within 8 repeated applications
    DOWN       retrograde_partial  returns 2 of 3 varying state fields to a state 1 step
                                   back; move_count does not follow, so the key never closes
    SPACEBAR   forward             no key repeat within 8 repeated applications
    CLICK      no_op               successor key equals its own key from every sampled state
    UNDO       no_op               successor key equals its own key from every sampled state

    PLANNER_STATE_GRAPH: all 7 declared operators: edges=400 distinct_keys=110 revisits=291

UP-then-DOWN restores both pieces to their original squares and yields a
*different* key, because the counter went 0->1->2 instead of back to 0. A
monotone convenience field turns the state graph into a near-tree. SPACEBAR,
which moves nothing, is classified `forward` for the same reason. `planner_key`
is absent from the model, so the default key sees every field.

The builder knows CLICK and UNDO do nothing — it wrote `# CLICK, UNDO, RESET
are no-ops` inside `transition()` — and declared them in `actions()` four lines
later. `actions()` is being authored as "what the game accepts", not "the
alphabet a search should expand". The two readings are visible in one file.

## Secondary measure: the forced action demand

    turn   1: 26 calls  <- FORCED DEMAND (25 spent + 1)
    turns 2-20: 2-9 calls each, all volunteered

One demand, at exactly the turn that killed 120904Z with a ProtocolError at
action 0. The card's suspicion that 25 steps is systematically too few does not
hold: turn 1 is a cold-start outlier, and once the workspace is read the actor
commits in 2-9 calls. One extra generate bought the remaining 19 actions.

## Loose end

LEVEL_START (712f892) never fired and could not have: `trigger_builder.py:99`
requires `turn.level > 0`, mirroring Tycho's own branch, and the run never left
level 0. Turn 0 falls through to `return None` and writes no trace row, which
is why `turn_in_level` starts at 1. Not a defect; still untested in vivo.

## Next — SUPERSEDED

This section originally proposed rewriting the seed `outcome()` docstring.
Retracted, on the run's own evidence: the docstring is another instruction
inside the same control topology, and this run is precisely the demonstration
that instructions at this location do not move the policy. The gate said the
right thing six times, in the builder's own words, and changed nothing. A
seventh wording has no reason to do better.

Two further corrections to the reading above:

- I described the verifier as punishing candidate predicates, making
  `return "ongoing"` the unique maximizer. Wrong. Replaying `verify.py`: a
  candidate that stays false on every observed decision frame yields
  `lc_false == 0`, so `outcome_status` lands on `unobserved` — the same cell as
  the seed constant. The landscape is **gradient-free**, not hostile. The seed
  only *weakly* dominates: same upside, and only the candidate has a path to
  `fail` by misfiring or raising. That is a much smaller force than claimed, and
  probably not the operative one.
- The gate's closing line, "let the verifier falsify it", was therefore empty
  regardless. The verifier cannot falsify a predicate that never fires; it
  reports `unobserved` indefinitely. That is the durable defect in the gate's
  design, and it is independent of which reading of the reward is right.

What the null actually indicates: the objective is the one thing in this
architecture with no evidence channel. Dynamics get dense per-transition error
feedback and a builder fired on divergence; purpose gets nothing that can be
wrong. The real next step is therefore not another message but a test of whether
the modelling apparatus is displacing task reasoning at all — the
`no_world_model` arm, precommitted in `tycho-nooa-no-wm-ar25-20260819T173935Z.md`.

The gate itself stays. It is free, static, correct, and it converted six
uninterpretable `planner_timeout` reports into one accurate sentence. It is a
diagnostic that now works; it was never going to be a lever.
