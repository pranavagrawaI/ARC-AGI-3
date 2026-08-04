# Iteration 0005 — learning_loop treatment

## Question

Iteration 0004 corrected a biased action interface and reopened action choice: the
old control spent 44/44 actions on action 6, the corrected control spread them
across 1/2/3/4. Both completed zero levels.

So choice reopened and progress did not follow. This iteration asks the next
question and only that one:

> With action choice already reopened, does a model-owned learning loop convert
> choice into level progress?

## The single variable

The treatment arm is `treatment_live_spec()` — the recorded corrected control
derived from the same `fixed_args()` object that ran it, plus the `learning_loop`
capability and a new `arm_id`. Nothing else is restated, because restating budgets
is how a matched pair stops matching without anyone noticing.

The authorized differences are `arm_id`, `treatment_capabilities`, and
`system_prompt` — the capability's addendum is appended by the shared factory and
hashed separately from the base prompt. Every other `ArmSpec` field is checked
equal at plan time by `_shared_field_diff`, and preparation refuses if any differ.

## Deliberate omissions

- no control rerun; the 0004 control run is the baseline, pinned by digest;
- no coaching, shadow teacher, or human feedback;
- no relaxed repair budget or raised action ceiling to obtain a completed run;
- no re-pin of `0002-R`; preparation and execution both refuse if it moved;
- no automatic successor run.

## The repair budget is the control's, on purpose

`max_decision_repairs` stays at 2. The `learning_loop` contract asks for a
six-field commitment on every action — `working_state`, `test`, `prediction`,
`objective`, `history_steps`, `reconciliation`. A model that cannot produce it will
exhaust the budget and the run will stop early.

That outcome is a result about the capability, not a defect to tune away. If the
run stops on `decision_repairs`, the finding is that this contract is beyond this
model at this size, and the next iteration is a simpler contract — not a larger
budget. `learning-loop-summary.json` reports `decision_repairs` and
`stop_reason` for exactly this reading.

## What the audit records

`_learning_loop_measures` counts and echoes; it does not score. Mechanism outcomes,
objective outcomes, objectives authored (with the model's own claim strings),
reconciliations written, history steps requested, and how often working state
changed. A reader deciding whether the loop worked is reading the model's
commitments and their mechanical results, never a host verdict. A test asserts no
measure name reads as one.

## Offline verification

```bash
uv run --with pytest pytest -q experiments/agents/tests/test_p15_learning_loop_live.py
uv run python3 -m experiments.agents.0006-capability-environment.p9.freeze
```

The first checks the pair contract, the plan pins, the frozen-control guard, and
the audit. The second must report empty drift before either command below runs.

## Authorized live generation

Preparation makes one unauthenticated metadata request and writes a plan. It
spends nothing. Execution is the only paid step and refuses any confirmation
string but the exact one.

```bash
uv run python3 experiments/agents/0006-capability-environment/learning_loop_live.py prepare
uv run python3 experiments/agents/0006-capability-environment/learning_loop_live.py \
  execute --confirm 'EXECUTE LEARNING LOOP TREATMENT CD82 V1'
```

Every run parameter lives in `fixed_args()`, shared with the control's runner.
There are no run-shaping command-line arguments to omit or mistype.

Expected cost, from the control's recorded $0.20080473 at 52 calls: the same order,
higher if the commitment contract lengthens completions or draws repairs.

## Live result

Run 2026-08-02. Completed on the action budget; the repair budget was not exhausted.

| Measure | Uniform-action control | learning_loop treatment |
|---|---:|---:|
| Actions | 44 | 44 |
| Action 1 / 2 / 3 / 4 | 9 / 13 / 13 / 9 | 14 / 10 / 11 / 8 |
| Action 6 | 0 | 1 |
| Levels completed | 0 | 0 |
| Model calls | 52 | 79 |
| Decision repairs | 1 | 7 |
| Reported cost | $0.20080473 | $0.37675850 |

Learning-loop measures: 44 commitments; mechanism predictions 35 match / 9 mismatch;
6 objectives authored, 5 falsified and 39 pending; 11 reconciliations; working state
changed on 31 of 44 actions; **0 history steps requested**.

## Reading

The capability answered its question in the negative — no level progress — but the
run is not a null result about the loop. Three findings, in the order they matter.

**The objective mechanism cannot express what the model is trying to do.**
`_validate_objective` constrains every objective's progress check to
`levels_completed > baseline`. The model's actual sub-goal lives in `claim`, which
is free text and never checked. So the model committed to "Move the purple diamond
onto the white square" while the host tested "did a level complete within 8
actions". All five falsifications fired at horizon expiry with `remaining_steps: 0`,
and the model read them correctly: *"falsified due to the step limit"* (a17),
*"falsified due to horizon expiration"* (a41). It then re-authored a byte-identical
claim three times, which is the rational response to a falsification that carries no
information about the claim.

**Objectives therefore ran on a metronome, not on evidence.** Authored at actions 1,
9, 17, 25, 33, 41 — every eight, always at the maximum horizon. A new objective is
permitted only after falsification, and falsification only happens at expiry, so the
cadence is a property of the design rather than of the game.

**The loop worked everywhere it was allowed to.** Immediate mechanism prediction ran
at 35/44, and the reconciliations are specific and evidence-grounded: *"blocked by a
wall when moving UP, I will now move it RIGHT to search for a gap"* (a22),
*"the top purple diamond UP failed (0 changed pixels), so I am switching to the
middle purple diamond"* (a33). Local mechanics were being learned and acted on. What
never closed was the link from local mechanics to the level.

**`history_steps` is dead weight.** Zero requests in 44 actions. The affordance
carries a per-turn prompt cost and returned nothing all run.

## Next

The single change this argues for is letting an objective's progress check use the
same observables mechanism predictions already use — `cell`, `changed_pixels`,
`state` — so a sub-goal becomes falsifiable on its own terms and a falsification
carries information. `levels_completed` stays available, not mandatory. Whether that
is one iteration or a redesign of the objective contract is a judgment call, not a
conclusion this run forces.
