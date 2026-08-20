# tycho-nooa-symmetry-uptake-ar25-20260820T101248Z — RESULT

Status: VALID. No INVALID clause fired. Stopped one action short of its cap by a
provider credit limit, after the outcome of interest had already been recorded.

Result: 19 of 20 actions, **1 level completed**, stop_reason `exception`
(OpenRouter daily key limit at level 1 turn 2), $4.327, 377 LLM calls.
Artifacts: `runs/tycho-nooa-symmetry-uptake-ar25-20260820T101248Z/` (untracked).

**This is the first level this solver has ever completed.** The contrast run scored
0 levels; the 60-action positive control scored 0 levels; every run before this one
scored 0 levels.

## Primary measure

    executed_symmetry_calls = 2   (reached = 4, ERR = 2)

    actor    run_python= 40  reached=1  executed=1  [ok ] level 0 turn 3
    builder  run_python=230  reached=3  executed=1  [ERR] level 1 turn 0
                                                    [ok ] level 1 turn 0
                                                    [ERR] level 1 turn 1

Not the prior's 0. The card said "the honest expectation here is 0"; it was wrong.

### One correction to the probe, made after the data and stated openly

`tools/symmetry_uptake.py` first scored this run 4/4 executed. Its `BROKEN` list
named four exception classes and did not include `ValueError`, which is what two of
the builder's calls raised. The card's prose defines ERR as "a reach that raises";
naming individual classes was an incomplete implementation of that sentence, so the
list was first replaced with a match on `Traceback (most recent call last)`. Review
then found the general contract already present in each trace row: `status`. The
final probe requires `status == "complete"` and also rejects traceback text, so a
`SandboxError` or any other non-traceback error cannot inflate the count.

The change makes the measure **stricter**, and it moved the headline number down
(4 → 2). The negative control was re-run under the corrected probe and still scores
0, so the fix did not buy the result. Recorded here because a measure edited after
seeing the data is worth less than one edited before, whichever way it moves.

## Result map: the middle branch, with the secondary that the map did not anticipate

    executed > 0, no goal predicate -> reached for, but not at hypothesis time;
                                       the gap is the loop, not the library

Half right. `outcome()` is still `return "ongoing"` — the constant, unchanged, even
after a real observed `level_complete`. No goal predicate appeared. But the map's
gloss is wrong: the library *was* reached for at hypothesis time, and it was the
actor's hypothesis-time reach that produced the level.

The map assumed a goal predicate in `world_model.py` was the only route from
hypothesis to test. It is not. The actor tested its own hypothesis in a scratch
`run_python` call and acted on the answer without the supervisor ever knowing.

## The causal chain, in the actor's own words

The plan was written **before** the actions and the environment then ratified it.
That is the strongest evidence a single run can carry.

| turn | event |
|---|---|
| pre-action | reads `symmetry.py`'s docstring unprompted: "Let's inspect symmetry.py docstring" |
| 0–2 | UP, DOWN, LEFT — establishes movement is 3 cells, mirrored across the barrier |
| 3 | `symmetry.congruent(fig4_shape, fig11_shape, translate=True)` → **True**, sizes 45 == 45 |
| 3 | "Great! 4 shape is congruent to b shape… Need to move down 30 rows and right 12 columns. 30/3 = 10 steps down. 12/3 = 4 steps. **So total moves: 10 DOWN + 4 LEFT = 14 moves.**" |
| 3–12 | DOWN × 10 |
| 13–16 | LEFT × 4 |
| 16 | **level 0 complete** |

Ten DOWNs and four LEFTs, exactly as costed, immediately after the congruence test.
`notes/actor_beliefs.md`: "Level 0 completed with LEFT action aligning 4 shape over
11 shape."

The comparison run wandered 19 distinct states, never repeating one, and wrote
"I'm stuck" at turn 15. Same game, same seed, same model, same cap, same code path.
One change.

Note what the test actually decided. The actor set both figures' labels to `1`
before comparing — it chose `colours` off, deliberately, because the shapes are
different colours and it wanted shape alone. Choosing the quotient is the modelling
act the brief said would be its job, and it did it correctly on the first attempt.

## Correction: the builder received the evidence but did not prioritize it

The builder owns `outcome()`. It fired at turns 1, 2, 3 (`dynamics`), went silent
turns 4–14 exactly as in the contrast run, and fired again at level 1 on
`level_start`. It ran **after** the completion and still wrote, in
`notes/world_model.md`:

> - Objective unknown. Hypotheses: shape matching via symmetry…
> - Win condition not observed; outcome() currently returns ongoing.

"Win condition not observed" is false on this trace, but the first causal account
above was also false: the builder prompt explicitly tells it to read
`notes/actor_beliefs.md`, and the live builder did so. At level start, call 209 read
that note before the scribe had added completion; call 227 read
`level_0/solved.json`. On the next builder pass, call 251 read actor beliefs that did
contain "Level 0 completed." The verifier independently reported
`post-winning-action missed terminal ... L0 turn 16`.

The channel therefore existed. The failure was priority and execution: later
verifier guidance set `REPAIR_FOCUS: current_level_dynamics`, the builder spent its
pass investigating level 1, and it made no `outcome()` edit. The repair must make a
host-confirmed completion a first-class checkpoint that cannot be displaced by
current-level dynamics, not add a redundant actor-to-builder read lane.

Consequence: `subgoals()` is still `[lambda s: outcome(s) == "level_complete"]`,
unsatisfiable, so `plan_subgoals` still expands to `max_nodes` and returns nothing.
The level was won by the actor's own arithmetic, with the planner inert.

## The library's parameterised API is not usable; its presets are

A clean 2×2 with no exceptions:

| call form | attempts | ok |
|---|---|---|
| presets — `congruent`, `directly_congruent` | 2 | **2** |
| raw — `canon(point='full')`, `canon(point='free')` | 2 | **0** |

Both failures are the same bug, and it is ours:

    ValueError: not enough values to unpack (expected 2, got 1)
      symmetry.py:80 in act -> (a, b), (c, d) = g

`point` is a *set of group elements* (`FULL`, `ROTATIONS`, …). `colours` in the same
signature is a *string* (`"fixed"`/`"any"`). The model passed `point='full'`, the
string iterated character by character, and `act` tried to unpack `'f'` as a matrix.
Two different models of the same parameter list, and the generalisation the model
made was the reasonable one.

`symmetry_notice.PROMPT_BRIEF` made this worse: it writes `point=...` and describes
what `point` means without ever naming the constants, while spelling `colours`
values out literally. The brief taught the concept and hid the vocabulary.

Two fixes, both cheap, and they are the next consequential action:
1. `canon` should accept the group by name (`point="full"`, `"rotations"`, `"none"`)
   as well as by object — the model's guess should have been right.
2. `PROMPT_BRIEF` should name the admissible values for every parameter it lists.

Neither is speculative: each is the direct repair of an observed failure.

## Contrast table

| | 162040Z (no symmetry) | this run |
|---|---|---|
| levels | 0 | **1** |
| actions | 20 (cap) | 19 (credit limit) |
| actor / builder calls | 71 / 165 | 86 / 291 |
| cost | $1.887 | $4.327 |
| cache hit | 79% | 85% |
| goal hypothesis | prose, untested to the last action | tested at turn 3, acted on, confirmed |
| trajectory | 19 distinct states, "I'm stuck" | 14-move straight-line transport, costed in advance |
| `outcome()` | `return "ongoing"` | `return "ongoing"` |

## What this does and does not license

Licensed: on ar25 seed 0, giving the actor a comparison vocabulary converted a
standing goal hypothesis into a test, and the test into a solved level. The
mechanism is legible end to end — docstring read, quotient chosen, relation tested,
transport costed, transport executed, level completed — and every step is in the
trace, not inferred.

Not licensed: **n = 1, on one game, at one seed.** The prior finding that this
project keeps re-learning is that single runs do not replicate (the P1 wording
effect, five null runs). One level is one level. This must be repeated — same game
for reproducibility, then a different game for generality — before it is treated as
anything more than a hypothesis with one confirmation.

Also not licensed: any claim about ar25's remaining levels. Level 1 changed the
mechanics (DOWN produced no cell change; the actor's live guess was that CLICK now
selects a piece first) and the run died two actions in, on money, not on merit.

## Next consequential action

**The API defects are fixed** (2026-08-20, after this card was first written):

- `point` now accepts `'full'`/`'rotations'`/`'halfturn'`/`'none'` as well as the
  constants, at every entry that takes a group — `canon`, `stabiliser`,
  `transporter`, `orbit`, `proper`. An unknown name raises a message naming all of
  them and stating that position freedom is `translate`, not a point group, which is
  the run's other mistake (`point='free'`).
- `colours` now rejects anything but `'fixed'`/`'any'` instead of treating a typo as
  `'any'` — the same defect, silent, and therefore worse.
- `PROMPT_BRIEF`, the `run_python` tool text, and the module docstring now spell out
  every admissible value rather than printing `point=...`.
- One hazard the run also exhibited and that no coercion can catch is now named in
  both: comparing canonical forms computed under *different* quotients (the live call
  compared `colours='fixed'` against `colours='any'`) is silently meaningless.

Regression tests replay the exact live failures against `symmetry_template.py`, the
bytes that ship, and cover empty-input validation plus non-factorial colour
canonicalization. The complete verification counts are recorded in the fix report.

Remaining: re-run this exact configuration, then raise the action cap — 19 actions is
no longer enough, because for the first time the agent is not the limiting factor.

Code: branch `refactor/tycho-nooa-core`, uncommitted symmetry work on top of 738c94f.
