# Kernel version 3: the contrast ran cleanly and is uninformative about the controller

`pranavagrawai/contingent-controller-20260909` version 3, contract
`2c1e13ef4ce25e59121bcd6c637a869a8cb09c49fc58712fc9212234b8c262c9`.
RUNNING 12:49:26Z, **COMPLETE** 13:31:11Z, 41m06s of a 10,800s envelope.

## The headline

**The `on` arm produced no Actor proposal, no branch selection, and no
continuation submission.** Zero `contingent_events.jsonl` in its IPC, zero
mentions of `contingent_plan` across 88 model calls and every tool call.

("The controller was never invoked" is too strong as literal wording:
`headless.py:90-97` calls `_continue_contingent_plan` every turn in the `on` arm,
and it returns False without writing an event when no plan is pending. The
warranted claim is the one above, and it survives intact.)

In the card's precommitted words: a run where the `on` arm proposes no plan is
uninformative about the controller, not evidence against it. That is what this
is. It is not a null result and it does not weigh against the controller.

## Both arms completed: no censoring

| | off | on |
| --- | --- | --- |
| stop reason | `step_budget` | `step_budget` |
| actions | 20 / 20 | 20 / 20 |
| gameplay | 1,134.5s | 878.9s |
| s/action | 56.73 | 43.94 |
| model calls (actor) | 71 (61) | 88 (75) |
| actions per model crossing | 0.282 | 0.227 |
| levels completed | 0 | **1** |
| reasoning observed | none | none |
| assessor | PASS, 0 errors | PASS, 0 errors |

Row 1 of `PRECOMMIT-censoring.md`: both arms complete, the contrast is valid on
the completion axis. The asymmetric-censoring hazard did not materialise.

## What must not be claimed

The `on` arm completed a level and the `off` arm did not. **This is not a
controller result.** The controller was never invoked. One game, one seed, no
repetition. Attributing a level to an unused tool would be the exact error this
package spent the day being rebuilt to prevent. The same applies to the 22.5%
per-action speed advantage, which falls in the flattering direction the addendum
committed to reporting either way.

### Correction: the arms were not behaviourally identical

An earlier version of this section claimed the arms "differed only in six prompt
sentences describing an interface the model never used". A peer session
falsified that and the check reproduces:

    ipc/actions.jsonl submission rows, per-row action counts
    off: 13 rows  [1,1,1,1,1,1,1,5,1,1,1,1,5]  = 21 submitted, 20 executed
    on:  20 rows  [1] x 20                      = 20 submitted, 20 executed

The `off` arm batched twice; the `on` arm never did. That difference is the whole
of the crossings result: `off` obtained 21 actions from 13 decisions, `on` needed
20 decisions for 20 actions. Per decision the `on` arm was in fact *cheaper*
(3.75 actor calls per submission against 4.69); it simply made more decisions.

And one of the six sentences is not about the interface. `headless.py:203` reads
"If these estimates lack a useful basis, use ordinary single-action exploration."
That clause fires precisely when the interface is *not* used — which here was
always — and it recommends the behaviour the `on` arm exhibited 20 times out of
20. So the crossings regression is potentially **attributable to the treatment**
through a path the card did not anticipate: the fallback clause suppressing
batching that was already saving crossings.

### The competing explanation, which the same evidence supports

Trajectory divergence is a live confound and it points the other way:

| | off | on |
| --- | --- | --- |
| action mix | UP 15, DOWN 2, LEFT 2, RIGHT 1, **RESET 1** | UP 9, LEFT 5, RIGHT 4, DOWN 2 |
| both batches | `['UP']x5` twice | — |
| levels | 0 | 1 |

The `off` arm spent 71% of its actions on UP, batched five identical UPs twice,
declared itself stuck ("The player is stuck, so I need...") and spent an action on
`RESET`, and completed nothing. The `on` arm explored more diversely, never reset,
and completed a level on its final transition. Its two batches are the behaviour
of an agent repeating one move down a corridor, not obviously an efficiency win.

### Second correction: the thrashing mechanism is also wrong

The paragraph above proposed that the `off` arm's batches were a symptom of being
stuck. Two offline tests, run by a peer session and reproduced here from
`v3-output/`, falsify it.

**Did the `on` arm have batchable moments and decline them?** Counting the same
action repeated across *consecutive single* submissions:

    off: [('UP', 2)]                                    — it batched instead, twice
    on:  [('UP',3), ('LEFT',3), ('RIGHT',3), ('UP',3)]  — four runs, 12 actions, never batched

The `on` arm faced four separate occasions of one action repeated three times and
spent a crossing on every one. Had it batched those four runs it would have used
12 submissions for 20 actions — 1.67 actions per submission, *better* than `off`'s
21/13 = 1.62. The arm that scored worse had the structure available to score best.

**Were the `off` arm's batches thrashing?** Hashing before/after grids on every
transition:

    off:  ###########.########   19/20 actions changed the grid
    on:   ####################   20/20

The single no-change is index 11 — the last UP of the first five-UP batch. The
batch moved the player four times and stopped on a wall. That is corridor
traversal terminating on a boundary, the textbook good use of a batch. The
turn-10 rationale "The player is stuck" is what the model *said*; the grid says it
was moving on nineteen of twenty actions.

So the mechanism is retired. The suppression reading — the fallback clause at
`headless.py:203` discouraging batching — is now the better-supported of the two,
though still not established: n=1 per arm, stochastic, and the `on` arm's
three-runs could be steps where checking each result was genuinely correct, since
a novel corridor justifies looking after every move.

Note also what the second test does *not* show. Grid change is not progress:
productive-looking traversal in a wrong or looping direction is indistinguishable
from useful movement by this measure. The `off` arm moved on nineteen actions,
reset once, and completed nothing. "Not thrashing" is not "doing well".

### Third correction: the on arm could not batch, and neither earlier story is needed

Both explanations above are superseded by a constraint neither session checked.
An independent analysis found it; verified here from the IPC:

    model_feedback.jsonl  batch_allowed
      off: 13 rows, True at exactly start_step 6 and 14 — immediately before its two batches
      on:  19 rows, False in every one

`submit_actions` gates multi-action submission on a committed prediction, on
`batch_allowed`, and on a matching helper hash. **The `on` arm was structurally
ineligible to batch for the entire run.** So the "four declined opportunities"
that retired the thrashing reading were not opportunities, and the counterfactual
built on them is arithmetic over batches that would have been rejected.

The helper hashes say why:

    off: 7 distinct helpers — 57599996, 3c9efdd7, 14ad3a78, 9a55b99d, a2ec83ca, d4ff14b8, 33a0a6bd
    on:  1 helper — a2e699fa, across all 19 rows

The `off` arm repaired its executable prediction until it earned batch permission
twice. The `on` arm retained one contradicted helper and never earned it —
nineteen consecutive `prediction_mismatch` stops, then `level_up` at step 20.

**This outranks both earlier stories.** The submission-shape difference is
explained by batch eligibility, which is gated on predictive-helper agreement.
The fallback sentence at `headless.py:203` may still have an effect, but it is no
longer *needed* to explain anything, and with one trajectory per arm it cannot be
ranked above the gate explanation.

The "hit a wall" reading of the five-UP batch was also wrong. `batch_stops.jsonl`
records step 11 as `prediction_mismatch` with `skipped_actions` 1 — the batch was
cut short because observed background colors diverged from the frozen helper's
prediction while the player position matched. The unchanged UP at step 12 is a
separate later submission with reason `no_impact`. A game interpretation was
substituted for a logged stop condition, which is the same error as building on
the model's "the player is stuck" rationale: reading a story into the record
instead of reading the record.

Thrashing is therefore restored to live rather than retired. It is not
established either; the grounds for retiring it were simply wrong.

### The metric finding, restated correctly

Actions per model crossing rewards a **batching** agent, and the batch gate is
the thing that decides who gets to be one. Batch eligibility is granted on
predictive-helper agreement, so the measure is a proxy for *how well the agent's
executable prediction matches observations* — not for how much useful work it
does. Those came apart completely here:

| | off | on |
| --- | --- | --- |
| helper revisions | 7 | 1 |
| batch permission earned | twice | never |
| actions per crossing | **0.282** | 0.227 |
| levels completed | 0 | **1** |

The arm with the better predictive helper scored better on the primary measure
and completed nothing. The arm that never earned a batch completed a level. The
measure tracks prediction quality, which this run shows is orthogonal to
progress.

This is a cleaner statement of the defect than "it rewards thrashing", and it
changes the fix. The problem is not censoring robustness, which is why the measure
was chosen. The measure has no access to whether the actions it counts
accomplished anything, and the card needs a progress term it currently lacks.

## What the run does establish

The repairs work on the real Kaggle path, not just locally:

- The worker recorded its own arm from inside the worker process:
  `off -> contingent_arm 'off'`, `on -> contingent_arm 'on'`. The environment
  variable survived the allowlist, and the run self-certifies which arm it was.
- Both arms' assessors returned `PASS` with zero errors. The two inherited
  constants that would have killed version 1 — reasoning demanded, action limit
  pinned to 30 — are gone, and the repaired gate accepted two real valid runs.
- Reasoning stayed off in both arms across 159 model calls.
- The notebook completed inside its envelope: 2,466s against a 7,450s worst case.

## A budget lesson, in the safe direction

v1's ten-action arm measured 104.6 s/action and set this run's budget. v3's
twenty-action off arm measured 56.73. Per-action cost falls as a run proceeds, so
a short arm overestimates a long arm's pace — here by 84%. The error was in the
safe direction (a generous budget) and both arms finished on `step_budget` with
roughly 62% of admission unused. A successor can set admission from 60 s/action
rather than 105 and reclaim the slack for more actions.

## Next

Three questions now, where the card had one.

1. Does the Actor ever choose to use the interface? Twenty actions on ls20
   produced zero invocations, and adding actions is unlikely to fix that alone.
2. Does the instruction that describes the interface change behaviour even when
   the interface is never invoked? A prompt addition that costs crossings while
   its mechanism lies dormant is worse than a no-op. The cheapest test is the
   fallback sentence as its own arm against no sentence — a far smaller
   experiment than the controller comparison, and it needs no GPU-side change.
3. Is actions-per-model-crossing measuring useful work at all? It counts
   commitment, not correctness, and here preferred an arm that completed nothing
   over one that completed a level. The successor card needs a progress term.

No per-arm gate can see any of this: nothing in the package is scoped to notice
that the arms' submission shapes differ. That is the step (0) blind spot again,
one level up from the censoring case.

A successor should establish invocation before measuring benefit — either a game state that presents a genuine
observable branch point, or a diagnostic that confirms the Actor can reach the
interface at all under this prompt. Adding actions is unlikely to help by itself.
This run authorizes no promotion and no score-improvement claim.
