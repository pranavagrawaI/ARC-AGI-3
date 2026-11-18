# Precommitment: asymmetric censoring between the arms

Written at 2026-09-09 while kernel version 3 (contract
`2c1e13ef4ce25e59121bcd6c637a869a8cb09c49fc58712fc9212234b8c262c9`) is RUNNING
and before any result exists. Raised by a peer session; the risk is real and no
gate in the shipped package can detect it.

## The hazard

Both arms get 20 actions and 3,000 seconds of admission. The per-action pace used
to set that budget — 104.6 s/action — was measured on v1's **off** arm, the arm
with the treatment inert. If the `on` arm's contingent path is slower per action,
it can hit `wall_budget` while the `off` arm finishes on `step_budget`. The
comparison would then be between a complete arm and a truncated one, which is
worse than both being truncated, and it would pass every existing check.

## Why nothing catches it

`assess_candidate.py` accepts `wall_budget` as a valid stop reason (line 64) and
records `actions` and the stop reason per session (lines 53-56). The data to
detect this will exist. But the assessor runs **once per arm, inside that arm's
cell**, and has no view of the other arm. Asymmetric censoring is a cross-arm
property and there is no cross-arm gate anywhere in the package. This is a
structural blind spot, not a missing assertion.

## Threshold

    off-arm measured pace                104.6 s/action
    on-arm truncation threshold   3000/20 = 150.0 s/action
    headroom before the on arm truncates          +43.5%
    off-arm projected finish     2,091s of 3,000s admission

So the `on` arm truncates only if it is more than 43.5% slower per action than
the off arm was. Direction is genuinely uncertain: a contingent continuation is
submitted *without* a model call, which should lower cost, while proposing a plan
costs a larger request and response, which raises it.

## Committed reading

Determined before the data, from `game-stops.jsonl` in each arm:

| Off arm stop | On arm stop | Reading |
| --- | --- | --- |
| `step_budget` | `step_budget` | Both complete. The contrast is valid on this axis. |
| `wall_budget` | `wall_budget` | Both truncated. Valid but underpowered; report actions completed per arm and treat completion counts as censored. |
| `step_budget` | `wall_budget` | **Asymmetric censoring. The completion comparison is INVALID** and must not be reported as a controller result. Actions-per-model-crossing remains readable because it is a rate, not a total. |
| `wall_budget` | `step_budget` | Same invalidation, mirrored. Also surprising, and would mean the controller is faster per action — worth reporting as its own finding. |
| either | `win` | Report the win; the budget question is moot for that arm. |

In the two asymmetric cases I will report the censoring as the headline finding
rather than the controller comparison, and the fix for a successor is equal
*completed actions* per arm rather than equal budgets — truncate the longer arm
to the shorter arm's action count before comparing.

## Not committed

Nothing here licenses a controller claim in any branch. The card's existing
limits stand: twenty actions may still be too few for a plan to arise, plan
estimates remain model conjectures, and no score-improvement or promotion claim
follows from this run.

## Follow-up for the successor package

Add a cross-arm check that runs after both arms and fails on mixed stop reasons.
Not applied to version 3: the package is frozen and running, and editing it now
would leave the local tree not matching the artifact under test.

---

## Addendum, 12:53:50Z — kernel still RUNNING, no result exists

The original table above is unchanged; this is appended, not a rewrite.

A peer session pointed out that the two asymmetric branches are not equally
dangerous, and they are right. `off wall_budget + on step_budget` means the
control was truncated while the treatment completed. That reads as a controller
win. By the failure pattern this whole day has demonstrated — verification effort
allocated by whether the answer is welcome — it is the branch that will receive
the least scrutiny, while its mirror invalidates in a direction nobody wants and
will be checked hard. The branch most likely to slip through is the one the table
above labels "its own finding".

So, committed now, before knowing which branch landed:

1. In `off wall_budget + on step_budget` the completion comparison stays
   **INVALID**, exactly as in the mirror branch. The "its own finding" clause
   does not soften that and must not become the headline.
2. The claim "the controller is faster per action" is **not** licensed by the
   stop reasons. `off` hitting `wall_budget` has other explanations that have
   nothing to do with the controller: a slower server warmup in that arm, a
   stall, a long single model call, or ordinary variance. Committing to the
   test: report that claim only if the `off` and `on` arms' measured
   `gameplay_elapsed_seconds / executed_solver_actions` from `game-stops.jsonl`
   show the `on` arm strictly lower, and report both numbers whether or not they
   support it.
3. If that branch lands I will state explicitly, in the result, that it is the
   flattering branch and that I applied this check because of the risk of not
   applying it — so a later reader can see the scrutiny was precommitted rather
   than chosen after the fact.

This addendum adds no new licence to claim anything. It removes one.
