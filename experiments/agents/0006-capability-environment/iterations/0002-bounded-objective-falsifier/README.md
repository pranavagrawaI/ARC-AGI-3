# Iteration 0002 — bounded objective falsifier

Status: **rejected by human review**.

## Observed failure selected from iteration 0001

Both arms confirmed that action 6 could change a bottom-strip pixel, named the strip a
progress bar, and promoted that local mechanism to the game objective. The control used
action 6 on 44/44 steps. The treatment used it on 43/44 steps, tried action 1 once, never
tried actions 2–5, completed zero levels, and never clicked the game-relevant location.

Highest-leverage bottleneck: **failure to distinguish a responsive mechanism from the
game objective**.

## Smallest question

Can a model preserve a local mechanism result while allowing a separately declared game-
objective claim to be falsified by bounded absence of level progress?

## Narrow treatment delta

The existing `learning_loop` commitment gains one `objective` field.

When no objective is active, the model must author:

```json
{
  "claim": "model-authored objective hypothesis",
  "progress": {
    "observable": "levels_completed",
    "op": ">",
    "value": 0
  },
  "horizon": 2
}
```

The host may only:

1. preserve the claim and public assertion verbatim;
2. count down its declared 1–8 action horizon;
3. compare `levels_completed` mechanically after each action;
4. report `pending`, `supported`, or `falsified`;
5. prevent the model from resetting an active horizon;
6. require reconciliation and a new model-authored objective after falsification.

`progress` must be exactly `levels_completed > N`, where `N` is the public level
count when the claim is created. A no-progress predicate such as
`levels_completed == 0` is rejected rather than counted as objective support.

The existing per-action `prediction` remains an immediate mechanism check. A mechanism
match does not alter the objective result.

## Explicit non-goals

- no action-coverage quota;
- no host-authored objective or competing hypothesis;
- no semantic plausibility/absurdity classifier;
- no salience detector;
- no claim that level completion is the only useful qualitative evidence;
- no automatic acceptance from contract uptake.

## Offline verifier

A three-action fixture produces:

```text
mechanism: match, match, match
objective: pending, falsified, pending
```

The second action falsifies the first objective despite another local mechanism match. The
third action must reconcile and author a new objective. Pair diff must show no unauthorized
arm differences; the frozen control manifest must remain byte-identical.

## Live falsifier

Reject this iteration if any of the following occurs:

1. local mechanism matches preserve an objective beyond its declared horizon without level
   progress;
2. objective falsification does not materially change the next hypothesis and action
   strategy;
3. the model merely rephrases the same objective and continues the same behavior;
4. the new contract adds enough ceremony or repair traffic to reduce useful exploration;
5. game progress or action efficiency remains no better and the trace shows no compensating
   epistemic improvement.

The complete viewer trace remains authoritative. Objective-result labels are navigation
evidence, not an acceptance verdict.

## Execution evidence

- `live-treatment-cd82-v1` failed before its first action because the frozen base
  final-round path validates the same response twice and treatment acceptance was
  initially mutating rather than idempotent. The failed artifact is preserved.
- Acceptance now treats only the identical pending response as idempotent before a
  transition; a different replacement remains invalid. This passed the real path in v2.
- `live-treatment-cd82-v2` completed 44 actions under matched conditions.
- Frozen control manifest: unchanged.
- Unauthorized arm differences: none.
- Objective results: 36 pending, 8 falsified.
- Every falsified claim was exactly `Complete the first level`; the model reinstated
  that tautological claim after each five-action expiry rather than authoring a causal
  objective hypothesis.
- Mechanism results: 34 match, 10 mismatch.
- Action IDs: action 6 on 44/44 actions.
- Completed levels: 0; official game RHAE: 0.

Mechanical evidence: [`evidence.json`](evidence.json). Human verdict:
[`human-review.json`](human-review.json).

The review found no material strategy change. More deeply, the model did not represent
objective uncertainty. At frame 26 it invented an absent user's intent—`The user wants to
fill the bottom row of the grid`—then devoted perception and coordinate-recovery effort to
optimizing under that unsupported premise. Contradiction at the objective-result layer did
not reopen objective discovery. Iteration 0002 is rejected.
