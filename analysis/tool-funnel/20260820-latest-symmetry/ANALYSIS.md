# Latest symmetry replication: zero-cost tool funnel

## Judgment

Do not promote this run to a shadow or another live evaluation. The run completed
one level, but its evidence failed the hermetic gate and contains no exact
decision checkpoints. A bounded retrospective replay is mechanically valid and
finds a narrower actionable defect: the final saved world model is exact for
`UP`, `DOWN`, and `LEFT` in the first five visited states, but its declared
`SPACEBAR`, `UNDO`, and `RIGHT` transitions are incomplete.

This is a diagnosis of the final saved model against visited states, not the
historical model used by the actor and not evidence about gameplay benefit.

## Precommitment and cost

- Card: `EXPERIMENT-CARD.md` (957 UTF-8 bytes, frozen before execution).
- Source: `runs/tycho-nooa-symmetry-replication-ar25-20260820T141506Z`.
- Cap observed: 0 model calls, 0 ARC attempts, 5 replay packets, $0.
- Source top-level file commitments were unchanged.

## Stage 0: hermetic gate

The generated campaign passed all 12 declared fault families and both
metamorphic relations. The preserved run audit failed one contract:

```text
missing_outcome_checkpoint
level=0 turn_in_level=19
host-confirmed completion was not checkpointed
```

This licenses an evidence-lifecycle finding only. The current checkout has a
host-completion repair path, and its worker/trigger focused controls pass (2/2),
so the historical failure is not evidence that the current implementation still
has the bug. It remains unvalidated in a real checkpointed run.

## Stage 1: failure packet

Mining produced 20 packets with physical `admissible/` and `oracle/`
separation. All 20 have:

```text
model_temporal_validity=retrospective_final_candidate_not_historical
```

There are no decision checkpoint references. Consequently the final model may
be screened against earlier states, but it cannot be attributed to those
earlier decisions.

## Stage 2: bounded branch replay

All five selected public prefixes matched the committed local engine exactly;
the only ignored difference was the documented online/offline initial
`full_reset` marker. No branch used `UNKNOWN`.

| Action | Failed/scored | Mean penalized accuracy | Mechanical finding |
|---|---:|---:|---|
| UP | 0/5 | 1.000000 | Exact in the screen |
| DOWN | 0/5 | 1.000000 | Exact in the screen |
| LEFT | 0/5 | 1.000000 | Exact in the screen |
| RIGHT | 2/5 | 0.997656 | Missed 24 cells when the engine placed color 5 into column 30 |
| SPACEBAR | 5/5 | 0.999756 | Missed exactly one counter cell every time |
| UNDO | 3/5 | 0.984180 | Predicted no change while the engine changed 108 cells |
| RESET | 3/5 | 0.975464 | Predicted no change; RESET is not declared in model `actions()` |
| CLICK | 0/1 | 1.000000 | One grounded branch only; insufficient coverage |

Overall: 13/36 scored branches failed. The shortest retained failure needs no
recorded prefix: `SPACEBAR` already contradicts the final model at the initial
state.

The source explains the signatures. `transition()` handles `CLICK` and the four
directions, then returns every other action unchanged, although `actions()`
declares `SPACEBAR` and `UNDO`. Its `RIGHT` rule also refuses to overwrite wall
columns 30--32, while the engine counterexample has actual color 5 at column 30.
These are direct model/engine contradictions, not errors hidden by unknown
cells.

## Stage 3: shadow eligibility

Not eligible. The audit found `checkpoint_calls=0`. Approximate prompt or
workspace reconstruction is forbidden, so `shadow_eval --execute` was not run.

## Consequence

The strongest improvement is not another broad game run. Add the three minimal
counterexamples to the world-model repair loop:

1. initial-state `SPACEBAR` counter update;
2. the shortest `UNDO` state that reverses 108 cells;
3. sequence-3 `RIGHT`, where the wall-column assumption fails.

The last two builder fires spent 86 tool calls, including 74 `run_python` calls,
and made no world-model edit. Yet the counterfactual screen finds declared-action
failures. This supports changing builder verification from recorded-path-only
simulation toward bounded legal-action branch checks. It does not show that the
builder report caused or failed to cause the actor's decisions.

After these cases pass locally, the next external evidence should be one short,
checkpoint-producing run—not a broad replication. Audit it first; only then
prepare a four-call shadow around one precommitted content perturbation.

## Raw evidence

- `hermetic-campaign.json`
- `hermetic-audit.json`
- `failure-atlas/`
- `replay-5/`
- `derived-metrics.json`
- `shadow-eligibility.json`
- `source-sha256.before.txt` and `source-sha256.after.txt`
