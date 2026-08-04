# 0006 iteration 0001 — minimum empirical loop

Status: implementation candidate; not accepted until manual full-trace review.

## Question

Can one thin treatment surface help the model preserve and test its own empirical account without prescribing ARC semantics or hiding regressions from the reviewer?

## Frozen comparator

Reuse one recorded 0002-R run while model, provider request, game, seed, budgets, generic harness contract, and observation interface remain identical. A changed condition invalidates the comparison and requires a new control anchor; iteration alone does not.

## Minimum treatment contract

The treatment adds one `learning_loop` capability. Before each action the model authors:

- an arbitrary bounded JSON `working_state`; the host preserves its semantic JSON
  value but may canonicalize key order and whitespace, and does not interpret or
  revise it;
- a brief test purpose;
- one to five mechanically checkable predictions over public outcomes;
- zero to four prior transition step numbers it wants returned in the next packet;
- a reconciliation only when the preceding prediction was contradicted.

The host:

1. validates only shape, bounds, and public observability;
2. echoes the latest model-owned state after transcript eviction;
3. returns exact recorded transition objects for model-selected steps, with observation-file pointers rather than copied pixel grids;
4. compares committed predictions with the executed public transition;
5. records the comparison in `process.jsonl` and the model-visible packet;
6. never creates, ranks, selects, or revises hypotheses.

Initial public observables are deliberately small: final-layer changed-pixel count, completed-level count, public state, and one final-layer cell value. Evidence may justify adding another observable; imagined usefulness does not.

## What automation establishes

- the treatment is the control plus one declared capability;
- the pre-action commitment was valid and recorded before the transition;
- the result was computed from public outcome fields;
- the requested prior transition records were returned exactly;
- contradiction makes reconciliation structurally mandatory;
- artifacts load in the existing viewer.

These are mechanism facts, not a behavioral verdict.

## Human verification gate

The reviewer reads the complete treatment trace in the existing viewer. The review is open-ended and outcome-directed: record anything that materially improves or harms game progress or action efficiency, including unknown failure classes.

For each material finding record:

- **Observation:** exact step/event and what happened;
- **Consequence:** effect on progress or action efficiency;
- **Interpretation:** plausible explanation, kept distinct from observation;
- **Hypothesis:** smallest mechanism change that may help;
- **Falsifier:** what the next run would need to show against that hypothesis.

After review, choose one highest-leverage evidenced bottleneck. The next iteration changes one mechanism while retaining this full-trace review. Interface uptake, prediction coverage, mismatch counts, and reconciliation counts are diagnostics only.

## Stop rule

Do not add taxonomies, automatic key-moment selection, semantic scoring, RHAE gates, endpoint orchestration, or production schemas before the manual review establishes demand. Do not run a second unchanged 0002-R control.
