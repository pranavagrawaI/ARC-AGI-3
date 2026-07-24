# Experiment 0003: Reconstruction Uptake Replay

## Governance status

**INVALIDATED PROTOTYPE — frozen 2026-07-21.** The activation predicate admits
vacuous action-echo and shared-state-mutation implementations, so this protocol
reached executable API compliance (claim L0) only. Its code and original
acceptance rules are intentionally preserved rather than silently repaired.
Experiment `0004-counterexample-gated-ledger` supersedes it prospectively.

Development-stage offline experiment testing whether a required executable contract induces prospective reconstruction where voluntary availability did not.

## Question

Does requiring a controller-invoked `world_model.py` increase verified reconstruction activation relative to making the same capability optional?

This experiment does **not** test gameplay benefit. It sends no action to ARC-AGI-3 and assigns no game score.

## Experimental unit

One fresh model conversation and fresh isolated Python workspace replaying a fixed contiguous window from one admissible `trace.jsonl`.

## Conditions

- `optional`: executable reconstruction is available but not required.
- `required`: every reveal is blocked until `world_model.py` loads with `parse_observation`, `predict`, and `update`; the controller executes `predict` for all legal actions and commits the recorded action's result.

Both conditions receive the same observations, recorded actions, model configuration, context limit, Python tool, and reveal order. The condition passage is the intended difference.

## Ordering contract

For each step:

1. Write only the current observation to the workspace.
2. Reveal the recorded action and legal actions.
3. Let the model inspect evidence and optionally or mandatorily edit `world_model.py`.
4. Controller-load the artifact and execute candidate predictions when available.
5. Persist `prediction_committed`.
6. Reveal the next observation.
7. Controller-execute `update` when an artifact and state exist.
8. Persist `model_updated`.

`SequentialReplay` has no interface that returns the next observation before a prediction commitment.

## Primary outcome

`activated` is true only when one run jointly satisfies:

- a SHA-256-bound executable artifact loaded;
- all three entrypoints existed;
- every eligible step had a nonempty pre-reveal commitment;
- candidate predictions were action-sensitive at least once;
- every revealed transition was followed by a successful update;
- every commitment was bound to a loaded artifact identity.

A file that exists but is not prospectively invoked fails. A constant or empty model fails action sensitivity or nonempty prediction criteria.

## Development window

The proposed first window is steps 1 through 17 of the accepted attempt-2 enabled trace. The trace path and SHA-256 must be committed in the run manifest. During development, use `--dry-run`; do not call the model until the prompt, source trace, scoring schema, and request-evidence contract have been reviewed and frozen.

## Interpretation matrix

- Optional fails, required passes: voluntary uptake was a bottleneck.
- Both pass: prior non-uptake may be stochastic or task-context dependent.
- Required creates a file but activation fails: compliance theater or an unusable contract.
- Required activates but predicts poorly: uptake succeeded; model quality is the next bottleneck.
- Both fail: the interface, model capability, or evidence presentation remains inadequate.

Passing licenses only the claim that the contract induced verified executable reconstruction on this replay. It does not license claims about live gameplay, level completion, generalization, or benefit.

## Safety and authorization

- `--dry-run` performs no model or game calls.
- `--execute-model` performs paid model calls but no game calls.
- Model execution requires the exact explicit flag `--confirm-model-cost I_ACCEPT_OFFLINE_MODEL_COST` and an API key environment variable.
- A later live-game pilot requires a separately repaired and accepted authorization pipeline.
