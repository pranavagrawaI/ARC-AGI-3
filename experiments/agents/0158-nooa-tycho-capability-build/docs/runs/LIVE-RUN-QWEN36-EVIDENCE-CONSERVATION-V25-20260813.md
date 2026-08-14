# 0158 Qwen3.6 evidence conservation v25 — 2026-08-13

## Outcome

The fresh non-resumable run executed six confirmed LS20 actions and completed no
level. The seventh claim artifact was returned, but the protected host rejected
its mechanically inconsistent residual status and closed the run.

| Measure | Observed |
|---|---:|
| Accepted claims / commitments | 6 / 6 |
| Actions / confirmed successors | 6 / 6 |
| Selected actions | `1, 2, 1, 2, 1, 2` |
| Verdicts | 3 supported, 3 contradicted |
| Reciprocal contrast / G2 | 0 / 0 |
| Levels completed | 0 |
| Provider calls / reported tokens | 69 / 790,471 |
| Provider-reported cost | $0.68773596 |

The scorecard is terminal `PAUSED_FOR_REVIEW`, `resumable=false`. Evidence is
under `artifacts/live-ls20/20260813T131651Z-qwen36-evidence-conservation-44`.

## Precommitted discriminator

The intervention changed the behavior at the intended boundary.

1. Action 1 contradicted the static 3x3 claim. Mechanical accounting preserved
   `0` changed cells inside and `52` outside.
2. The next claim explicitly `abandon`ed that hypothesis and kept the residual
   `unresolved`; it did not silently erase the observation.
3. After the second 3x3 contradiction, the third claim `revise`d onto a 2x5
   slice of the moving `9/12` block.

Thus the predicted post-contradiction update occurred, and the primary region
shifted from the proxy to action-varying movement before action 12. This is a
real improvement over v24, not L1 success.

## Remaining semantic failure

Qwen still narrowed the moving object to its two rows of `12`s and predicted
only their replacement by `3`. Actions 3, 4, and 6 supported this local effect;
action 5 contradicted it. Every one of those successors changed 10 selected
cells and 42 cells outside the primary region.

The update obligation made those 42 cells visible, but did not make their
meaning load-bearing. Qwen first called them `unresolved`, then described them
as “auxiliary color swaps” and marked them `explained`. They were principally
the rest of the same 5x5 translation plus the two-cell automatic strip update.
It therefore retained a smaller locally predictable slice rather than binding
the full movement `1 = up 5`, `2 = down 5`. No reciprocal action-effect pair
closed and the gate remained G0.

This is nominal evidence accounting: the residual no longer vanished from the
record, but a semantic label allowed it to vanish from the model's operative
hypothesis. The host cannot correct that label without becoming oracular.

## Terminal failure and corrected classification

After action 6, accounting again recorded `outside_primary_changed=true` with
42 outside changes. In claim 7, Qwen called:

```text
account_prior(disposition='revise', residual_status='none', ...)
```

The prompt explicitly said `none` is legal only when outside change is false.
However, the model-side `account_prior()` constructor validated the enum but not
that known consistency rule. It returned the artifact; only the protected host
parser then raised:

```text
ProtocolError: residual status must be none exactly when no outside-primary change occurred
```

The immutable run classifier recorded `model_semantic`. Causally, the model did
violate an explicit rule, but terminality also reflects validator placement: a
mechanical invariant known inside the sandbox was checked only after the
CodeAct repair loop. The strongest classification is mixed model-protocol plus
interface-validation placement, not evidence against the underlying action API.
The exact accepted artifact followed by the host rejection is the reproducing
trace. Moving the same check into `account_prior()` would make the violation
repairable, but does not establish that Qwen would repair it.

## Decision boundary

The host should continue to preserve coverage and enforce mechanical
consequences, while the model authors relevance and causal interpretation. A
next candidate should therefore avoid teaching which cells belong together.
The promising general constraint is operational accountability for semantic
labels:

- `explained` must carry a prospective prediction for the residual;
- `auxiliary` must carry a prospective action-insensitivity contrast;
- `unresolved` remains open and cannot silently count as explanatory closure.

That would test the model's interpretation without letting the host choose it.
It is a proposed next-run mechanism, not a conclusion established by v25.
