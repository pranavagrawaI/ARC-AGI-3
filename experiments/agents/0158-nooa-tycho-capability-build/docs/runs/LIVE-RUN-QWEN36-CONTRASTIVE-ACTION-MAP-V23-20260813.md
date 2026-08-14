# 0158 Qwen3.6 contrastive action-map v23 — 2026-08-13

## Outcome

The fresh run stopped before its first ARC action. The claim artifact closed
after five provider calls, but the commitment artifact exhausted all 12 CodeAct
iterations without returning a valid contrastive commitment.

| Measure | Observed |
|---|---:|
| Actions / successors / verdicts | 0 / 0 / 0 |
| Levels completed | 0 |
| Provider calls / HTTP 200 | 17 / 17 |
| Reported tokens | 131,651 |
| Provider-reported cost | $0.08762316 |

The terminal scorecard is `PAUSED_FOR_REVIEW`, `resumable=false`. Raw evidence is
in `artifacts/live-ls20/20260813T111406Z-qwen36-contrastive-action-map-44`.

## First broken link

The new prompt named `predict_replace()` but showed only a translation example;
it never exposed the exact replacement form
`predict_replace(expected_after, assertions=[cell_after(...)])`.

After three semantic explorations, Qwen spent its remaining calls reconstructing
that interface from errors. On call 11 it reached
`predict_replace(mask, cell_after(...))`. The helper therefore iterated the
assertion dictionary into `['row', 'col', 'value']`, and typed validation rejected
six assertion entries. On call 12 Qwen fell back to identical `Unchanged`
predictions, which correctly violated the required distinct-effect contrast.
The artifact then exhausted its iteration cap.

## Discriminating probe

An offline two-arm replay changed only the missing list wrapper:

- direct `cell_after(...)` argument: rejected with `CellAssertionModel` errors;
- `[cell_after(...)]`: accepted end to end in a fresh real NOOA sandbox.

This supports an interface-legibility/procedural-closure diagnosis. All 17 live
provider crossings returned HTTP 200, so transport is contradicted as the cause.
No commitment, action, successor, or verdict existed; therefore the run is
invalid for judging reciprocal action learning and provides no gameplay result.

## Decision boundary

Keep the contrast mechanism unpromoted and the run permanently closed. The
smallest proposed next change is to restore one exact replacement-constructor
example without changing action semantics, evidence, verifier, budgets, or
model. `review.yaml` remains pending for Pranav's initial causal review; no fresh
run is authorized by this record.
