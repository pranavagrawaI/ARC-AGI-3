# 0158 LS20 live run — required tool

## Result

The single approved, non-resumable run stopped with **0 actions** and **0 levels
completed** after five successful provider crossings. It is not a valid test of
the required-tool treatment.

- Run: `artifacts/live-ls20/20260812T125103Z-gemma31b-required-tool`
- Run ID: `886db695978c4aa080b9d64266642b0b`
- Provider: `coreweave/bf16`, fallback disabled
- Counted usage: 23,243 input plus output tokens
- Scorecard: terminal `PAUSED_FOR_REVIEW`, non-resumable

## Treatment did not reach the wire

The adapted wrapper replaced CodeAct's call-site value with `required`, and the
canonical audit recorded that value. However, pinned NOOA `ResponsesClient`
unconditionally assigns `tool_choice = "auto"` after merging call arguments.
Every audited HTTP request body therefore carried `auto`.

The preflight fake provider sat above that second overwrite, so its passing
assertion did not exercise the load-bearing HTTP representation. This run is
invalid for the v14 contrast and cannot establish an effect of required tool
choice.

## What Gemma actually did

Gemma nevertheless made one `execute_python` call on each of five HTTP-200
crossings; there were no 429s. Thus tool non-use did not recur in this run.

The first, second, fourth, and fifth cells submitted a `primary_region` using a
four-element `bbox` list and a `mask` field. The frozen schema requires a bbox
object (`row`, `col`, `height`, `width`) and `rows`. The third cell tried to call
`frames()` but shadowed the function with the local assignment `frames =
frames()`, producing `UnboundLocalError`. Gemma also redefined, ignored, or
miscalled the supplied `make_mask(row, col, rows)` helper. It exhausted the
initial attempt plus four allowed CodeAct errors without returning a claim.

## Classification correction

The immutable generic packet classified the terminal exception as
`model_generation`. A separate `classification-correction.json` records the
more specific evidence-backed result: `model_semantic /
schema_repair_exhaustion`, with `treatment_validity=false`. The original packet
was not edited and its checksum manifest verifies.

## Next boundary

A real required-tool candidate must change or wrap NOOA at the final Responses
request construction point and assert `tool_choice=required` from the captured
HTTP body—not only from a fake provider above it. That is a new candidate and
requires a fresh card and approval. Separately, the schema/helper interface
should be reconsidered because the model ignored exact repair feedback even
while using CodeAct.

## Verification

- All 20 executed candidate files were hash-checked and snapshotted before any
  further source change.
- All failure-packet checksum entries pass.
- The approval consumption is bound to this run and forbids resumption.
