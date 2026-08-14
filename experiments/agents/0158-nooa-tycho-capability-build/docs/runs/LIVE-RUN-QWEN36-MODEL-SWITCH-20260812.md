# 0158 LS20 live run — Qwen3.6-27B model switch

## Result

Qwen3.6-27B passed synthetic provider qualification, then the one approved
fresh LS20 run stopped with **0 actions** and **0 levels completed**. It reached
an accepted grounding claim but was stopped by an underspecified commitment
effect interface before action authority.

- Qualification: `artifacts/provider-qualification/20260812T130458Z-qwen36`
- Run: `artifacts/live-ls20/20260812T130550Z-qwen36-model-switch`
- Run ID: `d536e1da3bf041689ca0c5edd6161cfb`
- Route: `qwen/qwen3.6-27b` on `coreweave/fp8`, fallback disabled
- Calls: 11 successful HTTP-200 crossings; no 429s
- Counted usage: 135,118 input plus output tokens
- Scorecard: terminal `PAUSED_FOR_REVIEW`, non-resumable

## Qualification

Against a synthetic 4×4 public frame, Qwen called `execute_python` and returned
a typed 2×2 primary mask plus distinct typed alternatives on its first artifact.
No ARC environment was available to this qualification.

## Live grounding behavior

Qwen initially shadowed a guessed `grid()` function, then repaired to the actual
`frames()` API. It inspected the 64×64 frame, computed value counts and bounded
regions, used the supplied `make_mask`, and returned a schema-valid claim. The
host accepted a 3×3 primary region at row 31, column 20 and three distinct
alternatives. This is further through 0158 than the Gemma runs reached.

The claim is not evidence that the selected region was actually controllable;
no action executed, so prospective identity was never tested.

## Commitment boundary failure

The model-visible instruction described one primary-region-anchored
`translate`, `unchanged`, or `bounded replace` effect, but did not provide exact
keys. The sandbox's `ProbeCommitmentDraftModel` typed `predicted_effect` only as
an arbitrary dictionary. Qwen therefore returned:

```text
type: bounded_replace
primary_region_anchor: {row, col, height, width}
replacement: [binary mask rows]
```

The sandbox accepted that artifact. The host parser, which was not visible to
the model, expected a `kind` discriminator and different fields, then rejected
the commitment with `unsupported predicted_effect kind: None`.

This is a split-validation defect: an artifact accepted at the model boundary
must not be rejected later by a hidden stricter representation contract.

## Classification correction

The immutable packet uses the stale model-specific class `gemma_semantic`. A
separate `classification-correction.json` records the evidence-backed class
`interface_underspecification / commitment_effect_schema_not_model_visible`.
The original packet remains unchanged and its checksum manifest verifies.

## Next intervention

Keep Qwen and make `PredictedEffect` exact and model-visible at the sandbox
boundary, using the same parser/schema as the host. This is one representation
repair, not additional gameplay knowledge. Qwen's invented convention should
not be adopted blindly: its `replacement` was a binary membership mask and did
not fully express the host's required prospective cell assertions.

Any test of that repair requires a new card and fresh run; this trajectory can
never resume.

## Verification

- All 20 executed candidate files were hash-checked and snapshotted.
- Every failure-packet checksum passes.
- Approval consumption is bound to this run and is non-resumable.
- The provider qualification and live worker audits preserve exact model,
  provider route, tools, requests, responses, and usage.
