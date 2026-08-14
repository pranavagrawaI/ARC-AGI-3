# 0158 LS20 live run — exact effect interface

## Result

The exact interface passed local parity and two-phase Qwen provider
qualification. The single fresh LS20 run stopped with **0 actions** and **0
levels completed** because the claim submission interface still confused an
evidence-set digest with evidence record identifiers.

- Qualification: `artifacts/provider-qualification/20260812T132529Z-qwen36-exact-interface`
- Run: `artifacts/live-ls20/20260812T132630Z-qwen36-exact-interface`
- Run ID: `794d58bcd997489994dd43552507045d`
- Calls: 5 successful HTTP-200 crossings; no 429s
- Counted usage: 68,510 input plus output tokens
- Scorecard: terminal `PAUSED_FOR_REVIEW`, non-resumable

## What worked

Strict Pydantic translate, unchanged, and replace variants now match the host
parser. Runtime-accepted effects round-trip unchanged through the host. Qwen's
prior invented bounded-replace dictionary is rejected at the model boundary.
The new constructors and submission helpers passed claim-plus-commitment
qualification, including `translate_primary(d_row=-1, d_col=0)`.

## Live failure

Qwen inspected the real frame, used `make_mask`, and returned a structurally
valid claim. `submit_grounding_claim` still required a manual `evidence_refs`
list, while the prompt exposed the evidence-store content hash under the label
`evidence_commitment`. Qwen cited the real frame ID plus that hash. Runtime
validation accepted both nonempty strings; the host rejected the hash because
it is not an evidence record ID.

This again delegated a mechanical binding to the model and used ambiguous
terminology. A separate `classification-correction.json` supersedes the stale
`gemma_semantic` label with `interface_underspecification /
content_digest_exposed_as_evidence_reference` without editing the packet.

## Between-run repair

After snapshotting the executed candidate, claim submission was changed to:

```python
submit_grounding_claim(
    primary_region,
    alternatives,
    additional_evidence_refs=(),
)
```

The current observation ID is inserted mechanically. The evidence-set digest
is no longer included in the model task. Additional citations, when needed,
must be explicitly named genuine evidence record IDs. A regression test proves
automatic current-frame binding.

No further live run is authorized by the consumed approval.

## Verification

- All 20 executed candidate files were hash-checked and snapshotted.
- Every failure-packet checksum passes.
- Approval consumption is bound to this run and forbids resumption.
