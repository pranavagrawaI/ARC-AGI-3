# Independent local review and response

Scope: separate saturated S10 CPU reducer/card, no adapters, model, GPU, launch packet or production qualification. Reviewer: native paired_card_review (John Tukey); all review work read-only.

First review confirmed two concrete failures with actual immutable corpus wire bytes:

- A control native trace extended through 4000s overlapped the candidate capture at 2000–3500s, but the pair passed because only client capture end was compared. Repair: arm interval uses earliest client/native capture start and latest capture end; candidate must start after the entire control interval. Native capture must also cover replay origin.
- Readmitted ar25 could release immediately at the same trim cursor, skipping its FIFO turn without another response. Offered-load summary still showed ten pending and no starvation. Repair: unfinished release requires a fresh response-created handover entitlement; admission consumes the previous entitlement.

Both new public regression tests failed before the repair (2 failures in a 3-test run, 8.510s) and passed after it in the 25-test suite (30.217s). The third test independently demonstrated that a fast sequential fixture can reach QUALITY_INVESTIGATION_ONLY.

A subsequent raw-artifact probe demonstrated that distinct requests could reuse one raw response ID, even when the native receipt repeated it. That new regression failed before repair (2.457s). Client and native response IDs must now be nonempty unique strings. The regression then passed.

Independent recheck: all 26 public tests passed in 29.751s using cached dependencies and an 1800s timeout; no additional material finding remained in the local reducer/card scope. It confirmed rejection of overlap, FIFO bypass and response-ID reuse; acceptance of low native occupancy; and access to the quality-investigation branch. Gate tolerances and CPU-only authority were reviewed. It did not qualify actual runtime performance, adapters or the then-pending sensitivity output.

Independently reviewed SHA256:

- capacity.py: 445ff8ff1f25f2118c8aa97ce21bf4eb738e443935f32d38c943dd43c8b38e7d
- test_capacity.py: 1271ec7a348debc530cd4866b0f0007270fc02ee3b4e1a8aad71d1dcc6a3f693
- EXPERIMENT-CARD.md: ce8647252ea9a7373e0f3b5529665ee1128e9fb155df91e86a19a4f0872167ba

Unresolved by design: native/runtime reconciliation of the historical 11-token offset, actual image preprocessing, pinned native adapters and trace wiring, real dispatch timing, load fit, supervisor/cleanup, installed identities and temporal-order confounds. CPU fixtures do not provide this evidence. Trial-3 stays DESIGN BLOCKED; this review changes neither prior trial nor production status.

Final artifact recheck: reviewer reconstructed Reader requests/responses from cpu-proof/ and called capacity.compare on archived event bundles without running Simulation. The complete result exactly matched archived verdict.json (synthetic ratio 1.5006912855971657, QUALITY_INVESTIGATION_ONLY). All qualification, sensitivity and proof hashes matched. All 27 sensitivity cells reconciled: 9 feasibility-only, 12 close, 6 quality-investigation; all 54 arm offered-load checks passed. Invalid arms failed sample/block stability. Preservation checks were independently reproduced (39 trial-1 files, 24 trial-2 files, 56 trial-3 inventoried files). No additional material issue found. Hypothetical decode multiplier remains an input, not a replay-rate estimate or runtime forecast: input 1.5 produced ratio 1.2631 in a slower-prefill fixture. Final review used no network, GPU or source edits.
