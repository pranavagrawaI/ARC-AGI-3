# Independent review disposition

Three subagents reviewed the complete checker/test/docs candidate on separate
axes: [implementation](reviews/implementation/REVIEW.md),
[adversarial tests and preservation](reviews/adversarial/REVIEW.md), and
[Flash raw evidence](reviews/flash-evidence/REVIEW.md).
The parent reproduced findings, implemented fixes, and reran the affected checks.
Final machine reports are in `reviewed-final/`; earlier outputs remain preserved.

## Findings and disposition

| Finding | Disposition and verification |
| --- | --- |
| Malformed nested provenance/resource fields crashed after detecting OOM | Fixed type checks and empty GPU-field handling; known OOM remains invalid with a JSON evidence report |
| A string `actions: "UP"` counted as a batch; null actions crashed | Fixed action-list and join-key validation; malformed rows are explicit and cannot become a supported batch count |
| Null fixture usage crashed; malformed fixture counts disappeared | Fixed fixture validation with file/line diagnostics, matching live-count validation |
| Preservation silently omitted FIFO/special entries | Reject unsupported entries before creating the destination; regression verifies no successful manifest |
| v2 failed response excluded from accounting/hash inventory | Inventory raw artifacts before results index: 3 issued/3 preserved, 2 successful index rows; hash the failed request/response/SSE too |
| New strict hash validation rejected legitimate no-model feedback | Fixed against runtime unmeasured/unavailable false-permission records; both historical arm inventories retain supported scoped status |

Reviewer rechecks are preserved in [implementation RECHECK](reviews/implementation/RECHECK.md)
and [adversarial ADDENDUM](reviews/adversarial/ADDENDUM.md). The latter checked the
46-test intermediate revision; final integration expanded coverage to 49 tests.
No original supported finding remains open.

## Completed evidence review

The independent [raw reconciliation script](reviews/flash-evidence/verify_raw.py)
was also run by the parent. All 11 issued responses reconcile with raw SSE,
request hashes, saved response fields and offered tool schemas. Recorded model
revision, runtime and GPU identity match retained pins. Observed setup and
individual request durations satisfy the relevant caps; cleanup records show
owned shutdown. Both v2/v3 remain operationally invalid due to OOM.

The v2 JSON answer is valid JSON with failed expected content. Its exact-output
stop is authorized by the card. Prompt ambiguity is a plausible explanation,
not an isolated causal finding; the old reports now explicitly supersede that
attribution. Five Actor requests were not issued, rather than issued and lost.

The card was reused across distinct sessions, despite its single-session/no-retry
boundary. That cannot be described as one compliant attempt. The retained runs
do not establish whether separate human authorization was given.

## Evidence limits, not unfinished local review

Full model/runtime payload hashes are explicitly unverified in fast-mode
provenance and those payloads are not retained here. Platform quota billing is
not independently available. Live Actor logs retain reported token counts but
not full serialized requests, and candidate-tokenizer/template parity is not
established. These remain unavailable evidence, not grounds for a new claim.
Recorded durations do not test counterfactual timeout enforcement.

The generic offline Flash command intentionally continues to list manual identity,
semantics and cap obligations; this review closes the available historical parts
with the qualifications above, and is not a reusable automatic sign-off.
Future launcher/publication wiring remains outside this review. The original
probe's nonempty-reasoning abort is still absent from the explicit frozen Stop
list; document or remove it prospectively in a new candidate/card, not in the
preserved historical run.

## Final verification

```bash
uv run --no-project --with pytest --with numpy --with jsonschema python -m pytest tools/tests/test_experiment_checks.py -q
uv run --no-project --with jsonschema python analysis/deployment-probes/20260909-experiment-checks/reviews/flash-evidence/verify_raw.py
```

49 tests pass. Raw reconciliation passes. Final CLI outcomes: Flash v2/v3 invalid;
off/on batching inventories supported; reported context coverage invalid with
candidate-tokenization parity unresolved. No model, GPU or game calls occurred.
