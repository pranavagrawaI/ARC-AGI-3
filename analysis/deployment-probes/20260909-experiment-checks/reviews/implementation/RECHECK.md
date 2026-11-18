# Bounded second pass

Re-ran `reproduce.py` after the implementation fixes. All original reproductions
now produce the intended invalid or explicitly unresolved reports: OOM survives
`epoch:null`, malformed actions do not count as batches, and malformed fixture
usage produces source-specific diagnostics.

Reviewed changed evidence parsing, raw-issued-response accounting, batching
validation, and coverage validation. No additional accounting regression found
in this bounded pass.

One remaining P2 in the same resource-parser defect: `experiment_checks.py:212`
indexes the first token of a GPU memory column without handling an empty column.
An OOM log followed by `{"epoch":10,"gpu":"GPU, , 2 MiB, 0 %"}` still raises
`IndexError`, preventing the known invalid verdict from being returned. Validate
columns before indexing or handle `IndexError` as an unusable GPU observation.
Executable reproduction: `python3 analysis/deployment-probes/20260909-experiment-checks/reviews/implementation/recheck.py`.

No implementation edits were made by this reviewer. The reported full-suite
46-pass result was supplied by the parent; this second pass independently ran
the focused reproduction scripts, not the full suite.

## Resolution

Re-ran only `recheck.py` after the final fix. It exited 0 and printed
`OOM preserved with unusable-GPU diagnostic`; its assertions verified both the
invalid verdict and the explicit unusable-sample diagnostic. This closes the
remaining empty-column finding. Verified `tools/experiment_checks.py` SHA-256:
`193fee68b5dc234e31c827ff7d69e1728ac7dee3cf71d388c21678c0dcb2da6f`.
No additional broad review or full-suite rerun was performed.

## Runtime schema exception verification

Inspected only the subsequent `unmeasured`/`unavailable` false-permission schema
exception against `PredictionFeedback.observe()` in the frozen contingent
runtime. It is warranted: no commits returns `unmeasured` with false permission
and no identity fields; an uncommitted prediction produces `unavailable` with
false permission and `record.get('model_sha256')`, which may be null. The checker
retains these observations without manufacturing a permission opportunity. The
exception requires permission to be exactly false. No new finding from this
targeted check. Final inspected checker SHA-256:
`8caedfd8ea5e1f62e5f83254b49a53c64a811a30f8e79352bb51990c10c4c979`.
The 49-pass full-suite result is parent-supplied; this step was source inspection.
