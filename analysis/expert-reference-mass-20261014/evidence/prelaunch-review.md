# Prelaunch review

Reviewed the complete new routing module, exact-wheel overlay, paired orchestrator, generated notebook and launch/collection code against user scope.

- Constructor routing ABI is supported by the exact wheel; the callback sees pre-remap logits, returns contiguous fp32 weights/int32 IDs, and existing padding/remap/capture postprocessing still runs.
- Correction uses top10-logit softmax (the global normalizer cancels), then selects top5. Uniform routing sums to0.5 rather than redistributing missing weight. Shared expert remains in native path; is_nextn bypass preserves draft routing.
- Control restores original wheel bytes; source and wheel are hash-bound. Candidate config changes only target active count.
- Parent and prior experiment files remain untouched. Decoded execution snippets carry c14/fail-fast; the literal anchor checker is run against the actual decoded snippets because escaped asset strings conceal quoted anchors.
- Native Kaggle timeout3600 plus absolute notebook containment3450 bounds the authorized single launch. Fixed equal900s windows, fresh benchmarks/recording dirs and process-group cleanup contain phase contamination.
- Tests verify weight oracle, extreme/tied/empty/dtype inputs, draft exclusion, drift rejection, top-level-await execution and actual process reap. Changed-input CUDA graph test is reserved for run GPU before model loading.
- Runtime/source tests are mandatory evidence; a successful notebook alone is not a positive result. Collected versions/files are pinned and rehashed.

Accepted limitations: one replicate, control-first order, short fresh-game exposure, stochastic generations; cannot establish equivalence or sustained hidden-score gain. No supported blocking defect remains after local checks. Full model/backend correctness remains a run gate.

Prelaunch failures preserved: first notebook build rejected a cell-id under parent schema; removed the new cell id. Second literal-anchor check missed an escaped fail-fast source string; now checks exact decoded execution snippets with a bound hash. No GPU run had started during these corrections.
