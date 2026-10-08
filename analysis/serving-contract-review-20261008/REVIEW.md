# Independent serving repair review

Scope: `analysis/serving-headroom-v2-20261008`. One separate author owns implementation and eventual launch; this reviewer neither launches nor edits the shared candidate. Shared board thread504 records ownership and findings, events567–577.

The original headroom run remains immutable and invalid for its intended comparison. Repair qualification uses the incumbent max10/Mamba60/memory0.96 configuration; it makes no16-request superiority claim.

## Evidence

Command:

```bash
uv run --no-project --with prometheus-client==0.26.0 python analysis/serving-contract-review-20261008/test_contract.py
```

Nine independent contract tests execute unchanged, hash-checked methods from pinned SGLang d00d88efc8d6281b12be4f4073126aec95038c55, with real Prometheus counters and export. GPU execution is not simulated as evidence: trace fixtures check the validator only.

- Native report_decode_stats adds batch size plus accepted draft tokens to the decode mode every iteration, before its periodic log early return.
- Across40 native decode calls with10 requests and20 accepted drafts each, realtime decode reaches1200 while finished-generation tokens remain0. Only calling observe_one_finished_request changes the latter to4096.
- Native prefill compute/cache increments stay out of the decode numerator.
- Duplicate scheduler decode series are rejected.
- The recorded v1 requested16/effective12 receipt is rejected by the admission guard.
- A successful profile HTTP response alone cannot satisfy trace qualification. Missing or CPU-only traces fail.
- The profile trigger remains reachable with stable occupancy even if the measured throughput is zero.

Initial test run:6/8 pass; two real defects found. A reset or missing decode sample halfway through a40-second block could still yield valid=True by salvaging another interval. The implementation owner fixed both. Intermediate run7/8 and final8/8 logs are retained. before-source-hashes.json records the candidate read during initial review (captured immediately after that run).

Source review also established that native labelled retraction counters are created only on the first actual retraction. Missing before initialization means zero under this pinned source; disappearance after observation must invalidate accounting. Blindly requiring a nonempty counter before any retraction would introduce another false failure.

## Final local review

All9 native/Prometheus contract tests and2 independent raw-audit boundary tests pass. The candidate author also reports15 passing implementation/audit tests. reviewed-source.json binds the reviewed candidate files and native contract.

The raw-audit review reproduced another false PASS: the old fixture omitted all start/end scrapes and all profile-response receipts, and recorded2900 realtime tokens against40960 wire tokens for N10. audit-before.log retains the failure. The revised audit requires exact endpoint counts, recomputes finished-generation and realtime decode deltas from verified raw scrapes, checks both directions of token disagreement, checks retractions and summary agreement, and requires the successful profile response plus an actual CUDA trace. audit-final.log confirms that removing endpoints or the profile response from a positively qualified fixture now fails.

An additional native-method test shows that `_resolve_spec_v2_tokens` includes acceptance lengths from an already-finished request when calculating the aggregate draft count, despite skipping its new KV commitment. Therefore a proposed absolute4-tokens-per-request discrepancy bound was not justified under overlap. The owner replaced it before freeze with a declared2% tolerance on4096-token blocks, recording actual differences without correcting throughput; warmup decode differences are diagnostic only. This tolerance is an operational qualification criterion, not a physical theorem.

No remaining supported local-review findings. Actual CUDA capture, native occupancy, cache fit and token reconciliation still require the separately bounded GPU qualification. No local test establishes a serving speedup or full GPU correctness. The implementation/launch owner remains codex-serving-headroom-resume; this review session has no launch path.
