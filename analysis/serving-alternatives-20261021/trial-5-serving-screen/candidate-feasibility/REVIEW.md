# Review record

Local preparation only; no GPU/slot assignment. Coordinator independently reviews runner/build/launch and a separate read-only reviewer covers client/prepare. Native child independently reviewed main client/runtime code; prompt child reviewed the other-author setup/card and produced the exact tool-serialization audit.

Supported findings repaired with failing-before/passing-after evidence:

- Actual raw request bytes could be altered while a declared wire hash remained accepted. Reducer now hashes retained request.json and joins wire/header/attempt identities. Durable regression establishes baseline success before tampering.
- A tiny positive prompt count could falsely qualify long-input fit. Exact per-request normalized local expectations are now required; otherwise functional HTTP/tool successes retain a separate prompt-mismatch outcome.
- A hidden60s socket timeout clipped a declared280s request. Actual62s delayed-header fixture failed at60.1256s, then passed at62.0751s after parent-owned absolute cancellation replaced the I/O timeout.
- A declared329-byte response could deliver229 bytes of complete JSON and be treated as complete. Known remaining Content-Length is now checked independently of JSON parsing; truncated bytes remain preserved.
- Completed HTTP failures did not halt new admissions. Dispatcher now stops and reaps the other active child while retaining attempted/unattempted evidence.
- More subtly, immediate500 headers with a1s delayed error body let a fast200 sibling cause all six admissions before the failed child completed. Root reproduced offsets0,.004,.066,.127,.189,.251 on client8c86ac54. New exact regression failed (all six received,1.284s); parent now checks atomic live header receipts before replacements, latches admission stop, and retains the already-active error body's bytes within its original cap. The regression requires only initial cases0/1 received and exact error-body retention.
- Explicit launch_authorized:false in a matching fresh slot previously passed. It now requires literal true; regression failed before and passes after.

Latest affected client/launch suite:10 tests pass in6.381s, including the delayed-error-body regression. This is CPU loopback/temporary-artifact evidence, not engine performance. The real candidate command constructor also checks S10, context139264, graph40, MTP3/local argmax,58 images and explicit normalized template without loading runtime/model.

Runner: owner27 tests pass9.368s; native independent27 pass9.342s; root independently27 pass9.209s. One independent initial run had26 pass/1 fail: test_supervisor.SupervisorTests.test_trickle_response_has_absolute_deadline took20.537394675s for total_seconds8, exceeding8.3s assertion by12.237394675s. Isolated same fixture passed.66184s; unchanged full27 repeat passed9.430s. Failed temporary phase trace was deleted. Cause is unresolved, no invented fix or erased failure. See ../runner/TIMING-ANOMALY.md; separate5400s platform cutoff is mandatory and needs coordinator review.

Actual SDK request capture proves local serialization and zero network calls only. It does not prove future allocation, GPU fit, installed import/processor behavior or functional output. New executable/template/case bytes require refreshing packet/source digests and review. Prior trial preservation receipt checks39+24 committed files and56+23 inventoried files unchanged.
