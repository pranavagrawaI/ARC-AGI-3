# 0158 LS20 CoreWeave run — 2026-08-12

Run: `artifacts/live-ls20/20260812T083524Z-gemma31b-coreweave`  
Provider route: **`coreweave/bf16` only; fallback disabled**  
CodeAct ceiling per artifact: **12**  
Maximum output tokens per call: **8,192**

## Result

- fresh online LS20 reset: succeeded
- ARC actions: **0 / 44**
- levels completed: **0**
- provider calls: **5**, all successful
- reported input + output tokens: **39,130**
- accepted grounding claim: **none**

The provider switch cleared the repeated ModelRun HTTP 429 boundary. Gemma correctly used the newly exposed `#`/`.` binary-mask encoding, but returned the same 2×42 mask as both `primary_region` and its sole alternative. The host rejected the artifact because primary and alternative masks must be distinct, then closed action authority.

The host recorded `gemma_semantic_failure`, but the distinctness rule was not present in the model-visible phase instruction or structured schema. `classification-review.json` therefore preserves interface underspecification as a live alternative rather than attributing the failure solely to Gemma.

The scorecard is terminal and the one-run approval is consumed. The failure packet's 17 checksums and all 20 executed candidate-source hashes verify.
