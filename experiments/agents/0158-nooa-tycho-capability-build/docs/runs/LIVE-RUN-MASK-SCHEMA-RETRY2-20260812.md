# 0158 LS20 mask-schema retry 2 — 2026-08-12

Run: `artifacts/live-ls20/20260812T075931Z-gemma31b-mask-schema-retry2`  
CodeAct ceiling per artifact: **12**  
Maximum output tokens per call: **8,192**

## Result

- fresh online LS20 reset: succeeded
- ARC actions: **0 / 44**
- levels completed: **0**
- provider calls: **1**
- reported input + output tokens: **0**
- accepted grounding claim: **none**

The first provider call returned HTTP 429 with uncertain transport disposition. The worker aborted and the host closed the run as `substrate_failure` before action authority. The error type and redacted-message hashes exactly match the preceding two corrected-candidate failures, although those reached three successful calls before failing on call four.

This is the third consecutive corrected-candidate run stopped by the same transport condition. It provides no evidence for or against the mask-schema correction or Gemma's LS20 capability. Further live retries are blocked pending a transport change rather than another identical attempt.

The scorecard is terminal and the one-run approval is consumed. The failure packet's 17 checksums and all 20 executed candidate-source hashes verify.
