# 0158 LS20 mask-schema run — 2026-08-11

Run: `artifacts/live-ls20/20260811T222356Z-gemma31b-mask-schema`  
CodeAct ceiling per artifact: **12**  
Maximum output tokens per call: **8,192**

## Result

- fresh online LS20 reset: succeeded
- ARC actions: **0 / 44**
- levels completed: **0**
- provider calls: **4**
- reported input + output tokens: **4,408**
- accepted grounding claim: **none**

The first three provider calls succeeded. The fourth returned HTTP 429 with uncertain transport disposition, so the worker aborted and the host closed the run as `substrate_failure` before any action authority. This is not evidence for or against the mask-schema correction or Gemma's LS20 capability.

The scorecard is terminal and the one-run approval is consumed. The failure packet's 17 checksums and all 20 executed candidate-source hashes verify. Any retry must be a new run.
