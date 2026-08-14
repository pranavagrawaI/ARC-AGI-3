# 0158 LS20 complete-claim run — 2026-08-12

Run: `artifacts/live-ls20/20260812T084706Z-gemma31b-complete-claim`  
Provider route: **`coreweave/bf16` only; fallback disabled**  
CodeAct ceiling per artifact: **12**  
Maximum output tokens per call: **8,192**

## Result

- fresh online LS20 reset: succeeded
- ARC actions: **0 / 44**
- levels completed: **0**
- provider calls: **2**
- reported input + output tokens: **3,036**
- accepted grounding claim: **none**

The first CoreWeave call succeeded with 1,119 input and 1,917 output tokens. The second returned HTTP 429 with uncertain transport disposition, so the worker aborted and the host closed action authority. This run did not reach a claim artifact and provides no behavioral evidence about the complete claim specification or Gemma's LS20 capability.

The scorecard is terminal and the one-run approval is consumed. The failure packet's 17 checksums and all 20 executed candidate-source hashes verify. No retry was opened.
