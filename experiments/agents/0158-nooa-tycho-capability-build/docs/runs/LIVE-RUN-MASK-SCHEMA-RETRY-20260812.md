# 0158 LS20 mask-schema retry — 2026-08-12

Run: `artifacts/live-ls20/20260812T070434Z-gemma31b-mask-schema-retry`  
CodeAct ceiling per artifact: **12**  
Maximum output tokens per call: **8,192**

## Result

- fresh online LS20 reset: succeeded
- ARC actions: **0 / 44**
- levels completed: **0**
- provider calls: **4**
- reported input + output tokens: **4,400**
- accepted grounding claim: **none**

The first three provider calls succeeded. The fourth returned HTTP 429 with uncertain transport disposition, exactly matching the prior corrected-candidate attempt's stopping class and call position. The worker aborted and the host closed the run as `substrate_failure` before action authority. This remains transport evidence, not evidence for or against the mask-schema correction or Gemma's LS20 capability.

The scorecard is terminal and the one-run approval is consumed. The failure packet's 17 checksums and all 20 executed candidate-source hashes verify. No further run was opened.
