# 0158 LS20 12-iteration run — 2026-08-11

Run: `artifacts/live-ls20/20260811T220724Z-gemma31b-12iter`  
CodeAct ceiling per artifact: **12**  
Maximum output tokens per call: **8,192**

## Result

- fresh online LS20 reset: succeeded
- ARC actions: **0 / 44**
- levels completed: **0**
- provider calls: **6**, all successful
- reported input + output tokens: **66,664**
- accepted grounding claim: **none**

The higher ceiling removed iteration exhaustion: Gemma completed a claim artifact on iteration 6. It selected the 2×5 region at row 45, column 34 and returned its two rows as numeric cell strings. The frozen host rejected those rows because `CellMask` requires a width-matched binary `#`/`.` encoding, so the run closed before action authority.

The host recorded `gemma_semantic_failure`, but that is not yet a clean model-capability result. The model-visible JSON schema typed `rows` only as strings, and the frozen model contract did not define the required `#`/`.` encoding. `classification-review.json` therefore preserves a live alternative: an underspecified interface caused or materially contributed to the invalid artifact.

The scorecard is terminal and the one-run approval is consumed. The failure packet's 17 checksums and all 20 executed candidate-source hashes verify.

## Between-run correction

The host's binary-mask invariant remains unchanged. The next-run candidate exposes the invariant in both the phase instruction and structured JSON schema: each row must match `^[#.]+$`, `#` means selected, `.` means unselected, and row dimensions must match the bounding box. An offline regression test reproduces and rejects the exact numeric-row artifact from this run. This correction does not alter or reopen the scored trajectory above.
