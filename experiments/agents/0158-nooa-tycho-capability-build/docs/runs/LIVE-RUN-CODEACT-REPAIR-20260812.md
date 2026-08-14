# 0158 LS20 CodeAct-repair run — 2026-08-12

Run: `artifacts/live-ls20/20260812T112548Z-gemma31b-codeact-repair`  
Provider route: **`coreweave/bf16` only; fallback disabled**  
CodeAct limits: **12 iterations; 4 errors**  
429 policy: **initial attempt plus at most five audited retries**

## Result

- fresh online LS20 reset: succeeded
- ARC actions: **0 / 44**
- levels completed: **0**
- provider crossings: **12**, all successful on their first attempt
- HTTP 429 responses: **0**
- reported input + output tokens: **47,041**
- accepted grounding claim: **none**

Gemma used eight calls to inspect the initial frame. It then called `return_result` four times with `{}`. Each failed with exact feedback that the required `result` field was missing and instructed the model to call again with valid arguments. Gemma repeated the identical empty tool call until the four-error budget exhausted.

The expanded repair path therefore worked mechanically and was exercised, but Gemma did not act on its feedback. The generic worker-error path recorded `substrate_failure`; `classification-correction.json` records the evidence-backed class `model_tool_argument_repair_exhaustion`.

The scorecard is terminal and the approval is consumed. The failure packet's 17 checksums and all 20 executed candidate-source hashes verify. No additional run was opened.
