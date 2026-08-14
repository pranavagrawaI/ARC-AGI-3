# 0158 LS20 429-backoff run — 2026-08-12

Run: `artifacts/live-ls20/20260812T105920Z-gemma31b-429-backoff`  
Provider route: **`coreweave/bf16` only; fallback disabled**  
429 policy: **initial attempt plus at most five audited exponential-backoff retries**

## Result

- fresh online LS20 reset: succeeded
- ARC actions: **0 / 44**
- levels completed: **0**
- provider crossings: **6**, all successful on their first attempt
- HTTP 429 responses: **0**
- reported input + output tokens: **14,022**
- accepted grounding claim: **none**

The retry envelope was not exercised because transport stayed healthy. On call six Gemma returned `alternatives: []` and a binary primary mask, demonstrating uptake of the revised convention. Its bbox declared width 8, but both mask rows (`##.##.#`) had width 7. Pydantic rejected the artifact.

The generic worker-error path recorded `substrate_failure`; `classification-correction.json` records the evidence-backed class `model_schema_validation`. A configuration flaw also became visible: NOOA CodeAct had `max_retries=1`, which exhausts on the first validation error, so the intended in-artifact repair opportunity did not exist.

The scorecard is terminal and the approval is consumed. The failure packet's 17 checksums and all 20 executed candidate-source hashes verify. No additional run was opened because no 429 occurred.
