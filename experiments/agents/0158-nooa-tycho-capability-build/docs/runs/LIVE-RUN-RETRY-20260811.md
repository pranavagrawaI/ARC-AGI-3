# 0158 LS20 retry — 2026-08-11

Run: `artifacts/live-ls20/20260811T215726Z-gemma31b-retry`  
Model: `google/gemma-4-31b-it`  
Card SHA-256: `83e890766f8711f7bc3cc15c0029aadee2399691a878aebb634b1277f325fb4c`

## Result

- fresh online LS20 reset: succeeded
- ARC actions: **0 / 44**
- levels completed: **0**
- provider calls: **2**, both HTTP 200
- reported input + output tokens: **18,837**
- provider-reported cost: **$0.00647525**
- accepted grounding claim: **none**
- scorecard: terminal and non-resumable

The first call inspected the current frame. Its printed 64x64 grid expanded the second request to 13,525 input tokens. Gemma then used all 4,095 available output tokens narrating the grid, reached `incomplete_details.reason=max_output_tokens`, and returned no tool call or `GroundingClaim`. NOOA raised `GenerationError`; no environment action was authorized.

The generic worker-error path recorded `substrate_failure`, but the provider crossings falsify that label: both calls completed successfully at HTTP 200. `classification-correction.json` records the evidence-backed post-run classification `model_generation_failure` without modifying the immutable packet.

All 17 packet checksums and all 20 candidate-source hashes verify. A scan of 55 run files found no configured credential value. The retry approval is consumed.
