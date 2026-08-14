# 0158 Qwen primary-relative cells v24
Question: Does the relative cell API close commitments and let Qwen learn an LS20 action map sufficient for L1?
Posture: probe
Contrast: v23 plus only relative `cell_after` and variadic `predict_cells`; verifier/model/budgets frozen.
Prediction: first commitment closes without schema repair and executes; by action 12, >=2 actions close reciprocal contrast.
Measure: commitments, provider trace, transitions, probe_results, scorecard. API pass = commitment then successor. Success = `levels_completed >= 1`.
Cap: 44 actions; 1200 calls; 12 iterations/artifact; 4 errors/artifact; 5 retries/call; 16384 output tokens/call; 15M tokens; 4h; spend unknown.
Stop: success; G2; death/reset; malformed artifact; provider/environment failure; budget/cap.
Invalid if: drift, bad IPC/evidence, fallback, resumption, duplicate POST, mismatch.
Result map: API pass -> judge learning; API fail -> diagnose first link; L1 -> retain; no L1 -> no success claim.
Outputs: artifacts/live-ls20/<run-id>
