# 0158 Qwen contrastive action map v23
Question: Can Qwen learn a discriminative opaque LS20 action map and complete L1?
Posture: capability probe
Contrast: v22 plus reciprocal action-effect commitments and two-axis verdicts.
Prediction: common-mode change earns no specificity; by action 12, >=2 actions close a reciprocal contrast, then reuse it.
Measure: commitments, probe_results, verdicts, transitions. Acquisition = reciprocal contrast supported for >=2 actions; preservation = later matching effect after >=1 decision. Success = `levels_completed >= 1`.
Cap: 44 actions; 1200 calls; 12 iterations/artifact; 4 errors/artifact; 5 retries/call; 16384 output tokens/call; 15M tokens; 4h.
Stop: success; death/reset; malformed artifact; provider/environment failure; budget/cap.
Invalid if: drift, bad IPC/evidence, mediation, fallback, resumption, duplicate POST, mismatch.
Result map: pass -> retain; fail -> diagnose first link; ambiguous -> refine; never resume.
Outputs: artifacts/live-ls20/<run-id>
