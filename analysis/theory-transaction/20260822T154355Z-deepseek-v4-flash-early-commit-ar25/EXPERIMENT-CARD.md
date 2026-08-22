# DeepSeek V4 Flash early-commit gate
Question: Does latency-calibrated commit telemetry let DeepSeek exercise the theory transaction before timeout?
Posture: corrected matched zero-ARC engineering gate.
Contrast: 12-step control vs transaction; same 10/18 typed frontier, DeepSeek V4 Flash 0731, Sail Research FP4, reasoning, 4096 tokens, commit at <=180 s.
Prediction: treatment assesses an edit and persists no protected loss.
Measure: typed deltas, bytes, decision, ledger, notices, route, latency, cost.
Cap: 14 calls/arm, 28 total, 240 s/arm, $0.20; reserve $0.01/call; 0 ARC.
Stop: report/timeout, cap, missing cost, provider/tool error, or route drift.
Invalid if: commitments drift, treatment assesses no edit, route/cost unavailable, or boundary exceeded.
Result map: pass -> precommit short ARC prefix; regression/deadlock/invalid -> no ARC run.
Outputs: this directory.
