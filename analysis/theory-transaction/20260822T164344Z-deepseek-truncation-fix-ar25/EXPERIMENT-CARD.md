# DeepSeek truncation-handling gate
Question: Did the Builder, not DeepSeek, lose the semantic edit by treating token truncation as completed construction?
Posture: matched zero-ARC regression replay at commit 07f6c56.
Contrast: control vs transaction; same 10/18 frontier, DeepSeek V4 Flash 0731, Sail FP4, reasoning, 4096 tokens, 12 steps, commit at <=180 s.
Prediction: after a length-truncated reply, construction continues; treatment reaches and assesses a semantic edit without preserving a protected loss.
Measure: finish reasons, next request phase, typed deltas, bytes, ledger, route, latency, cost.
Cap: 14 calls/arm, 28 total, 900 s/arm, $0.20; reserve $0.01/call; 0 ARC.
Stop: report/timeout, cap, missing cost, provider/tool error, or route drift.
Invalid if: commitments drift, no treatment edit is assessed, route/cost unavailable, or boundary exceeded.
Result map: pass -> short ARC prefix; regression/deadlock/invalid -> no ARC.
Outputs: this directory.
