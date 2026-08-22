# Transactional Builder provider replay 3
Question: At AR25 10/18 typed exact (legacy 8/16), does the bounded transaction prevent obligation loss live?
Posture: matched zero-ARC gate after a latency-invalid replay.
Contrast: 12-step control vs transaction; same frontier, Gemma 4 31B with reasoning, exact OpenInference BF16 route, task, 4096 tokens.
Prediction: transaction assesses an edit and persists no protected loss.
Measure: typed obligation deltas, bytes, decision, ledger, notes/report, route/cost.
Cap: 14 calls/arm, 28 total, 600 s/arm, $0.42; reserve $0.015/call; 0 ARC.
Stop: report/timeout, cap, missing cost, provider error, or route drift.
Invalid if: commitments drift, no treatment edit is assessed, route/cost unavailable, or boundary exceeded.
Result map: pass -> one AR25 treatment, max 30 actions; regression/deadlock/invalid -> no ARC run.
Outputs: this directory.
