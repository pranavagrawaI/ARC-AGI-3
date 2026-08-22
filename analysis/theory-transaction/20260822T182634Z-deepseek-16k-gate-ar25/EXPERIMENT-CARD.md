# DeepSeek 16k transactional gate
Question: Did the 4096-token ceiling prevent DeepSeek from submitting a typed-assessed theory candidate?
Posture: single-arm zero-ARC feasibility replay at commit e5b162b; no treatment-effect claim.
Contrast: prior 4096 vs current 16384 tokens; same 10/18 AR25 frontier, DeepSeek V4 Flash 0731, Sail FP4, reasoning, transaction.
Prediction: DeepSeek submits world_model.py; candidate_assessed occurs; obligations persist by acceptance or restore.
Measure: finish reasons, tokens, tool disposition, typed delta, bytes, ledger, route, latency, cost.
Cap: 14 calls, 1800 s, $0.10; reserve $0.01/call; 0 ARC.
Stop: completion/timeout, cap, missing cost, provider/tool error, or route drift.
Invalid if: commitment drift, route/cost missing, no candidate_assessed, or boundary exceeded.
Result map: pass -> matched replay; no candidate -> revise evidence/commit contract; regression -> inspect restore; invalid -> no ARC.
Outputs: this directory.
