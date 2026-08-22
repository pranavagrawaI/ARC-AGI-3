# Offered-tool enforcement regression
Question: After an unoffered remembered probe is rejected, will DeepSeek commit an assessed theory edit?
Posture: single transactional zero-ARC regression at commit e06cebb; no treatment-effect claim.
Setup: preserved 10/18 AR25 frontier; DeepSeek V4 Flash 0731; Sail FP4; reasoning; 4096 tokens; 12 steps; 900 s.
Mechanism: truncation continues; commit advertises mutation/checkout only; dispatch rejects every unoffered tool.
Prediction: stale run_python is rejected; a later permitted edit is typed-assessed; incumbent keeps all obligations by acceptance or restore.
Measure: offered/chosen tools, rejection, candidate delta, bytes, ledger, route, latency, cost.
Cap: 14 calls, $0.10; reserve $0.01/call; 0 ARC.
Stop: report/timeout, cap, missing cost, provider/tool error, or route drift.
Invalid if: commitment drift, no candidate_assessed event, route/cost missing, or boundary exceeded.
Result map: pass -> matched replay; otherwise revise before ARC.
Outputs: this directory.
