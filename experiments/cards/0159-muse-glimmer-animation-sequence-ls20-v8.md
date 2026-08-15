# 0159 Muse ARC frames LS20 v8
Question: Can 0159 run LS20 after correcting its frame contract?
Posture: bounded feasibility; diagnosis.
Contrast: v7 rejects multi-frame responses; v8 archives all and uses the final grid.
Prediction: animations no longer cause host failure; action authority stays with Muse.
Measure: summary levels/actions/stop/validity/hash; transcript decisions, budgets, results, animations; provider log; raw recording.
Cap: 44 actions; one/turn; at most 12 provider calls/turn (11 analysis + one `submit_action`); 8,192 output tokens/call.
Stop: level completion, game over/reset, cap, missing/illegal commitment, provider/worker/sandbox failure, or uncertain ARC action.
Invalid if: wrong model/provider/source; discarded animation; non-final decision grid; host-selected action; mutable evidence; unenforced sandbox; resume; missing successor.
Result map: preserve all; fix technical/interface failures only; add no game semantics or fallback.
Outputs: 0159 `artifacts/` run directory.
