# 0159 Muse native encoding LS20 v12
Question: Can Muse complete exhausted turns when commitment accepts its observed NOOA encoding?
Posture: bounded feasibility; diagnosis.
Contrast: v11 expects flat action fields; v12 specifies and unwraps `return_result({"result": Decision})`.
Prediction: observed wrapped commitments validate; authority stays with Muse.
Measure: summary; transcript; budget/commit/provider tool events and usage; raw recording.
Cap: 44 actions; one/turn; at most 14 calls/turn (11 analysis + 3 named commit attempts); 8,192 output tokens/call.
Stop: level completion, game over/reset, cap, exhausted/illegal commitment, provider/worker failure, or uncertain ARC action.
Invalid if: wrong model/provider/source; analysis tool exposed in commit; host-selected action; semantic action inference; mutable evidence; unenforced sandbox; resume; missing successor.
Result map: preserve all; fix interface failures only; add no game semantics or fallback.
Outputs: 0159 `artifacts/` run directory.
