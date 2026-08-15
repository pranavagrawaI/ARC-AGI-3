# 0159 Muse native commit LS20 v11
Question: Can Muse complete exhausted turns when commitment keeps NOOA's established function name?
Posture: bounded feasibility; diagnosis.
Contrast: v10 switches from `return_result` to `submit_action`; v11 keeps `return_result` while removing Python.
Prediction: convention continuity reduces malformed commitments; authority stays with Muse.
Measure: summary; transcript; budget/commit/provider tool events and usage; raw recording.
Cap: 44 actions; one/turn; at most 14 calls/turn (11 analysis + 3 named commit attempts); 8,192 output tokens/call.
Stop: level completion, game over/reset, cap, exhausted/illegal commitment, provider/worker failure, or uncertain ARC action.
Invalid if: wrong model/provider/source; analysis tool exposed in commit; host-selected action; mutable evidence; unenforced sandbox; resume; missing successor.
Result map: preserve all; fix interface failures only; add no game semantics or fallback.
Outputs: 0159 `artifacts/` run directory.
