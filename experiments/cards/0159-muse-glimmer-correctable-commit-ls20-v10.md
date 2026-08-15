# 0159 Muse correctable commit LS20 v10
Question: Can Muse continue when an exhausted-turn commitment first calls an unavailable tool?
Posture: bounded feasibility; diagnosis.
Contrast: v9 permits one commit attempt; v10 permits up to three action-only validation attempts.
Prediction: invalid historical tool calls receive feedback and Muse corrects to `submit_action`; authority stays with Muse.
Measure: summary; transcript; budget/commit/provider tool events and usage; raw recording.
Cap: 44 actions; one/turn; at most 14 calls/turn (11 analysis + 3 named commit attempts); 8,192 output tokens/call.
Stop: level completion, game over/reset, cap, exhausted/illegal commitment, provider/worker failure, or uncertain ARC action.
Invalid if: wrong model/provider/source; analysis tool exposed in commit; host-selected action; mutable evidence; unenforced sandbox; resume; missing successor.
Result map: preserve all; fix interface failures only; add no game semantics or fallback.
Outputs: 0159 `artifacts/` run directory.
