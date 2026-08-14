# 0158 Qwen3.6 semantic interface v17
Question: Can Qwen execute an LS20 action after a valid semantic claim and commitment?
Contrast: host owns IDs/schema; prompt explains harness rationale, lifecycle, functions, and examples.
Prediction: Qwen submits an accepted claim and anchored effect, causing one ARC action.
Measure: public `levels_completed >= 1`; actions/gates are diagnostics only.
Cap: 44 actions; 200 calls; 12 iterations/artifact; 4 errors/artifact; 5 HTTP 429 retries/call; 8192 output tokens/call; 5M input+output tokens; 90 active minutes.
Stop: failed provider qualification; success; consequential non-429 failure; exhausted retry; G2; cap; missing usage.
Invalid if: parity fails, drift, bad IPC, missing evidence, mediation, fallback, resumption, or model mismatch.
Result map: qualification fail -> no ARC; success -> preserve; G2/failure -> review; continuation requires a fresh run.
Input: public observations/actions only; no LS20 roles, mechanics, route, map, or semantic state.
