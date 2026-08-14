# 0158 Qwen3.6 autonomous 44-action v19
Question: How does frozen Qwen behave in a longer LS20 run?
Contrast: current full frame only, exact probe history, <=3 exploratory calls, then submit; 16384 output tokens.
Prediction: Qwen revises after contradictions and executes multiple prospective actions.
Measure: `levels_completed >= 1`; actions, verdicts, cycles, and G2 are diagnostics.
Cap: 44 actions; 1200 provider attempts; 12 iterations/artifact; 4 errors/artifact; 5 HTTP 429 retries/call; 16384 output tokens/call; 15M input+output tokens; 4 active hours.
Stop: success; death/reset; malformed artifact; provider/environment failure; exhausted 429 retry; budget; 44 actions.
Invalid if: drift, bad IPC, missing evidence, human mediation, fallback, resumption, or model mismatch.
Result map: failure -> preserve/review; success/cap -> preserve; never resume.
Input: public evidence only; no LS20 roles, mechanics, route, map, or semantic state.
