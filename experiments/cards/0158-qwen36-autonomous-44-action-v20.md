# 0158 Qwen3.6 autonomous 44-action v20
Question: How does frozen Qwen behave in a longer LS20 run?
Contrast: v19 semantic interface plus 60s ARC read timeout; zero environment POST retries.
Prediction: Qwen revises from probe history and reaches multiple prospective actions without a 10s false stop.
Measure: `levels_completed >= 1`; actions, verdicts, cycles, and G2 are diagnostics.
Cap: 44 actions; 1200 provider attempts; 12 iterations/artifact; 4 errors/artifact; 5 HTTP 429 retries/call; 16384 output tokens/call; 15M input+output tokens; 4 active hours.
Stop: success; death/reset; malformed artifact; provider/environment failure; exhausted 429 retry; budget; 44 actions.
Invalid if: drift, bad IPC, missing evidence, human mediation, fallback, resumption, duplicate environment POST, or model mismatch.
Result map: failure -> preserve/review; success/cap -> preserve; never resume.
Input: public evidence only; no LS20 roles, mechanics, route, map, or semantic state.
