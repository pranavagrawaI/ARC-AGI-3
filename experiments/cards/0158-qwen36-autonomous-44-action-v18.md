# 0158 Qwen3.6 autonomous 44-action v18
Question: How does frozen Qwen behave over a longer LS20 run?
Contrast: ordinary grounding contradictions, exhausted alternatives, and G2 record/reset autonomously; no human mediation or new API.
Prediction: Qwen uses prior probe results to revise grounding and executes multiple prospective actions.
Measure: `levels_completed >= 1`; action count, verdicts, cycles, and G2 are diagnostics.
Cap: 44 ARC actions; 1200 provider attempts; 12 iterations/artifact; 4 errors/artifact; 5 HTTP 429 retries/call; 8192 output tokens/call; 15M input+output tokens; 4 active hours.
Stop: success; death/reset; malformed artifact; provider/environment failure; exhausted 429 retry; budget; 44 actions.
Invalid if: drift, bad IPC, missing evidence, human mediation, fallback, resumption, or model mismatch.
Result map: terminal failure -> preserve/review; success/cap -> preserve; never resume.
Input: public evidence only; no LS20 roles, mechanics, route, map, or semantic state.
