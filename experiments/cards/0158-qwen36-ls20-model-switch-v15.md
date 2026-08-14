# 0158 Qwen3.6 LS20 model switch v15
Question: Can Qwen3.6-27B pass fresh LS20 level 1 through 0158?
Contrast: replace Gemma 4 31B with Qwen3.6-27B; retain auto tool choice and all host semantics.
Prediction: Qwen uses execute_python, returns valid claims, and reaches actions or grounded failure.
Measure: public `levels_completed >= 1`; gates are diagnostic, not success.
Cap: 44 ARC actions; 200 provider crossings; 12 iterations/artifact; 4 errors/artifact; 5 HTTP 429 retries/call; 8192 output tokens/call; 5,000,000 reported input+output tokens; 90 active minutes; spend unknown.
Stop: failed provider qualification; success; consequential non-429 failure; exhausted retry; G2; cap; missing usage.
Invalid if: drift, bad IPC, missing evidence, mediation, fallback, resumption, or model mismatch.
Result map: qualification fail -> no ARC run; success -> preserve; G2/failure -> review; continuation requires new run.
Input: public observations/actions only; no LS20 roles, mechanics, route, map, or semantic state.
