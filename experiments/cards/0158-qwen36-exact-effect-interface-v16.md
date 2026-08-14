# 0158 Qwen3.6 exact effect interface v16
Question: Can Qwen pass LS20 level 1 with one shared effect contract?
Contrast: replace arbitrary predicted_effect with an exact union and constructors.
Prediction: Qwen returns an accepted commitment and reaches action/verdict evidence.
Measure: public `levels_completed >= 1`; gates are diagnostic, not success.
Cap: 44 ARC actions; 200 provider crossings; 12 iterations/artifact; 4 errors/artifact; 5 HTTP 429 retries/call; 8192 output tokens/call; 5,000,000 reported input+output tokens; 90 active minutes; spend unknown.
Stop: failed two-phase provider qualification; success; consequential non-429 failure; exhausted retry; G2; cap; missing usage.
Invalid if: parity fails, drift, bad IPC, missing evidence, mediation, fallback, resumption, or model mismatch.
Result map: qualification fail -> no ARC; success -> preserve; G2/failure -> review; continuation requires new run.
Input: public observations/actions only; no LS20 roles, mechanics, route, map, or semantic state.
