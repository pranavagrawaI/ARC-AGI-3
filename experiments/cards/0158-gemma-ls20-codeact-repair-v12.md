# 0158 LS20 CodeAct-repair v12
Question: Can 0158 get Gemma through fresh LS20 level 1 with bounded in-artifact schema repair?
Contrast: prior run; only intended change is CodeAct error budget 1 -> 4.
Prediction: schema feedback yields accepted claim/action, or a grounded failure.
Measure: public `levels_completed >= 1`; gates are diagnostic, not success.
Cap: 44 ARC actions; 200 provider crossings; 12 iterations/artifact; 4 errors/artifact; 5 HTTP 429 retries/call; 8192 output tokens/call; 5,000,000 reported input+output tokens; 90 active minutes; spend unknown.
Stop: success; consequential non-429 failure; exhausted retry; G2; cap; missing usage.
Invalid if: drift, bad IPC, missing evidence, mediation, fallback, or resumption.
Result map: success -> preserve; G2 -> review; semantic failure -> inspect; substrate failure -> repair; continuation requires a new run.
Output: `artifacts/live-ls20/`.
Input: public observations/actions only; no LS20 roles, mechanics, route, map, or semantic state.
