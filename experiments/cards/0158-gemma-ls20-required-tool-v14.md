# 0158 LS20 required-tool v14
Question: Can Gemma pass fresh LS20 level 1 when the sole tool is mandatory?
Contrast: v13 allowed tool_choice=auto; v14 requires execute_python on every provider turn.
Prediction: Gemma enters CodeAct, inspects frames(), and returns a typed claim/action or grounded failure.
Measure: public `levels_completed >= 1`; gates are diagnostic, not success.
Cap: 44 ARC actions; 200 provider crossings; 12 iterations/artifact; 4 errors/artifact; 5 HTTP 429 retries/call; 8192 output tokens/call; 5,000,000 reported input+output tokens; 90 active minutes; spend unknown.
Stop: success; consequential non-429 failure; exhausted retry; G2; cap; missing usage.
Invalid if: drift, bad IPC, missing evidence, mediation, fallback, or resumption.
Result map: success -> preserve; G2 -> review; semantic/model failure -> inspect; substrate failure -> repair; continuation requires a new run.
Input: public observations/actions only; no LS20 roles, mechanics, route, map, or semantic state.
Output: artifacts.
