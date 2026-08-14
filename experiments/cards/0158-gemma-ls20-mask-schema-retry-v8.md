# 0158 LS20 mask-schema retry v8
Question: Can frozen 0158 get Gemma through fresh LS20 level 1 with visible binary-mask encoding?
Contrast: two attempts ended on call-four HTTP 429; candidate mechanism is unchanged.
Prediction: accepted claim/action, or the transport boundary reproduces.
Measure: public `levels_completed >= 1`; gates are diagnostic, not success.
Cap: 44 ARC actions; 200 provider attempts; 12 CodeAct iterations/artifact; 8192 output tokens/call; 5,000,000 reported input+output tokens; 90 active minutes; spend unknown.
Stop: success; first consequential failure; G2 review boundary; cap; missing usage; uncertain transport.
Invalid if: card drift, bad IPC, missing evidence, mediation, or resumption.
Result map: success -> preserve; G2 -> review; semantic failure -> inspect; substrate failure -> repair offline; continuation requires a new run.
Output: `artifacts/live-ls20/<run-id>/`.
Input: public observations/actions only; no LS20 roles, mechanics, route, map, or semantic state.
