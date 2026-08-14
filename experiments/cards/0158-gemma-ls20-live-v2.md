# 0158 Gemma LS20 live v2
Question: Can frozen 0158 get `google/gemma-4-31b-it` through fresh LS20 level 1?
Posture: feasibility and first-failure diagnosis.
Contrast: none.
Prediction: complete level 1 or yield a claim-anchored failure before 44 actions.
Measure: public `levels_completed >= 1`; gates are diagnostic, not success.
Cap: 44 ARC actions; 200 provider attempts; 4096 output tokens/call; 5,000,000 reported input+output tokens; 90 active minutes; spend unknown.
Stop: success; first consequential failure; G2 review boundary; cap; missing usage; uncertain transport.
Invalid if: card drift, stale/duplicate IPC, missing evidence, human mediation, or resumption.
Result map: success -> preserve and replicate; G2 -> review without resume; semantic failure -> inspect packet and change only a future run; substrate failure -> repair offline and start a new run.
Outputs: `artifacts/live-ls20/<run-id>/`.
Input: public observations/actions only; no LS20 roles, mechanics, route, map, tracker, or semantic state.
