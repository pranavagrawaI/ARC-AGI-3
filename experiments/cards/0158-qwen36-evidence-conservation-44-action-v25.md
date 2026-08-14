# 0158 Qwen evidence conservation v25
Question: Do residual accounting and explicit revision prevent proxy narrowing and enable LS20 L1?
Posture: probe
Contrast: v24 plus only inside/outside change counts, a probe-bound retain/revise/abandon update, and an evidence-conservation instruction; model, transport, action API, verifier frozen.
Prediction: after a contradicted claim with outside change, the next claim revises/abandons; by action 12 its primary covers action-varying movement and >=2 actions close reciprocal contrast.
Measure: updates, accounting, claims, commitments, transitions. Success = `levels_completed >= 1`.
Cap: 44 actions; 1200 calls; 12 iterations/artifact; 4 errors; 5 retries/call; 16384 tokens/call; 15M tokens; 4h.
Stop: success; G2; death/reset; malformed artifact; provider/environment failure; cap.
Invalid if: drift, bad evidence, fallback, resumption, mismatch.
Result map: region shift -> test map; no shift -> model-duty failure; L1 -> retain; no L1 -> no success claim.
