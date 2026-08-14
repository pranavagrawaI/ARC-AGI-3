# 0158 DeepSeek V4 Flash CoreWeave matched 44-action v30
Question: How does V4 Flash act under frozen v26, and can it complete L1?
Posture: matched candidate diagnosis
Contrast: v28/v29; host, prompt, seed, caps, default reasoning matched; model/route differ. V26 is descriptive.
Prediction: V4 uses Translate and tests at least 3 action IDs; L1 remains uncertain.
Measure: levels; actions; commitments/verdicts; closures; register; calls/tokens/cost; traces.
Cap: 44 actions; 1200 calls; 12 iterations/artifact; 4 errors; 16384 output/call; 15M tokens; 4h; spend unknown.
Stop: L1; cap; G2; reset; malformed artifact; provider/environment/budget failure.
Invalid if: fallback, wrong model/provider, resumption, input/host drift, or missing usage.
Result map: L1 -> retain; stronger model/no L1 -> add objectives/constraints; no action map -> reject; invalid -> no comparison.
Outputs: artifacts/live-ls20/<run-id>.
