# 0158 DeepSeek V4 Flash CoreWeave preflight v30
Question: Can exact CoreWeave V4 Flash complete a real 0158 claim-commit loop?
Posture: qualification
Contrast: v27 official route was privacy-blocked; model and input fixed, route changes.
Prediction: valid bound claim and commitment return through execute_python with usage.
Measure: accepted artifacts, provider attempts, tool calls, usage, exact route.
Cap: 2 artifacts; 24 iterations; 8 errors; 144 provider attempts; 16384 output/call; 30 min; spend unknown.
Stop: accepted pair or first terminal schema/provider/budget failure.
Invalid if: fallback, wrong provider/model, changed historical input, or missing audit.
Result map: pass -> authorize matched live run; fail -> no ARC action and diagnose.
Outputs: artifacts/model-preflight/<run-id>.
