# 0004 LS20 minimal affordance planning E4B v7

Question: Can E4B search a route from explicit dynamics and causal effects?
Posture: offline causal-planning gate
Contrast: V6 emitted an immediate plan but omitted a constant field whose only value was 1; v7 removes that non-decision field and changes no evidence.
Prediction: 0/3 valid plans complete because E4B ignores the trigger or maze constraints.
Measure: `raw-responses.json` plans and `responses.json` engine outcomes.
Cap: 3 local E4B calls; 261 offline engine actions; 0 live actions; $0; 10 minutes.
Stop: 3 calls or first transport/parser/checkpoint error.
Invalid if: future route/successor leakage, engine drift, missing raw response, or incomplete response.
Result map: 3/3 completion → perturb effect; valid/no completion → route search bottleneck; invalid → stop this interface.
Outputs: `experiments/hypothesis/0004-autonomous-acquisition/runs/<run-id>/`
