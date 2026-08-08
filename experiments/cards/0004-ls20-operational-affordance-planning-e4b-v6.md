# 0004 LS20 operational affordance planning E4B v6

Question: Can E4B search a route once action-unit semantics and causal effects are explicit?
Posture: offline causal-planning gate
Contrast: V5 stalled by treating coordinate units as action counts; v6 states the general divide-by-delta rule and requires an immediate tool call. It adds no route.
Prediction: Valid plans cite the affordance but 0/3 complete because maze-aware search remains unresolved.
Measure: `raw-responses.json` plans and `responses.json` engine outcomes.
Cap: 3 local E4B calls; 261 offline engine actions; 0 live actions; $0; 10 minutes.
Stop: 3 calls or first transport/parser/checkpoint error.
Invalid if: future route/successor leakage, engine drift, missing raw response, or incomplete response.
Result map: 3/3 completion → perturb effect; valid/no completion → route search bottleneck; invalid → stop this interface.
Outputs: `experiments/hypothesis/0004-autonomous-acquisition/runs/<run-id>/`
