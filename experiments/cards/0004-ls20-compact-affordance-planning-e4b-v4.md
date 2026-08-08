# 0004 LS20 compact affordance planning E4B v4

Question: Can E4B plan when causal evidence is represented as explicit cells and effects?
Posture: offline interface discriminator
Contrast: V2/V3 nested requests failed provider formatting; v4 preserves the effect but names directions, trigger/target cells, contact semantics, and uses one map. No route is supplied.
Prediction: 3/3 plans use the trigger twice and complete the level.
Measure: `raw-responses.json` plans and `responses.json` engine outcomes.
Cap: 3 local E4B calls; 261 offline engine actions; 0 live actions; $0; 10 minutes.
Stop: 3 calls or first transport/parser/checkpoint error.
Invalid if: future route/successor leakage, engine drift, missing raw response, or incomplete response.
Result map: 3/3 completion → perturb effect content; valid/no completion → route search fails despite sufficient causal state; invalid → stop and diagnose provider.
Outputs: `experiments/hypothesis/0004-autonomous-acquisition/runs/<run-id>/`
