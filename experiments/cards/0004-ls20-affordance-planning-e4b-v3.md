# 0004 LS20 affordance planning E4B v3

Question: Do replay-supported trigger effects make route-free E4B planning executable?
Posture: retry after invalid transport
Contrast: V1 lacked effects; v3 repeats v2's supported rotation affordance, position, and two required applications, with no route. V2 was invalid on call 2.
Prediction: 3/3 plans cite the affordance, visit its trigger twice, and complete the level.
Measure: `raw-responses.json` plans and `responses.json` engine outcomes.
Cap: 3 local E4B calls; 261 offline engine actions; 0 live actions; $0; 10 minutes.
Stop: 3 calls or first transport/parser/checkpoint error.
Invalid if: future route/successor leakage, engine drift, missing raw response, or incomplete response.
Result map: 3/3 completion → perturb affordance; changed/no completion → geometry/search bottleneck; unchanged direct route → affordance ignored; invalid → stop and diagnose provider.
Outputs: `experiments/hypothesis/0004-autonomous-acquisition/runs/<run-id>/`
