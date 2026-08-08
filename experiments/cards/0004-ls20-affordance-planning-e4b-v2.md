# 0004 LS20 affordance planning E4B v2

Question: Do replay-supported trigger effects make route-free E4B planning executable?
Posture: offline causal-planning discriminator
Contrast: V1 had objective and geometry; v2 adds one supported rotation affordance, its position, and required applications, but no route.
Prediction: 3/3 plans cite the affordance, visit its trigger twice, and complete the level.
Measure: `raw-responses.json` plans and `responses.json` real-engine completion outcomes.
Cap: 3 local E4B calls; 261 offline engine actions; 0 live actions; $0; 10 minutes.
Stop: 3 calls or first transport/parser/checkpoint error.
Invalid if: future route/successor leakage, engine drift, missing raw response, or incomplete response.
Result map: 3/3 completion → perturb affordance content; changed/no completion → geometry/search bottleneck; unchanged route → affordance ignored; invalid → repair/new card.
Outputs: `experiments/hypothesis/0004-autonomous-acquisition/runs/<run-id>/`
