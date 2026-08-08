# 0004 LS20 compact affordance planning E4B v5

Question: Can E4B execute route search from explicit cells and replay-supported effects?
Posture: offline causal-planning gate
Contrast: V4 cleared provider formatting but new evidence names were absent from the tool enum; v5 closes them without changing the case.
Prediction: Plans cite the affordance but 0/3 complete due distance-count or maze errors.
Measure: `raw-responses.json` plans and `responses.json` engine outcomes.
Cap: 3 local E4B calls; 261 offline engine actions; 0 live actions; $0; 10 minutes.
Stop: 3 calls or first transport/parser/checkpoint error.
Invalid if: future route/successor leakage, engine drift, missing raw response, or incomplete response.
Result map: 3/3 completion → perturb effect content; valid/no completion → route search fails despite sufficient causal state; invalid → repair/new card.
Outputs: `experiments/hypothesis/0004-autonomous-acquisition/runs/<run-id>/`
