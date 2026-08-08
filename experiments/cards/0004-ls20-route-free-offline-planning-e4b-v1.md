# 0004 LS20 route-free offline planning E4B v1

Question: Can E4B complete a level from the acquired objective without a supplied route?
Posture: offline planning gate
Contrast: prior ceiling supplied every action; this request supplies 4 prefix frames, action dynamics, tracked positions, and the acquired objective only.
Prediction: 0/3 submitted plans complete the level; planning remains the bottleneck.
Measure: `raw-responses.json` plans and `responses.json` real-engine completion outcomes.
Cap: 3 local E4B calls; 261 offline engine actions; 0 live actions; $0; 10 minutes.
Stop: 3 calls or first transport/parser/checkpoint error.
Invalid if: route/successor leakage, engine drift, missing raw response, or incomplete response.
Result map: 3/3 completions → closed-loop offline-from-reset gate; 1–2 → diagnose instability; 0 → inspect failure mechanics before widening; invalid → repair/new card.
Outputs: `experiments/hypothesis/0004-autonomous-acquisition/runs/<run-id>/`
