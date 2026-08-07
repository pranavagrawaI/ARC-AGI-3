# 0003-g-widened-prefix-batch-novita-bf16-v1
Question: Does retained candidate-local G remain reproducible across wider frontier/action coverage?
Posture: bounded offline transfer test
Contrast: five prefix-derived cases: original three plus left execution and down planning controls.
Prediction: all ten repeated decisions pass with one stable signature per case.
Measure: `ladder-summary.json` and ten rows in `results.json`.
Cap: 10 model requests; 30 HTTP attempts; $0.02; 20 min; 0 environment actions.
Stop: pass after G; fail on any score/signature mismatch; abort on drift or cap.
Invalid if: protected data enters requests, provider/model drifts, parse fails, or batch is incomplete.
Result map: pass → E4B transfer; fail → localize unsupported frontier/action; invalid → retry with new card.
Outputs: `experiments/hypothesis/0003-temporal-game-judgment/runs/0003-g-widened-v1-20260807T203156Z/`
