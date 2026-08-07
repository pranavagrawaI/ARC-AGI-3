# 0003-e4b-widened-transfer-v1
Question: Does the retained 31B G interface transfer reproducibly to official local E4B Q4_0?
Posture: bounded offline model-transfer test
Contrast: Novita 31B baseline then local E4B on the same five candidate-local G cases.
Prediction: both models pass all ten repeated decisions with stable signatures.
Measure: `transfer-summary.json` and 20 rows in `results.json`.
Cap: 20 model requests; 60 HTTP attempts; $0.02; 30 min after start; 0 environment actions.
Stop: pass after transfer; fail on any E4B mismatch; abort on drift or cap.
Invalid if: model hashes/audits mismatch, provider drifts, parse fails, or sequence is incomplete.
Result map: pass → reconstruction-disabled control; fail → deepen Machinery or reject E4B; invalid → retry with new card.
Outputs: `experiments/hypothesis/0003-temporal-game-judgment/runs/0003-e4b-transfer-v1-20260807T210033Z/`
