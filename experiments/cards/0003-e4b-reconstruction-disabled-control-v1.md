# 0003-e4b-reconstruction-disabled-control-v1
Question: Does reconstruction availability change E4B decisions on the widened batch?
Posture: bounded offline matched control
Contrast: candidate-local G versus D with the same images, chronology, numeric transitions, runtime, and five cases.
Prediction: G passes 10/10; D emits stable ABSTAIN decisions 10/10.
Measure: `reconstruction-control-summary.json` and 20 rows in `results.json`.
Cap: 20 local requests; 60 HTTP attempts; $0; 30 min; 0 environment actions.
Stop: pass after both conditions; fail on enabled error or disabled non-abstention/instability; abort on drift or cap.
Invalid if: model hashes/audit mismatch, non-reconstruction evidence differs, parse fails, or sequence is incomplete.
Result map: pass → precommit one live action; fail → localize interface dependence; invalid → retry with new card.
Outputs: `experiments/hypothesis/0003-temporal-game-judgment/runs/0003-e4b-recon-control-v1-20260807T210711Z/`
