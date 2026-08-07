# 0003-e4b-reconstruction-disabled-control-v2
Question: Does authority-sensitive tool exposure yield safe E4B reconstruction dependence?
Posture: bounded offline matched control
Contrast: candidate-local G versus D; D exposes only ABSTAIN, with other evidence/runtime fixed.
Prediction: G passes 10/10; D emits stable ABSTAIN decisions 10/10.
Measure: `reconstruction-control-summary.json` and 20 rows in `results.json`.
Cap: 20 local requests; 60 HTTP attempts; $0; 30 min; 0 environment actions.
Stop: pass after both conditions; fail on enabled error or disabled non-abstention/instability; abort on drift or cap.
Invalid if: model hashes/audit mismatch, non-reconstruction evidence differs, parse fails, or sequence is incomplete.
Result map: pass → precommit one live action; fail → stop live promotion and deepen interface; invalid → retry.
Outputs: `experiments/hypothesis/0003-temporal-game-judgment/runs/0003-e4b-recon-control-v2-20260807T211303Z/`
