# 0003-g-reconstruction-perturbation-novita-bf16-v1
Question: Does changing one reconstruction result change Gemma's G decision and reverse when restored?
Posture: bounded offline causal probe
Contrast: intact → perturbed → intact; make both discriminator outcomes identical, changing nothing else.
Prediction: intact returns frozen PROBE; perturbed returns ABSTAIN(insufficient_evidence, cycle_delta), twice each.
Measure: `perturbation-summary.json.pass` and six perturbation rows in `results.json`.
Cap: 12 model requests; 36 HTTP attempts; $0.02; 15 min; 0 environment actions.
Stop: pass after sequence; fail on mismatch; abort on audit/provider/model drift or cap.
Invalid if: unrecorded request changes, drift, parse failure, or incomplete sequence.
Result map: positive → E4B; negative → inspect dependence; invalid → repair/retry with new card.
Outputs: `experiments/hypothesis/0003-temporal-game-judgment/runs/0003-g-perturb-v1-20260807T184300Z/`
