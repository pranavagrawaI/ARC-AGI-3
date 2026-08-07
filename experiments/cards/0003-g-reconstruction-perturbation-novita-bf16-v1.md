# 0003-g-reconstruction-perturbation-novita-bf16-v1
Question: Does one reconstruction change predictably change Gemma's G decision and reverse when restored?
Posture: bounded offline causal probe
Contrast: intact → perturbed → intact; make both discriminator outcomes identical, changing nothing else.
Prediction: intact returns the frozen PROBE; perturbed returns ABSTAIN(insufficient_evidence, cycle_delta), twice each.
Measure: `perturbation-summary.json.pass` and all six perturbation rows in `results.json`.
Cap: 12 model requests; 36 HTTP attempts; $0.02; 15 minutes; 0 environment actions.
Stop: pass after sequence; fail on signature mismatch; abort on audit/provider/model drift or cap.
Invalid if: unrecorded request differences, provider drift, parse failure, or incomplete sequence.
Result map: positive → E4B gate; negative → inspect dependence; invalid → repair/retry under a new card.
Outputs: `experiments/hypothesis/0003-temporal-game-judgment/runs/0003-g-reconstruction-perturbation-novita-bf16-v1-20260807T184300Z/`
