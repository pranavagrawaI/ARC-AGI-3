# 0003-g-reconstruction-perturbation-novita-bf16-v2
Question: Does the repaired G interface make Gemma's decision depend on reconstruction content and reverse when restored?
Posture: bounded offline causal probe
Contrast: intact → perturbed → intact; identical outcomes make the discriminator inadmissible and model-invisible.
Prediction: intact returns frozen PROBE; perturbed returns ABSTAIN(insufficient_evidence, cycle_delta), twice each.
Measure: `perturbation-summary.json.pass` and six perturbation rows in `results.json`.
Cap: 12 model requests; 36 HTTP attempts; $0.02; 15 min; 0 environment actions.
Stop: pass after sequence; fail on mismatch; abort on audit/provider/model drift or cap.
Invalid if: unrecorded request changes, drift, parse failure, or incomplete sequence.
Result map: positive → E4B gate; negative → inspect residual interface failure; invalid → retry with new card.
Outputs: `experiments/hypothesis/0003-temporal-game-judgment/runs/0003-g-perturb-v2-20260807T200116Z/`
