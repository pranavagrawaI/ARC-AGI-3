# 0003-g-reconstruction-perturbation-novita-bf16-v1
Question: Does one decision-relevant reconstruction change predictably change Gemma's G decision and reverse when restored?
Posture: bounded offline causal probe
Contrast: intact → perturbed → intact; perturbation makes the two discriminator outcomes identical, leaving IDs, order, prose, legal actions, and protected successor unchanged.
Prediction: both intact blocks return the frozen PROBE signature; perturbed returns ABSTAIN(insufficient_evidence, cycle_delta), twice each.
Measure: perturbation-summary.json pass plus all six perturbation rows in results.json.
Cap: 12 model requests, 36 HTTP attempts, USD 0.02, 15 minutes, 0 environment actions.
Stop: pass after sequence; fail after any signature mismatch; abort on audit/provider/model drift or cap.
Invalid if: request differs beyond recorded intervention, provider unhealthy/drifts, parsing fails, or sequence incomplete.
Result map: positive → proceed to E4B gate; negative → inspect decision dependence; ambiguous/invalid → repair measurement or retry with a new card.
Outputs: experiments/hypothesis/0003-temporal-game-judgment/runs/0003-g-reconstruction-perturbation-novita-bf16-v1-20260807T184300Z/
