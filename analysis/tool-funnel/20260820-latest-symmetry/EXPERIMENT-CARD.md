# 20260820-latest-symmetry
Question: Does the latest ar25 symmetry run survive evidence audit and bounded saved-model replay?
Posture: forensic diagnosis
Contrast: none
Prediction: Audit rejects absent checkpoints; retrospective replay, if runnable, finds a mechanics discrepancy or material unknown coverage.
Measure: audit status/violations; atlas provenance; replay parity, shortest failure, canonical accuracy, claimed-cell errors, unknown coverage.
Cap: 0 model calls; 0 ARC attempts; 5 packets; 10 min; $0.
Stop: abort on tool error; stop after audit if mining is inadmissible; otherwise after 5-packet replay.
Invalid if: source mutates, prefix parity fails, or the final model is attributed to an earlier decision.
Result map: audit fail -> repair evidence before spend; audit pass + replay fail -> repair model; both pass -> shadow preflight; ambiguous -> preserve packet and redesign probe.
Outputs: analysis/tool-funnel/20260820-latest-symmetry/
