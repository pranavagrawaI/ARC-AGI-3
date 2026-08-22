# seq2-current-builder-report-gemma4-31b-20260821
Question: Does the current sequence-2 builder report change Gemma's immediate actor tool call?
Posture: matched causal shadow
Contrast: checkpoint 000036 exact request vs removal of only the current builder-report block.
Prediction: each arm is internally stable and first_tool_call differs.
Measure: result/report.json assessment.arm_signature_counts[first_tool_call], paired_change_rates, errors, usage.
Cap: 2 repeats/arm; 4 calls; 4096 output tokens/call; 900 s; $0.03 observed cost.
Stop: 4 calls or 900 s; preserve partial output on failure.
Invalid if: request/config/route drift, more than one changed leaf, any call error, or within-arm variability.
Result map: supported -> test builder budget quality; contradicted -> reduce this builder spend; inconclusive -> no live promotion.
Outputs: analysis/shadows/seq2-current-builder-report-gemma4-31b-20260821/{packet,result}
