# Resource-aware epistemic revision diagnostic
Question: Does self-checking the critic revision improve warranted corrections, and does explaining resource implications add value?
Posture: Development diagnostic.
Contrast: Framework base / base plus revision check / revision check plus shared resource explanation; four identical authored cases per arm, cyclic order, one sample each.
Prediction: Revision check removes circular threshold advice. Resource explanation improves deadline-aware inquiry without converting scarcity into evidence of falsity.
Measure: Grade raw responses against CASES.json, separating assessment, revision validity, unsupported premises, restraint, schema delivery, latency, tokens and cost.
Cap: 12 calls, $0.25 total conservative reservation, 480 seconds admission window, 30 seconds per HTTP call, 384 generated tokens per call, zero gameplay actions. Existing Qwen3.8-27B/CoreWeave FP8 endpoint, reasoning off.
Stop: All calls complete or transport/provider/finish/accounting failure; no retries or relaunch. Preserve semantic output failures and continue.
Invalid if: Frozen input drift, wrong model/provider, diagnosis leaked into input, unaccounted requests. Semantic failures do not become no_concern.
Result map: Improved corrections -> fresh-case or Actor uptake test; unchanged -> no incremental benefit demonstrated; harmful/ambiguous -> no promotion, preserve exact counterexamples. No gameplay efficacy or generalization claim from authored development cases.
Outputs: This directory, exact requests/responses, frozen inputs, execution ledger, manual result report.
