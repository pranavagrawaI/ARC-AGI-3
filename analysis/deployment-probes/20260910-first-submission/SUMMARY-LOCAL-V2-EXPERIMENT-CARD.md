# Background-summary fixture schema repair
Question: Does disabling background summarization remove its transport under an actually triggered, schema-valid fixture?
Posture: Qualification with diagnostic comparison.
Contrast: Original summary-enabled runtime versus no-summary runtime; otherwise same real direct entrypoint. New runtime also runs gateway and payload/budget tests.
Prediction: Original makes 10 Actor requests plus at least one summarizer request; new runtime makes exactly 10 Actor requests and no summarizer requests for 10 actions.
Measure: requests.json and model_calls.jsonl by purpose/schema, RESULT.json, stderr. Prior original fixture returned non-JSON for a strict JSON-schema response, retried and timed out: invalid comparison, preserved. Prior corrected suite passed 9 tests but did not exercise that response branch.
Cap: One original direct-entrypoint test and one corrected 9-test suite; 180s per subprocess, at most 40 successful controlled responses per entrypoint and 10 executed actions; one further rejected request stops serving if unexpectedly over cap. No GPU or external model calls.
Stop: Test completion or timeout; no automatic retry.
Invalid if: Wrong schema/runtime, premature unrelated failure, missing evidence, orphan worker or external model call.
Result map: Triggered original and corrected pass permit new public qualification; otherwise diagnose. No comparative solver-quality or long-context claim.
Outputs: summary-local-original-v2/ and summary-local-corrected-v2/ under analysis/deployment-probes/20260910-first-submission/.
