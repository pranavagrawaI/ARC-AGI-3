# Saved-trace critic screen v1
Question: Does a fresh-context Qwen critic identify unsupported commitments while allowing justified exploration in saved Actor decisions?
Posture: Diagnostic inference screen, not architecture efficacy.
Contrast: Three preserved decisions (S5I5 step 3, LS20 step 7, CD82 step 1), each with an authored contrasting proposal on identical host evidence.
Prediction: S5I5 and LS20 original commitments challenged; CD82 original exploration allowed; corresponding contrasts receive opposite dispositions with grounded explanations.
Measure: Six raw response files, manual rubric judgments, provider/model identity, reasoning tokens, elapsed seconds and usage cost. Verdict alone is insufficient.
Cap: Six calls, 1024 output tokens each, 45-second HTTP timeout each, zero retries, $0.25 reservation cap, zero environment actions, zero Kaggle launches. Frozen requests conservatively reserve $0.03867852 at enforced price ceilings.
Stop: Complete six calls or abort on first transport/protocol/provider/reasoning/cost failure. Client timeout does not guarantee server-side cancellation.
Invalid if: Future outcomes or rubric leaked; prompt changed mid-run; reasoning enabled; provider fallback; incomplete generation; missing raw response. Provider advertises FP8; exact Kaggle artifact equivalence remains unverified and is not claimed.
Result map: All semantic checks pass -> test Actor uptake separately. Any semantic failure -> inspect concrete limitation before integration. Ambiguous -> manual adjudication, no promotion. Operational failure -> preserve invalid attempt, no automatic rerun.
Outputs: This directory; cases.json with provenance, separate rubric.json, numbered requests/responses/timings, STARTED.json and EXECUTION.json after execution, manual RESULT.md afterward.
Model: qwen/qwen3.8-27b, provider only coreweave/fp8, fallback disabled, thinking disabled, temperature zero. provider-catalog.json preserves advertised FP8 identity and pricing.
Authorization: Prepared pending user's answer about remote inference from this machine. No paid call has started; this card does not grant authorization.
