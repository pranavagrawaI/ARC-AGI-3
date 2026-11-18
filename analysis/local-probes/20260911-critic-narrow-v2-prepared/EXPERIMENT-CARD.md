# Narrow epistemic critic regression v2

Question: Does the narrower evidence-to-claim critic correct the failures found
in the first critic while preserving its justified non-objections?

Posture: Local saved-trace regression test.

Contrast: The same six frozen user packets and inference settings as the v1
screen. Only the system instructions change from broad action criticism to the
narrow epistemic audit in `narrow-v2/system.md`.

Prediction: The critic reports the S5I5 exact-one contradiction before weaker
issues, reports its scope or objective overreach as the second issue, and does
not manufacture an issue for the cautious S5I5, LS20, or CD82 controls. For the
actual LS20 and authored CD82 overclaim, it identifies only errors supported by
the supplied evidence and introduces no unsupported facts in its correction.

Measure: Six raw response files, structured-format validation, provider/model
identity, reasoning-token count, elapsed time and usage cost. Manual grading
checks exact claim quotation, correct evidence, valid inference, minimal
correction, direct-contradiction priority, false objections and length limits.
Status alone is insufficient.

Cap: Six calls, 1024 output tokens each, 45-second HTTP timeout each, zero
retries, $0.25 total cost cap, zero environment actions and zero Kaggle launches.

Stop: Complete six calls or abort at the first transport, provider, reasoning,
cost, generation or response-format failure. Do not rerun automatically.

Invalid if: Any user packet or inference setting differs from v1; future
outcomes or grading rubric enter a request; reasoning is enabled; provider
fallback occurs; a response is incomplete or missing. OpenRouter advertises
CoreWeave FP8; exact Kaggle artifact equivalence remains unverified.

Result map: If the S5I5 direct contradiction is missed, reject this critic
prompt. If any cautious control receives a false material objection, reject it.
If both gates pass but other cases contain unsupported corrections, revise and
test held-out cases before integration. If all six pass semantically, qualify a
separate Actor-uptake test; do not claim gameplay efficacy.

Outputs: This directory: frozen requests, `PREPARED.json`, this card,
`STARTED.json`, raw responses, timings, `EXECUTION.json`, manual grades and
result report.
