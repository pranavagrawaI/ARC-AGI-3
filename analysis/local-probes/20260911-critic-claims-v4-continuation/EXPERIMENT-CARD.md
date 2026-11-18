# Claim screen continuation: remaining eleven requests

Question: How do explicit, listed and extracted claim verification differ on the
same two saved prefixes? This continues the user-authorized twelve-request test.

Posture: Diagnostic continuation following a transport-format stop.
Prior attempt: ../20260911-critic-claims-v4-run1 completed one model request, then
stopped because its otherwise JSON response was fenced in markdown. That raw reply
is retained byte-for-byte and scored as the first result, including its semantic
error. It will not be requested again. The old card and outputs remain unchanged.

Change: Parser accepts plain JSON or a single json/anonymous code fence. No model
prompt, input, parameter or judgment is edited. No leading narrative is stripped.
Sixteen offline checks pass, including exact semantic preservation of fenced JSON.

Contrast: Original run1 design: six individual-claim calls, two claim-list calls,
two narrative-only extraction calls and two dependent verification calls. One of
the single calls is imported; eleven new calls remain. Extracted claims are passed
unchanged to verification. Evidence and verification instructions match conditions.

Prediction and measure: Frozen expected values in DATA.json are identical to run1.
S5I5 exact-one=contradicted, current-four=supported_in_scope, threshold=unresolved.
CD82 red-count and white bounds=supported_in_scope, border-role=unresolved. Require
complete claim coverage and sound reasons, not just statuses. Assess extracted
claim coverage and additional judgments manually. Track format and semantic errors.

Cap: Eleven new calls; twelve total including one preserved result; cumulative
$0.25 reservation and actual-cost cap including that prior call. 1024 output tokens
per call, 45-second HTTP timeout, zero request retries, temperature0, thinking off,
same qwen/qwen3.8-27b and coreweave/fp8-only route. No gameplay or Kaggle.

Stop: Complete remaining calls or abort on transport/provider/reasoning/unfinished
generation/invalid JSON or extraction failure. Semantic mistakes are recorded.
No model output repair or repeat of any completed request. A client timeout does
not certify cancellation of provider-side computation.

Invalid if: Prior response altered; initial inputs or model settings changed;
rubric or future data supplied; extraction manually filled in; invalidity concealed.
This parser change is recorded and does not erase the first attempt's format failure.

Result map: Single-claim mistakes demonstrate unresolved verification limitations.
If singles pass but list fails, coverage becomes the priority. If list passes and
extraction loses targets, extraction is implicated. All pass only qualifies held-out
and Actor-uptake tests. Do not claim general reliability from development examples.

Outputs: STARTED.json links the prior attempt; EXECUTION.json separates reused calls;
all twelve requests/replies and raw timings, DATA.json, PROMPTS.json, grades and
RESULT.md are retained here. No additional requests beyond the original twelve.
