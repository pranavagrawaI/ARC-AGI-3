# Claim selection versus verification v4

Question: Can the same Qwen critic correctly verify individual claims when given
measured facts, and does failure appear when claims are grouped or extracted?
Posture: Saved-prefix diagnostic, no Actor execution.
Contrast: Two original narratives (S5I5 step3, CD82 step1). Six single-claim calls,
two three-claim list calls, two narrative extraction calls and two dependent
verification calls over the unedited extracted claims. Evidence and verifier prompt
match between verification conditions. Extraction has no measurements or labels.

Prediction: S5I5 exact-one is contradicted by 1,2,1; its current count4 is supported
within scope; the completion-threshold claim is unresolved. CD82 red count30 and
white-color bounds are supported in scope; whether the border change is a counter
or player indicator remains unresolved. DATA.json freezes these expectations and
exact original excerpts. Other extracted claims are manually assessed against facts.

Measure: Per-claim judgment, exact ID coverage, supplied evidence references, sound
reasoning, and evidence-grounded corrections. Check extraction retention of each
target proposition, particularly exactly-one; never equate a verdict label with
valid reasoning. Record unsupported added assertions, latency, tokens and cost.

Cap: Twelve calls, 1024 output tokens per call, 45-second HTTP timeout each,
zero retries, $0.25 total conservative reservation and actual-cost ceilings.
Use qwen/qwen3.8-27b with only coreweave/fp8, no fallback, temperature0 and thinking
disabled. No gameplay or Kaggle launch. Provider FP8 identity is routing evidence,
not verified equivalence of all Kaggle artifact bytes.

Stop: Complete the twelve calls or abort on transport/provider/reasoning/unfinished
generation failure, invalid JSON, invalid extraction shape or non-exact extraction.
Do not manually repair extraction, rerun calls or change prompts after results.
Semantic mistakes remain scored failures, not operational exceptions.

Invalid if: Future evidence or rubric leaks; source/prompt changes mid-run;
different facts between verification conditions; dropped/changed claims hidden;
unrecorded response; reasoning enabled; provider fallback. Budget is shared by
all twelve calls; dependent requests are reserved before admission.

Result map: Failure on explicit claims -> isolated verification remains inadequate,
do not expand orchestration. Explicit succeeds but list fails -> investigate coverage.
List succeeds but extraction loses targets -> extraction is the observed boundary.
All succeed -> capability qualified only on these development prefixes, next test
held-out claims and actual Actor uptake. Operational failure -> preserve partial
results, no unqualified cross-condition conclusion. No automatic follow-up run.

Outputs: This directory: DATA.json, PROMPTS.json, initial and dependent requests,
raw replies, timings, STARTED.json hashes, EXECUTION.json, grades and RESULT.md.
Fifteen offline tests passed before inference. Earlier v4 directory was an unrun
preparation; this run adds generic counts for changed rows so current-count claims
are actually measurable in the supplied packet.

Authorization: User explicitly requested implementation and testing of this
three-condition design; existing OpenRouter authorization applies.
