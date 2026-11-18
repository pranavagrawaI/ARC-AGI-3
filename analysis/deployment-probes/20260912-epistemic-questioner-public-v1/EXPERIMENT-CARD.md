# Epistemic questioner public run v1

Question: What operational, epistemic, and gameplay behavior does the event-driven
Actor plus epistemic-questioner solver demonstrate during one three-game public
Kaggle run?

Posture: Descriptive public evaluation. This is one candidate run, without a
control arm, causal attribution, or competition submission.

Contrast: none. Historical v7 results may provide context but cannot identify the
effect of this candidate.

Candidate: `deployment/kaggle/submission-20260912-v11/submission.ipynb`, embedding
`deployment/kaggle/submission-runtime-20260912-v8`, contract
`8532c0a7d18fbbf7d4fa4afd583a868cbfe347d03977053533864d2cb9e105d9` and notebook
SHA-256 `276c0434267e4848a65e3e33a4b06eb19c8b4482a067da1280e27134d47ca4b4`.
Kernel: `pranavagrawai/arc3-epistemic-questioner-20260912`; record the version
returned by the single push.

Prediction: Reviews occur only on level advance, new prediction failure, observed
multi-action sequence, three observed actions, or eight Actor calls since the last
valid review. Triggers coalesce. The critic has no tools, reasoning is off, output
is at most 192 tokens, and a nonempty question must bind to an ActorEvidence event.
Malformed output leaves the boundary pending. Questions should challenge scope,
warrant, objective proxies, alternatives, falsifiability, plan choice, or failure
to seek evidence; whether the Actor pivots usefully is unknown.

Measure: Preserve the notebook output, server log, qualification result, request
lifecycle, model calls, critic reviews, action ledger, and budget allocations. For
each game report levels, actions, Actor and critic calls, trigger distribution,
valid/invalid/empty questions, question type, time to first action, actions/minute,
call wall time, tokens/cache, repeated actions, Actor response to questions, and
objective progress. Inspect every nonempty question and its next Actor continuation.
Keep pixel change, prediction agreement, objective progress, and level completion
separate.

Cap: One private public-only Kaggle notebook execution on NvidiaRtxPro6000, session
timeout 3600 seconds, and at most one GPU hour. Run LS20, S5I5, and CD82 in that
order. Each game has a 600-second model-admission window and at most 200 Solver
actions. Actor output maximum 16384 tokens; critic output maximum 192 tokens. HTTP
timeout 90 seconds, retries zero, Qwen3.8-27B-FP8, reasoning off, MTP off, summarizer
off. Notebook work deadline 3100 seconds, hard outer deadline 3300 seconds, model
readiness 1200 seconds, launcher at most 2700 seconds reduced by remaining work time.
No competition submission, automatic retry, model change, or additional execution
under this card.

Stop: After the single kernel reaches COMPLETE, ERROR, or CANCELLED. On exception,
preserve all retrievable evidence and diagnose the first failing boundary. Do not
launch another version under this card.

Invalid if: OOM; transport failure or timeout; failed, unresolved, late, or
unrecorded Actor/critic request; deleted request state; cleanup failure; missing
required session/accounting evidence; source/runtime mismatch; shortened game
allocation; cap violation; thinking or retry enabled; or competition execution.
Malformed, empty, generic, misleading, or unbound critic output is a measured critic
failure rather than an operational invalidator when logged and safely withheld as
applicable. Kaggle COMPLETE alone is insufficient.

Result map: A valid run with a completed level demonstrates capability of the
combined candidate on that session without attributing it to the critic. Earlier
grounded questions followed by narrower beliefs or discriminating actions support a
controlled comparison. Questions that reinforce unsupported objectives, cause
delay, or are ignored localize critic/uptake failure and block promotion. Zero levels
still localizes the earliest binding boundary. Invalid evidence receives no solver
performance interpretation. No result under this card authorizes competition
submission or automatic promotion.

Outputs: raw and derived evidence under
`analysis/deployment-probes/20260912-epistemic-questioner-public-v1/`.

Authorization: Pranav said, “if we ready to launch a kaggle notebook then do it.”
The candidate passed its frozen local and exact entrypoint gates, the prior kernel is
terminal, and quota exceeds the numeric cap. This authorizes this one public-only
execution.

Launch prerequisite: Recheck the frozen runtime and package hashes, authenticated
terminal status, and fresh quota immediately before the push. Preserve them in
`PRELAUNCH.json`. This card is immutable after the push.
