# Tempo critic public run v1

Question: What operational and gameplay behavior does the current Actor plus
epistemic-critic solver demonstrate during one three-game public Kaggle run?

Posture: Descriptive public evaluation. This is one candidate run, not a causal
comparison and not a competition submission.

Candidate: `deployment/kaggle/submission-20260912-v10/submission.ipynb`, embedding
`deployment/kaggle/submission-runtime-20260912-v7`, contract
`b294519f23a9d9d33d9ee8e773a8db14fd3ddc84e21b862da8a1bcc9201da9b9`.
Kernel: `pranavagrawai/arc3-first-submission-20260910`; record the version returned
by the single push.

Prediction: The critic is invoked before an Actor call after two completed Actor
calls or two environment steps, sees the complete rendered Actor context, receives
no tools, and emits at most 256 tokens with reasoning off. Valid source-bound advice
is inserted once into the pending Actor request. The intervention may increase
action tempo and cause earlier belief revisions; level completion is unknown.

Measure: Preserve the notebook output, server log, qualification result and budget
ledgers. For each game report levels, actions, Actor calls, critic calls, valid and
invalid reviews, review verdicts, time to first action, actions/minute, call wall
time, prompt/cache/completion tokens, review-to-next-action delay, explicit Actor
acceptance or contest when observable, and objective progress. Inspect critic advice
for factual invention, overbroad claims, and unnecessary delay. Distinguish a pixel
change, prediction match, objective progress, and level completion.

Cap: One private public-only Kaggle notebook execution on NvidiaRtxPro6000, session
timeout 3600 seconds, at most one GPU hour. Run LS20, S5I5 and CD82 in that order.
Each game has a 600-second model-admission window and at most 200 Solver actions.
Actor output maximum 16384 tokens; critic output maximum 256 tokens. HTTP timeout
90 seconds, retries zero, Qwen3.8-27B-FP8, reasoning off, MTP off, summarizer off.
At most 20 critic reviews per game. Notebook work deadline 3100 seconds, hard outer
deadline 3300 seconds, model readiness 1200 seconds, launcher at most 2700 seconds
reduced by remaining work time. No competition submission, automatic retry, model
change, or monetary spend beyond existing Kaggle quota.

Stop: After the single kernel reaches COMPLETE, ERROR or CANCELLED. On exception,
preserve all retrievable evidence and diagnose the first failing boundary. Do not
launch another version under this card.

Invalid if: OOM; transport failure or timeout; failed, unresolved, late or unrecorded
Actor or critic request; deleted request state; cleanup failure; missing required
session/accounting evidence; source/runtime mismatch; shortened game allocation; cap
violation; thinking or retry enabled; or competition execution. Malformed or
unbound critic output is a measured critic failure, not an operational invalidator,
provided it is logged, withheld from the Actor, and request accounting remains exact.
Kaggle COMPLETE alone is insufficient.

Result map: A valid run with one or more levels completed demonstrates capability of
this combined candidate on those sessions, without attributing the gain to the
critic. A valid run with faster action production or useful early pivots supports
further controlled comparison. A valid run with zero levels remains useful if it
localizes whether delay, epistemic advice, Actor uptake, mechanics, planning or goal
inference binds. Invalid evidence receives no solver-performance interpretation.

Authorization: Pranav explicitly requested a Kaggle public run with this solver to
get an initial performance reading. This authorizes one public-only execution and no
competition submission or repeat.

Launch prerequisite: The exact frozen runtime suite and notebook entrypoint checks
pass; fresh authenticated kernel status is terminal; fresh quota reports at least
one GPU hour. Preserve those checks here. This card is immutable after the push.
