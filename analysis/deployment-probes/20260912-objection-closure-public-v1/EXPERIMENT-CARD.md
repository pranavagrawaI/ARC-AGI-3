# Objection closure public run v1

Question: Under the same public-only three-game regime as the 2026-09-12
epistemic-questioner run, does runtime v9 return the wall clock the v8 critic
consumed, and does the objection contract behave as specified — one pending
objection, local restatement instead of repeat reviews, dispositions carried on
commitments, retired actions not resuming silently, no deadlock?

Posture: descriptive public evaluation of one candidate run. This is the first
contact of the objection contract with live play. It is not a controlled
comparison and cannot attribute gameplay outcomes to the contract.

Contrast: none. The v8 run (`20260912-epistemic-questioner-public-v1`, one
execution) is the reference for operational metrics on the same games and
budgets; it is historical and uncontrolled. A behavioral resolution band for
this candidate has not been estimated.

Candidate: `deployment/kaggle/submission-20260912-v12/submission.ipynb`,
embedding `deployment/kaggle/submission-runtime-20260912-v9`, contract
`875abf0bb2b1746b98ac95e20d3d39ed14be1fa0210713187161c72ced951fad`, notebook
SHA-256 `227b1eb016af6154d25ef91ca6b14c41fba886e92d9ce1c59866071030fec899`.
Kernel: `pranavagrawai/arc3-objection-closure-20260912`; record the version
returned by the single push.

Prediction, operational (each stated so the run can refute it):
- Critic model calls consume under 8% of each game's session wall (v8:
  18.5-26.0%), and mean critic call wall is under 8 seconds (v8: 18.2s).
- Critic prompt tokens per call stay under 8,000 at every review (v8 grew to
  116,851) and never exceed 15% of the concurrent Actor prompt.
- No critic model call is admitted while an objection is pending; every review
  boundary reached while pending is recorded as `restated` in
  `critic_reviews.jsonl` with no matching `request_purpose: critic` row.
- Every objection that opens ends in `objections.jsonl` as `closed` (with a
  disposition) or `expired`; no game ends with a pending objection unless it
  opened within the final Actor call.
- No game stalls: every game reaches an executed action after any objection
  opens, and `expired` events are at most one per game.
- Executed actions per game exceed v8's on at least two of three games (v8:
  LS20 17, S5I5 18, CD82 12), because the returned wall clock is spent on
  actions. This is the one gameplay-adjacent prediction; it is about tempo,
  not levels.
- No prediction is made about levels completed.

Measure: preserve notebook output, server log, qualification result, request
lifecycle, model calls, `critic_reviews.jsonl`, `objections.jsonl`, action
ledger, transitions, and budget allocations. Per game report levels, executed
and submitted actions, Actor and critic calls, critic wall share and per-call
wall, critic prompt tokens and ratio to Actor prompt, review statuses
(completed / invalid / restated), objections opened, dispositions, reminders
before disposition, recurrence rejections, probe budgets and their accounting,
expiries, and the Actor's next commitment after each disposition. Inspect every
nonempty question and every `closes` payload verbatim. Keep pixel change,
prediction agreement, objective progress, and level completion separate.

Cap: one private public-only Kaggle notebook execution on NvidiaRtxPro6000,
session timeout 3600 seconds, at most one GPU hour. LS20, S5I5, CD82 in that
order; 600-second model-admission window and 200 Solver actions per game. Actor
output max 16384 tokens; critic output max 192. HTTP timeout 90 seconds, retries
zero, Qwen3.8-27B-FP8, reasoning off, MTP off, summarizer off. Notebook work
deadline 3100 seconds, hard outer deadline 3300, model readiness 1200, launcher
at most 2700 seconds. No competition submission, automatic retry, model change,
or additional execution under this card.

Stop: after the single kernel reaches COMPLETE, ERROR, or CANCELLED. On
exception preserve all retrievable evidence and diagnose the first failing
boundary. Do not launch another version under this card.

Invalid if: OOM; transport failure or timeout; failed, unresolved, late, or
unrecorded Actor or critic request; deleted request state; cleanup failure;
missing session or accounting evidence; source or runtime mismatch; shortened
game allocation; cap violation; reasoning or retry enabled; competition
execution. Malformed, empty, or unbound critic output is a measured critic
failure, not an invalidator, when logged and withheld. An Actor `closes` payload
rejected by the ledger is a measured contract event, not an invalidator. Kaggle
COMPLETE alone is insufficient.

Result map:
- Operational predictions hold and the contract events are as specified →
  the contract is live-qualified; the permitted next step is a repeated-run
  card to estimate a behavioral resolution band before any v9-versus-v8 claim.
- Critic wall share under 8% but a contract prediction fails (a call while
  pending, an undisposed objection, a stall, more than one expiry) → localize
  the failing rule in `objections.jsonl`; fix under a new runtime and card; no
  promotion.
- Critic wall share at or above 8% → the packet or restatement path is not
  saving what the trace arithmetic predicted; diagnose before any further run.
- Executed actions do not exceed v8 on two games despite the wall returned →
  the wall went somewhere else; locate it in `model_calls.jsonl` before
  attributing anything to the contract.
- A completed level → capability of the combined candidate on that session;
  no attribution to the contract.
- Invalid evidence → no solver performance interpretation.
- No result under this card authorizes competition submission or promotion.

Outputs: raw and derived evidence under
`analysis/deployment-probes/20260912-objection-closure-public-v1/`.

Authorization: Pranav said, "we should keep 3, next we need to do a kaggle run."
The candidate passed freeze-check, 118 unit, 35 entrypoint, and 4 worker tests
against runtime v9 (`20260912-objection-closure-local-v1/VERIFIED.json`). This
authorizes this one public-only execution, conditional on the launch
prerequisite.

Launch prerequisite: immediately before the push, recheck the frozen runtime
and package hashes, the prior kernel's terminal status, and fresh GPU quota
above the one-hour cap; preserve them in `PRELAUNCH.json`. This card is
immutable after the push.
