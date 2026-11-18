# Objection closure public run v2

Question: With the Actor's DONE affordance removed and the projected action count
removed from the critic packet (runtime v10), does the three-game public run
complete operationally, does the objection contract behave as specified across
all three games, and does the critic's wall-clock cost stay an order of
magnitude below v8's?

Posture: descriptive public evaluation of one candidate run; the corrected
retry of `20260912-objection-closure-public-v1`, which was invalidated on LS20
by an Actor DONE. Not a controlled comparison; no causal attribution.

Contrast: none. Historical references, both single runs on the same games and
budgets: v8 (`20260912-epistemic-questioner-public-v1`, valid, 0 levels) and v9
(`…-public-v1`, invalid, LS20 only). No behavioral resolution band exists for
this candidate.

Candidate: `deployment/kaggle/submission-20260912-v13/submission.ipynb`,
embedding `deployment/kaggle/submission-runtime-20260912-v10`, contract
`56e2e8218a1d3e564fe479d5c13b4cd58a8ff2b441499f81a477911a0313ebf2`, notebook
SHA-256 `e894e92f393115e2` (full digest in `PRELAUNCH.json`).
Kernel: `pranavagrawai/arc3-objection-closure-v2-20260912`; record the version
returned by the single push.

Prediction, operational (each stated so the run can refute it):
- All three games receive their 600-second admission window and each ends by
  `model_admission_budget`, `wall_budget`, `step_budget`, or `win` — never by a
  launcher exception.
- `done_refused.jsonl` is absent from every workspace; if present, the run
  continues past every refusal to a later executed action in the same game.
- Critic model calls consume under 8% of each game's session wall (v8:
  18.5-26.0%; v9 LS20: 1.3%), mean critic call under 8 seconds, critic prompt
  under 8,000 tokens at every review and under 15% of the concurrent Actor
  prompt.
- No critic model call while an objection is pending; every boundary reached
  while pending is `restated` with no matching critic request.
- Every objection ends `closed` or `expired`; at most one `expired` per game;
  no game ends with an objection pending unless it opened in the final call.
- No critic question states a number of remaining actions as a fact (checked
  by reading every question; the packet no longer supplies one).
- Actor calls per executed action are reported per game against v8 (LS20 3.7,
  S5I5 2.5, CD82 4.5) and v9-LS20 (5.4). No prediction: this is the measure the
  v9 run showed to be the binding tempo problem once the critic was cheap.
- No prediction about executed actions exceeding v8, and none about levels.

Measure: preserve notebook output, server log, qualification result, request
lifecycle, model calls, `critic_reviews.jsonl`, `objections.jsonl`,
`done_refused.jsonl` if any, action ledger, transitions, budget allocations,
`game-stops.jsonl`. Per game report stop reason, levels, executed and submitted
actions, Actor and critic calls, calls per executed action, critic wall share
and per-call wall, critic prompt tokens and ratio, review statuses, objections
opened, dispositions, reminders, recurrence rejections, probe accounting,
expiries, refused DONEs, and the Actor's next commitment after each
disposition. Print every question and every `closes` payload verbatim.

Cap: one private public-only Kaggle notebook execution on NvidiaRtxPro6000,
session timeout 3600 seconds, at most one GPU hour. LS20, S5I5, CD82 in that
order; 600-second admission and 200 Solver actions per game. Actor output max
16384 tokens; critic max 192. HTTP timeout 90 seconds, retries zero,
Qwen3.8-27B-FP8, reasoning off, MTP off, summarizer off. Notebook work deadline
3100 seconds, hard outer deadline 3300, model readiness 1200, launcher at most
2700 seconds. No competition submission, automatic retry, model change, or
additional execution under this card. Queue time is not capped and spends no
quota; a queued kernel is waited on, not re-pushed.

Stop: after the single kernel reaches COMPLETE, ERROR, or CANCELLED. On
exception preserve all retrievable evidence and diagnose the first failing
boundary. Do not launch another version under this card.

Invalid if: OOM; transport failure or timeout; failed, unresolved, late, or
unrecorded Actor or critic request; deleted request state; cleanup failure;
missing session or accounting evidence; source or runtime mismatch; shortened
game allocation; cap violation; reasoning or retry enabled; competition
execution; RTX PRO 6000 requested but another accelerator allocated. Malformed,
empty, or unbound critic output, a ledger-rejected `closes` payload, and a
refused DONE are measured events, not invalidators, when logged. Kaggle
COMPLETE alone is insufficient.

Result map:
- Valid, all operational predictions hold → the contract and its cost are
  live-qualified on three games; the permitted next step is a repeated-run card
  to estimate a behavioral resolution band before any v10-versus-v8 claim, and
  a separate diagnosis of Actor calls per action if it exceeds v8's.
- Valid, a contract prediction fails → localize the rule in `objections.jsonl`
  or `done_refused.jsonl`; fix under a new runtime and card; no promotion.
- Valid, critic wall share ≥ 8% → the restatement or packet path is not saving
  what v9-LS20 showed; diagnose before any further run.
- A refused DONE that the run survives → the defense worked; report the
  Actor's stated reason verbatim and whether it recurred.
- A completed level → capability of the combined candidate on that session; no
  attribution to the contract.
- Invalid → no solver performance interpretation; diagnose the first failing
  boundary.
- No result under this card authorizes competition submission or promotion.

Outputs: raw and derived evidence under
`analysis/deployment-probes/20260912-objection-closure-public-v2/`.

Authorization: Pranav said, "make a fresh card and push." The candidate passed
freeze-check, 175 runtime, 119 unit, 35 entrypoint, and 4 worker tests against
runtime v10 (`20260912-objection-closure-local-v2/VERIFIED.json`), and
`test_done_refusal.py` reproduces the v1 failure on v9 and passes on v10. This
authorizes this one public-only execution, conditional on the launch
prerequisite.

Launch prerequisite: immediately before the push, recheck the frozen runtime
and package hashes, the prior kernel's terminal status, and fresh GPU quota
above the one-hour cap; preserve them in `PRELAUNCH.json`. This card is
immutable after the push.
