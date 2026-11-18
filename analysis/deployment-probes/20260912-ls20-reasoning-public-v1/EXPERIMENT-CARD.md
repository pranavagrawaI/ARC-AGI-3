# LS20 reasoning-on public run v1

Question: On the same LS20 game, runtime, model, 600-second admission window, and host contract as the reasoning-off v10 reference, does enabling Qwen3.8 thinking change the Actor's objective formation or false host-event claims enough to justify a reasoning-on solver despite lower action throughput?

Posture: one descriptive public-only contrast against the single historical v10 LS20 run; not a causal estimate, score claim, promotion, or competition submission.

Contrast: candidate runtime v11 with reasoning enabled versus historical runtime v10 reasoning off (`20260912-objection-closure-public-v2`, one run: 80 Actor calls, 6 executed actions, 13.3 calls/action, zero terminal-class predictions, 16 false `Context was evicted` reorientations). V11's gameplay behavior is otherwise v10; its only source change preserves provider-bound request messages and rendered tool results.

Candidate: `deployment/kaggle/submission-20260912-v14/submission.ipynb`, embedding runtime v11 contract `864838a6acde0530e4e54d6cd74b607ad2046c35dcbdff3b15a222f6ad836c2a`, notebook SHA-256 `ace1c4f33719764e2cb473ae3fac1a91f01aecb18d56885897b86266ac07e877`. Kernel `pranavagrawai/arc3-ls20-reasoning-v1-20260912`; record the version returned by the single push.

Prediction:
- Thinking is actually active: at least one Actor call has nonzero reasoning tokens and nonempty reasoning content; otherwise the contrast is invalid.
- The Actor makes at least one `level_up` or `terminal` predicted-path commitment. V10 made zero in six LS20 submissions and zero across about ninety submissions in seven game-runs.
- The Actor makes zero false host-event claims matching `Context was evicted`, `Budget exhausted (100/100)`, or an assertion that its model/action budget is exhausted while the preserved request contains no corresponding host surface.
- Slower reasoning reduces throughput to 2–4 executed actions, roughly half v10 LS20's 6; report Actor calls per executed action regardless.
- The run reaches the full 600-second model-admission boundary and ends by `model_admission_budget`, unless it reaches WIN/GAME_OVER first. No prediction that it completes a level.

Measure: preserve notebook output, server log, qualification result, request lifecycle, model_calls with request_messages/tool_results, actions, predictions, transitions, rejections, critic reviews, objections, DONE refusals, budget records, and system/GPU evidence. `audit.py` reports validity, reasoning presence/tokens, levels, executed actions, Actor/critic calls, calls/action, terminal-class predictions, false-host-event phrases in Actor outputs, prompt/tool-result evidence presence, stop reason, and request lifecycle reconciliation. Any apparent host-event claim must be checked against that call's preserved request before classification.

Cap: one private public-only Kaggle notebook execution on NvidiaRtxPro6000; one push and no retry; session timeout 3600 seconds; at most 1 GPU hour. One game, LS20, 600-second model-admission window, 200 Solver-action cap, Actor output max 8192 tokens, critic max 192, HTTP timeout 90 seconds, retries zero, Qwen3.8-27B-FP8, reasoning on, MTP off, summarizer off. Notebook work deadline 3100 seconds, hard outer deadline 3300, model readiness 1200, launcher at most 2700 seconds. Queue time is uncapped and spends no GPU quota. No competition execution, submission, model change, automatic retry, second kernel version, or promotion.

Stop: after the single pushed kernel reaches COMPLETE, ERROR, or CANCELLED. A queued kernel is waited on, never re-pushed. On failure preserve every retrievable artifact and diagnose the first failing boundary; do not launch another version under this card.

Invalid if: reasoning is not observably active; OOM; transport failure or timeout; failed, unresolved, late, or unrecorded Actor/critic request; missing request_messages/tool_results on any model call; deleted request state; cleanup failure; missing accounting or GPU evidence; source/runtime/notebook mismatch; shortened admission allocation; cap violation; competition execution; or RTX PRO 6000 requested but another accelerator allocated. A malformed critic output, rejected submit, refused DONE, zero level, or false host-event claim is a measured result, not an invalidator, when fully logged.

Result map:
- Valid, terminal-class prediction appears and no false host event appears → reasoning-on is operationally promising for objective formation; repeat under a paired card before changing the incumbent.
- Valid, no terminal-class prediction and no false event → reasoning cost did not move the objective-formation gate; keep reasoning off and pursue an objective-formation intervention.
- Valid, false host event appears → inspect its exact preserved request/tool results and diagnose the triggering surface before any solver-policy change.
- Valid, throughput falls below 2 actions or calls/action exceeds v10's 13.3 without improving objective formation → reject reasoning-on for this solver.
- Completed level → capability of the combined candidate in this one session only; no attribution or promotion.
- Invalid → no solver-performance interpretation; diagnose without retry.

Outputs: raw and derived evidence under `analysis/deployment-probes/20260912-ls20-reasoning-public-v1/`.

Authorization: Pranav said, `okay let's see what we can do then, do a kaggle run.` This authorizes the one bounded public-only execution above, conditional on the launch prerequisite. It does not authorize a competition submission or retry.

Launch prerequisite: immediately before the push, recheck runtime freeze and package hashes, local qualification, prior target-kernel absence or terminal status, current GPU quota above the one-hour cap, exact private/public-only metadata, and card hash; preserve them in `PRELAUNCH.json`. This card is immutable after push.
