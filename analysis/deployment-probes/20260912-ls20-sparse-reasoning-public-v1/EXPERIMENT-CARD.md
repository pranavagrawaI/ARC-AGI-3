# LS20 sparse-reasoning public run v1

Question: With routine Actor and critic requests explicitly reasoning-off, can runtime v11 retain the reasoning-off LS20 throughput while allowing its existing `deep_investigate` activation to use at most one bounded reasoning call only if the Actor selects it for a hard abstraction gap?

Posture: one descriptive public-only run; not a causal estimate, score claim, promotion, competition submission, or retry of the closed reasoning-on card.

Contrast: candidate runtime v11, routine reasoning off, optional one-call sparse activation, versus historical runtime v10 reasoning-off LS20 (`20260912-objection-closure-public-v2`: 88 completed calls, 6 actions, zero levels) and the invalid global-reasoning-on v11 run (`20260912-ls20-reasoning-public-v1`: 6 completed calls, zero actions, seventh-call timeout).

Prediction:
- Every routine Actor/critic `model_calls.jsonl` record has zero reasoning tokens and no reasoning content.
- The run submits at least 5 actions before the 600-second model-admission boundary.
- The Actor invokes `deep_investigate` zero or one time. If invoked, its record has at most 2048 completion tokens, nonzero reasoning evidence, one concrete question, and one to eight evidence references.
- No model request times out; lifecycle reconciles.
- No prediction that LS20 is completed.

Measure: preserve qualification result; exact request messages and tool results; request lifecycle; actions, transitions, predictions and checks; `deep_investigations.jsonl` and `deep_outcomes.jsonl` if created; budgets; model/server/system/GPU evidence; teardown evidence; notebook log and external status. `audit.py` reports validity, routine reasoning state, sparse activation count and tokens, calls, actions, levels, stop reason, timeouts, and lifecycle reconciliation.

Cap: one private public-only Kaggle notebook version on NvidiaRtxPro6000; one push and no retry; kernel timeout 3600 seconds; at most 1.5 Kaggle-accounted GPU hours. One LS20 game; 600-second model-admission window; 200 Solver-action cap; routine Actor output max 8192 tokens; one optional deep call max 2048 tokens; critic max 192; HTTP timeout 90 seconds; retries zero; Qwen3.8-27B-FP8; MTP off; summarizer off. Notebook work deadline 3100 seconds, hard outer deadline 3300, model readiness 1200, launcher at most 2700 seconds. No competition execution, submission, second kernel version, automatic retry, model change, or promotion.

Stop: after the single pushed kernel reaches COMPLETE, ERROR, or CANCELLED. A queued kernel is waited on, never re-pushed. On failure preserve every retrievable artifact and diagnose the first failing boundary.

Invalid if: routine reasoning appears; more than one deep investigation; a deep call exceeds 2048 completion tokens; OOM; transport failure or timeout; failed, unresolved, late, or unrecorded request; missing request_messages/tool_results on a routine model call; deleted request state; cleanup failure; missing accounting/GPU evidence; source/runtime/notebook mismatch; shortened admission allocation; cap violation; competition execution; or RTX PRO 6000 requested but another accelerator allocated. Zero level, zero sparse activation, malformed critic output, rejected submit, or refused DONE is measured behavior rather than invalidity when fully logged.

Result map:
- Valid with at least 5 actions and no deep activation: keep routine reasoning off; the activation remains available but untested live.
- Valid with at least 5 actions and one deep activation: inspect its question, result, and subsequent actions before deciding whether sparse reasoning earns a paired repeat.
- Valid below 5 actions without timeout: diagnose solver/tool loop rather than reasoning latency.
- Any routine reasoning, timeout, or excess sparse call: reject the candidate configuration and diagnose without retry.
- Completed level: capability of this combined candidate in one session only; no attribution or promotion.
- Invalid: no solver-performance interpretation; preserve evidence and stop.

Outputs: `analysis/deployment-probes/20260912-ls20-sparse-reasoning-public-v1/`.

Authorization: Pranav — `you will do another run without the reasoning, btw I thought we had a reasoning activation (which activated reasoning for only some calls)`. This authorizes the one bounded public-only execution above with routine reasoning disabled and the existing sparse activation retained. It does not authorize a competition submission or retry.

Launch prerequisite: immediately before push, recheck runtime freeze and package hashes, local qualification, target-kernel absence, current GPU quota above the 1.5-hour cap, exact private/public-only metadata, and card hash; preserve in `PRELAUNCH.json`. This card is immutable after push.
