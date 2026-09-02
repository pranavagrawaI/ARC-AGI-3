# public1-action-demand-v14

Question: Will the exact Kaggle SGLang route honor a forced `take_actions` request that disables thinking and caps output at 2,048 tokens, so the Solver reaches eight reconciled public `ls20` actions without another no-action failure?

Posture: bounded transport and commitment qualifier on one public game; no gameplay-improvement claim.

Contrast: V13 used the same model route, public game, official launcher, one model-facing action primitive, action budget, call budget, and containment mode. Its exploratory and forced action-only requests both allowed 16,000 output tokens with thinking enabled; Actor call 9 spent all 16,000 tokens reasoning and emitted no tool. V14 changes the Solver wheel and only the final action-demand request: `extra_body.chat_template_kwargs.enable_thinking=false` and `max_tokens=2048`. Exploratory requests remain at 16,000 tokens with no thinking override.

Prediction: every forced action-demand checkpoint records only `take_actions`, `tool_choice=required`, `max_tokens=2048`, and the exact no-thinking `extra_body`; every exploratory checkpoint records `max_tokens=16000` and no such override; every required response emits `take_actions`; and the run reaches eight reconciled actions or an earlier WIN/GAME_OVER.

Measure: exact installed source hashes; decision-checkpoint request parameters and responses; response reasoning-token counts; `llm-trace.jsonl`; `tool-trace.jsonl`; `action-batch-trace.jsonl`; official recording; SGLang access log; outer `result.json`; per-sequence creation-to-action timing. Reconcile every released action to an official successor.

Cap: one RTX PRO 6000 notebook; one public game (`ls20`); eight Solver actions; at most 13 Actor calls and 14 successful chat-completion accesses including warmup; at most 96,384 Actor completion tokens; 1,200 seconds startup plus 7,200 seconds launcher time; financial spend not applicable.

Stop: stop after eight reconciled actions or an earlier WIN/GAME_OVER; abort on source/hash drift, competition-rerun mode, malformed action, missing successor reconciliation, any completed legacy `take_action` call, required-request parameter drift, model-server error, decision timeout, launcher timeout, or containment/lifecycle failure.

Invalid if: the v11 wheelhouse or installed source hashes differ; any Actor request offers `take_action`; the required no-thinking options are absent from a forced checkpoint; an exploratory request receives the no-thinking override; the official launcher, adapter, model route, public game, caps, or network setting differ from the frozen notebook; persisted Actor requests do not reconcile one-for-one with completed provider responses; an action cannot be reconciled to the official recording; output or immediate source bytes are not preserved.

Result map: eight reconciled actions or earlier terminal state with the request contract intact -> qualify the bounded action-demand transport and inspect latency/gameplay before expanding to three public games; a required response still ends without `take_actions` and its checkpoint contains the exact options -> no-thinking is ineffective on this route, so inspect the provider response and choose a host-side deterministic fallback rather than raising the token cap; request-option drift or ambiguous evidence -> repair the transport or measurement defect and issue a new card; legacy action execution or unreconciled release -> reject the candidate. Do not create a competition submission.

Outputs: local candidate `deployment/kaggle/s0-public1-action-demand-v14-candidate`; frozen card and reconciliation under `analysis/deployment-probes/20260902T-public1-action-demand-v14`; exact remote source and output downloaded before interpretation.
