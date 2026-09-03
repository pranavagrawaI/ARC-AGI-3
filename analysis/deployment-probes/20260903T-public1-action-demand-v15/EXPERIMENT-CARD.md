# public1-action-demand-v15

Question: Now that `local_solver.py` actually forwards the two action-demand knobs to the Solver
worker, will the exact Kaggle SGLang route honor a forced `take_actions` request that
disables thinking and caps output at 2,048 tokens, so the Solver reaches eight reconciled
public `ls20` actions without another no-action failure?

Posture: bounded transport and commitment qualifier on one public game; no gameplay-improvement
claim.

Contrast: V14 (Kaggle kernel 12, 2026-09-03) used the same model route, public game, official
launcher, one model-facing action primitive, action budget, call budget, and containment mode, and
the same intent for the final request. V14 did not test the question: the notebook set
`ARC_LLM_ACTION_DEMAND_MAX_TOKENS=2048` and `ARC_LLM_ACTION_DEMAND_DISABLE_THINKING=1`, but
`runtime/local_solver.py` rebuilt the worker subprocess environment from a hardcoded allowlist,
`WORKER_ENVIRONMENT_KEYS`, that omitted both keys, so the worker read a clean environment. The
forced request (Actor call 5) went out with `max_tokens=16000` and thinking enabled, consumed all
16,000 completion tokens as reasoning, emitted no tool call, and the launcher failed rc=71 — the
same non-commitment failure class as V13 (V13 call 9). V15 changes exactly three things and nothing
else in the route: (1) `WORKER_ENVIRONMENT_KEYS` in `runtime/local_solver.py` gains
`ARC_LLM_ACTION_DEMAND_MAX_TOKENS` and `ARC_LLM_ACTION_DEMAND_DISABLE_THINKING`
(SHA-256 `517572979803…` → `ba6268dcda6eee366498598dda930a8bac261cdddaa31e2e4afe11386498ec7f`,
re-pinned in the builder); (2) `public25_launcher_probe.summarize_solver_sessions` adds
`action_demand_max_output_tokens` and `action_demand_disable_thinking` to the expected decision
envelope so it reconciles against the values the v11 worker already records; (3) a test in
`deployment/kaggle/tests/test_local_solver.py` asserts `worker_environment()` preserves both keys.
The notebook env vars, the Solver wheel (wheelhouse v11, unchanged), exploratory-request tokens,
and every cap are identical to V14. Exploratory Actor requests remain at 16,000 tokens with no
thinking override.

Prediction: the worker's `worker-metadata.json` decision envelope records
`action_demand_max_output_tokens=2048` and `action_demand_disable_thinking=true`; every forced
action-demand checkpoint records only `take_actions`, `tool_choice=required`, `max_tokens=2048`,
and the exact no-thinking `extra_body` `{"chat_template_kwargs": {"enable_thinking": false}}`;
every exploratory checkpoint records `max_tokens=16000` and no such override; every required
response emits `take_actions`; and the run reaches eight reconciled actions or an earlier
WIN/GAME_OVER.

Measure: exact installed source hashes including the new `local_solver.py` hash; the worker
decision envelope in `worker-metadata.json`; decision-checkpoint request parameters and responses;
response reasoning-token counts; `llm-trace.jsonl`; `tool-trace.jsonl`; `action-batch-trace.jsonl`;
`solver_sessions` / `solver_sessions_error` in the outer `result.json`; official recording; SGLang
access log; outer `result.json`; per-sequence creation-to-action timing. Reconcile every released
action to an official successor.

Cap: one RTX PRO 6000 notebook; one public game (`ls20`); eight Solver actions; at most 13 Actor
calls and 14 successful chat-completion accesses including warmup; at most 96,384 Actor completion
tokens; 1,200 seconds startup plus 7,200 seconds launcher time; financial spend not applicable.

Stop: stop after eight reconciled actions or an earlier WIN/GAME_OVER; abort on source/hash drift,
competition-rerun mode, malformed action, missing successor reconciliation, any completed legacy
`take_action` call, required-request parameter drift, model-server error, decision timeout,
launcher timeout, or containment/lifecycle failure.

Invalid if: the v11 wheelhouse or installed source hashes differ from the frozen manifest and the
new pinned `local_solver.py` hash; the worker decision envelope does not record
`action_demand_max_output_tokens=2048` and `action_demand_disable_thinking=true`; any Actor request
offers `take_action`; the required no-thinking options are absent from a forced checkpoint; an
exploratory request receives the no-thinking override; the official launcher, adapter, model route,
public game, caps, or network setting differ from the frozen notebook; persisted Actor requests do
not reconcile one-for-one with completed provider responses; an action cannot be reconciled to the
official recording; output or immediate source bytes are not preserved.

Result map:
- Eight reconciled actions or an earlier terminal state with the request contract intact and the
  worker envelope showing 2,048 / no-thinking → qualify the bounded action-demand transport and
  inspect latency and gameplay before expanding to three public games.
- The worker envelope shows 2,048 / no-thinking but a required response still ends without
  `take_actions` and its checkpoint contains the exact options → no-thinking is ineffective on this
  route; inspect the provider response and choose a host-side deterministic fallback rather than
  raising the token cap.
- The worker envelope still shows `null` / `false` → the env-forwarding repair did not take on the
  real route; diagnose the surviving boundary (worker python, launcher re-exec, dataset copy) from
  the preserved checkpoints and issue a new card. Do not raise the token cap.
- Request-option drift, `solver_sessions` still failing to reconcile, or ambiguous evidence →
  repair the transport or measurement defect and issue a new card.
- Legacy action execution or an unreconciled release → reject the candidate.

Do not create a competition submission.

Outputs: local candidate `deployment/kaggle/s0-public1-action-demand-v15-candidate`; frozen card,
candidate hashes, and reconciliation under
`analysis/deployment-probes/20260903T-public1-action-demand-v15`; exact remote source and output
downloaded before interpretation.
