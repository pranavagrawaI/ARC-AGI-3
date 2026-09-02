# public1-singular-action-v13

Question: When Qwen has exactly one model-facing action primitive, `take_actions`, will every required commitment request produce a valid one-or-more-action batch instead of exhausting its 16,000-token response without acting?

Posture: bounded interface and lifecycle qualifier on one public game; no gameplay-improvement claim.

Contrast: V12 offered both `take_action` and `take_actions`. It executed two singular actions, never called the batch primitive, and then failed when the second successor request spent all 16,000 completion tokens without a tool call. V13 keeps the public game, model-server route, action budget, call budget, token ceilings, official launcher, and containment mode fixed. It changes the Solver wheel and removes `take_action` from every model-facing request. A one-action commitment is represented as a one-element `take_actions` batch.

Prediction: every accepted environment action is released from `action-batch-trace.jsonl`; every Actor request exposes exactly one model-facing action primitive; no completed `take_action` event exists; and the run reaches eight reconciled actions or an earlier WIN/GAME_OVER without a no-action failure. A batch longer than one and an offset greater than zero are useful secondary evidence, not required to qualify the single action surface.

Measure: exact installed source hashes; decision-checkpoint requests and responses; `llm-trace.jsonl`; `tool-trace.jsonl`; `action-batch-trace.jsonl`; official recording; SGLang access log; outer `result.json`; per-sequence creation-to-action timing. Verify that all executed environment actions originate from `take_actions` releases and that every required request contains only `take_actions`.

Cap: one RTX PRO 6000 notebook; one public game (`ls20`); eight Solver actions; at most 13 Actor calls and 14 successful model-server chat accesses including warmup; 16,000 completion tokens per call, at most 208,000 Actor completion tokens; 1,200 seconds startup plus 7,200 seconds launcher time; financial spend not applicable.

Stop: stop after eight reconciled actions or an earlier WIN/GAME_OVER; abort on source/hash drift, competition-rerun mode, malformed action, missing successor reconciliation, any completed legacy `take_action` call, model-server error, decision timeout, launcher timeout, or containment/lifecycle failure. Lack of a multi-action batch is a completed secondary negative result, not grounds for an automatic retry.

Invalid if: the v10 wheelhouse or installed source hashes differ; any Actor request offers `take_action`; the official launcher, adapter, model route, public game, caps, or network setting differ from the frozen notebook; persisted Actor requests do not reconcile one-for-one with completed provider responses; an action cannot be reconciled to the official recording; output or immediate source bytes are not preserved.

Result map: eight reconciled actions or earlier terminal state with the singular surface intact -> qualify the action interface and inspect latency/gameplay before expanding to three public games; any required response still ends without `take_actions` -> the choice-confusion hypothesis is contradicted and the next probe should bound commitment generation rather than restore a second primitive; a completed `take_action` or unreconciled release -> reject the candidate; ambiguous or invalid -> repair the measurement/source defect and issue a new card. Do not create a competition submission.

Outputs: local candidate `deployment/kaggle/s0-public1-singular-action-v13-candidate`; frozen card and reconciliation under `analysis/deployment-probes/20260902T-public1-singular-action-v13`; exact remote source and output downloaded before interpretation.
