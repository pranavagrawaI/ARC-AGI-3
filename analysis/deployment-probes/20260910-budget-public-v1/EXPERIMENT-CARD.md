# Budget-aware public Kaggle qualification v1

Question: Does the locally qualified budget-aware candidate complete all three public game sessions with reconciled requests and clean lifecycle handling on the target GPU?
Posture: Operational qualification, not comparative solver evaluation.
Contrast: none.
Prediction: LS20, S5I5 and CD82 each terminate by action cap, win or acknowledged budget exhaustion; no in-flight request is abandoned at game or server shutdown.
Measure: raw/qualification/result.json; game-progress.jsonl and budget-allocations.jsonl; each Solver session's ipc/request_lifecycle.jsonl, model_calls.jsonl, budget_exhausted.json when present, actions and transitions; whole qualification/server.log; teardown lifecycle artifacts and terminal notebook log. Preserve identity and notebook source readback.
Cap: One private Kaggle notebook execution, RTX PRO 6000, session timeout 3600 seconds (at most 1 GPU hour); three public games LS20/S5I5/CD82; at most 10 Solver actions per game; model-call admission capped at 75 seconds per game, client request timeout 90 seconds, post-request tool reserve 15 seconds, cleanup reserve 30 seconds. No independent numeric call-count cap; calls are admitted under the wall-time cap and output limit 16384 tokens per request, retries zero. No competition submission. Starting GPU quota: 8.88 hours remaining. No monetary spend requested beyond existing quota.
Stop: One terminal COMPLETE or ERROR notebook outcome; no automatic retry. Session cap stops execution at 3600 seconds. Any exception follows the notebook's failure/cleanup path. A budget stop is normal only at the acknowledged idle model boundary. Stop interpretation at operational invalidity before interpreting solving performance.
Invalid if: Any OOM, transport timeout or error, unresolved/failed/late/unrecorded model request, aborted request-state notice, failed cleanup, missing game/session/accounting evidence, source or runtime identity mismatch, or time/action cap violation. Recovered failures receive no exemption. COMPLETE alone is insufficient.
Result map: Valid completion -> report operational public qualification only and retain the candidate; no submission under this authorization. Invalid -> preserve evidence, identify the earliest failing boundary and withhold promotion; no retry. Ambiguous/incomplete evidence -> withhold qualification and retrieve/check evidence without launching another run. Zero-action budget stops are reported explicitly, never called solving success.
Outputs: analysis/deployment-probes/20260910-budget-public-v1/raw/ and terminal-log/; source snapshot and remote-source/ alongside this card. Result report added after execution without modifying this card.

Candidate: deployment/kaggle/submission-20260910-v5/submission.ipynb.
Notebook SHA256: b4ff2fc1e8ba3b7c916992252c1e942dff72db24a1f1bbd75014dd09051e1661.
Runtime contract: 8cb520c9b5070a9e50ed898d57461fbb6530d1a863d57b9a67528ea32374d6c1.
Kernel: pranavagrawai/arc3-first-submission-20260910; new version assigned by push.
