# actor-critic-competition-20260917-v1
Question: Does the frozen Actor plus critic package complete the official competition rerun and yield submission.parquet within the 9-hour envelope?
Posture: Feasibility qualification.
Contrast: none.
Prediction: The gateway supplies all 110 games and writes a complete schema-valid submission.parquet before the enforced deadline.
Measure: Exact-version provider status, gateway output row_id/game_id/end_of_game/score, all-game ledger, request lifecycle and server-count reconciliation. Completion is operational evidence only, not capability or scientific improvement.
Cap: At most one kernel push and one competition submission; no retry; 110 private games; provider 32400 seconds maximum on one RTX PRO 6000; 9 GPU hours maximum per execution; existing quota only (discovery snapshot 25.92 hours, must refresh before authorization). No paid external model calls, internet, dataset push, or second candidate. Request and time caps are mechanically frozen in budget.py. Commit execution writes dummy parquet only.
Stop: Stop on completion, deadline, startup failure, request failure, accounting mismatch, source/metadata drift or cleanup failure. Exhausted per-game budgets end that game and advance. Never relaunch on uncertain provider state.
Invalid if: Wrong identity/version/bytes, incomplete games or output schema, missing raw evidence, request-count mismatch, exceeded cap, unexpected endpoint, fixture routing in competition, or ambiguous cleanup. A valid local fixture is not a remote result.
Result map: Complete valid rerun -> retain exact output for review; failed valid run -> preserve evidence and diagnose without retry; invalid/ambiguous -> HOLD, preserve evidence, no promotion or resubmission.
Outputs: This directory's future exact-version provider receipts and raw-output directory, submission.parquet, qualification/result.json, game-progress.jsonl, budget-allocations.jsonl, server/request and shutdown logs; local qualification evidence is stored separately under checks/.
Authority: This card is not authorization. T3 authorizes local qualification only. Fresh independent review and exact-byte authorization are required before any push; competition submission also requires explicit authorization binding the exact pushed version. No authorization or consumed marker is created by T3.
