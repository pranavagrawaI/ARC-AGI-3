# 20260904T-nooa-join-public3

Question: Does the q3c NOOA chassis, joined through MyAgent/protocol-v3 on the accepted R8-L3B Kaggle route, commit at least one non-RESET action on each of ls20, s5i5, and cd82 in one Phase A session?

Posture: Execution. Feasibility of the live join, not RHAE competence.

Contrast: none

Prediction: The notebook produces recordings for all three games and at least one non-RESET action across the set. Levels completed may be zero.

Measure: `analysis/deployment-probes/20260904T-nooa-join-public3/official-launcher-result.json` fields `public_progress.total_solver_actions`, `public_progress.games`, `solver_sessions.games.*.submitted_batches`, and kernel `result.json` status.

Cap: one private Phase A push; internet off; no competition rerun; no retry; RTX PRO 6000; launcher 7200 s; 16 actions/game; 3 games; GPU-hours ≤ 2.0.

Stop: success if each named game has a recording and the set has ≥1 non-RESET action. Futility if the worker never writes `actions.jsonl`. Abort on wrong GPU, competition rerun, or quota preflight failure.

Invalid if: the stub agent ran; chassis metadata is not `q3c-mdfiles-after`; accelerator is not RTX PRO 6000; more than one push occurred.

Result map:
- positive → freeze this join as the Sep 10 article base; next contact is packaging/scorecard, not q5.
- negative (load/join) → repair the notebook join; do not open q5.
- negative (model never submits) → that is V15-class behavior on this chassis; keep the join, change the request/cell contract under a new card.
- ambiguous → preserve pulled output; do not retry.

Outputs: pulled kernel output under `analysis/deployment-probes/20260904T-nooa-join-public3/`; notebook `deployment/kaggle/s0-nooa-join-candidate/s0_nooa_join_public3.ipynb`.
