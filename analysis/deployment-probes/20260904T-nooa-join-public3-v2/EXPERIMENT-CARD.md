# 20260904T-nooa-join-public3-v2

Question: With the hardened-host uid/gid assignment copied into the NOOA wrapper, does the q3c chassis on the accepted R8-L3B route commit at least one non-RESET action across ls20, s5i5, and cd82?

Posture: Execution. One-factor successor of `20260904T-nooa-join-public3`. That run listed the three games and died in `create_solver` on missing `TYCHO_KAGGLE_GAME_UID`.

Contrast: v1 wrapper without identity assignment (same games, same route, same caps).

Prediction: `create_solver` no longer raises on GAME_UID. The set produces ≥1 non-RESET environment action. Levels may still be zero.

Measure: pulled `official-launcher-result.json` fields `status`, `public_progress.total_solver_actions`, `public_progress.games`, `solver_sessions_error`; kernel `result.json` `error` must not contain `TYCHO_KAGGLE_GAME_UID`.

Cap: one private Phase A push (kernel version 2 of `pranavagrawai/prototype-nooa-join-public-3`); internet off; no competition rerun; no retry; RTX PRO 6000; launcher 7200 s; 16 actions/game; 3 games; GPU-hours ≤ 2.0.

Stop: success if GAME_UID is absent from the error and the set has ≥1 non-RESET action. Futility if the worker still dies before `actions.jsonl`. Abort on wrong GPU or competition rerun.

Invalid if: the stub agent ran; chassis metadata is not `q3c-mdfiles-after`; more than one push; the notebook zip still lacks the identity assignment.

Result map:
- positive → freeze this join as the Sep 10 article base; next is packaging/scorecard, not q5.
- negative (GAME_UID again) → the repair did not ship; stop and inspect the zip.
- negative (worker starts, model never submits) → join is live; cell/request contract is the next card.
- negative (other join/load) → repair that defect under a new card; do not open q5.
- ambiguous → preserve pulled output; do not retry.

Outputs: `analysis/deployment-probes/20260904T-nooa-join-public3-v2/`; rebuilt notebook in `deployment/kaggle/s0-nooa-join-candidate/s0_nooa_join_public3.ipynb`.
