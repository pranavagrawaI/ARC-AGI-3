# 20260905T-nooa-join-public3-reasoning

Question: With the GAME_OVER Solver episode closed, does thinking-on Qwen on the frozen q3c join complete a level in 80 official actions on ls20, s5i5, or cd82?

Posture: Execution. Two changes vs depth v1, named here and not separable by this run.

Contrast: depth v1 (`20260904T-nooa-join-public3-depth`): thinking off, GAME_OVER → adapter RESET left `LocalSolver` on the dead grid, 0 levels, crash at s5i5 frame 49. Same games, same 80-action cap, same chassis pin `q3c-mdfiles-after`, same R8-L3B route.

Prediction: At least one game reaches `max_levels_completed >= 1`, or all three stay at 0 with thinking actually on. Either is a decision. A GAME_OVER must not raise `turn does not match the last public observation`.

Measure: pulled `official-launcher-result.json` fields `public_progress.games.*.max_levels_completed`, `solver_sessions.games.*.episode_count`, `public_progress_error`, `timed_out`; kernel `result.json` `error` must not contain `turn does not match the last public observation`. Thinking-on launch condition: notebook `ARC_LLM_EXTRA_BODY` contains `enable_thinking":true` and does not contain `reasoning":{"enabled":false}`. Secondary: worker/server logs for reasoning/thinking spans if present.

Cap: one private Phase A push of a new kernel `pranavagrawai/prototype-nooa-join-public3-reasoning`; internet off; no competition rerun; no retry; RTX PRO 6000; launcher 21600 s; worker 7200 s; session timeout 28800 s; 80 actions/game; games `ls20,s5i5,cd82`; `ARC_LLM_MAX_TOKENS=8192`; LLM/request timeout 300 s; GPU-hours ≤ 8.0. Concurrent with contact only.

Stop: success if any game `max_levels_completed >= 1`. Futility if 80 actions complete (or GAME_OVER episode + remaining cap) and all three stay at 0 with thinking-on confirmed. Abort on wrong GPU or competition rerun. Timeout with a nonempty `public_progress.games` is a result; do not retry.

Invalid if: chassis is not `q3c-mdfiles-after`; extra_body still has thinking off; more than one push; q5 seed is in the join zip; `turn does not match the last public observation` recurs; competition submit.

Result map:
- positive (level ≥ 1) → thinking-on + GAME_OVER resync produced a level. Next card is a thinking-off control with the same GAME_OVER fix, own card, so the two factors can be separated.
- negative (0 levels, thinking-on held, no continuity crash) → thinking-on did not rescue this chassis at 80. Next is representation (q5 if local S1 ≥ 1) not 160 actions and not another thinking-on retry.
- GAME_OVER continuity crash recurs → the resync did not ship; stop and inspect the adapter zip.
- timeout / truncated → score completed games; do not retry this card.
- ambiguous (recordings absent) → preserve pulled output; do not retry.

Outputs: `analysis/deployment-probes/20260905T-nooa-join-public3-reasoning/`; notebook `deployment/kaggle/s0-nooa-join-public3-reasoning-notebook/`.
