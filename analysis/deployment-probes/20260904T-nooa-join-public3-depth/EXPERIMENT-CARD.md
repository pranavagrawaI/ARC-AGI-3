# 20260904T-nooa-join-public3-depth

Question: On the three public games the incumbent join already plays, do 80 official actions complete a level?

Posture: Execution. Action-horizon test on the frozen q3c join. Not a chassis change. Not q5.

Contrast: v2 same three games (`ls20`, `s5i5`, `cd82`) at 16 actions, 0 levels. Local q3c `ls20` completed L0 in 32 actions.

Prediction: At least one of the three reaches `max_levels_completed >= 1`, or 80 actions still leave all three at 0. Either is a decision. Clock remaining after a level is a secondary field, not the pass/fail.

Measure: pulled `official-launcher-result.json` fields `public_progress.games.*.max_levels_completed`, `public_progress.games.*.grid_hash_changes`, `public_progress.total_levels_completed`, `solver_sessions.games.*.submitted_env_actions`, `timed_out`. `recording_action_input_ids` remaining `0` is not "no play."

Cap: one private Phase A push of a new kernel `pranavagrawai/prototype-nooa-join-public3-depth`; internet off; no competition rerun; no retry; RTX PRO 6000; launcher 21600 s; worker 7200 s; session timeout 28800 s; 80 actions/game; games `ls20,s5i5,cd82`; GPU-hours ≤ 8.0. Concurrent with `20260904T-nooa-join-public25-contact` only.

Stop: success if any game has `max_levels_completed >= 1`. Futility if 80 actions complete and all three stay at 0. Abort on wrong GPU or competition rerun. Timeout after some games is a result for those games; do not retry 80.

Invalid if: chassis metadata is not `q3c-mdfiles-after`; more than one push; q5 files are in the join zip; the probe still treats `action_input.id == 0` as no play; the kernel is a competition submit; games are not exactly `ls20,s5i5,cd82`.

Result map:
- positive (level ≥ 1) → Sep 10 0→10 binding "complete a level" is live on this join. Keep 80 as the article budget for these three until a cheaper cap also completes a level.
- negative (80 actions, 0 levels) → more actions on these three is the wrong next spend. Next card is representation or judgment, not 160 actions.
- timeout with a completed game and a truncated one → score the completed game; do not retry the truncated one at 80.
- ambiguous (recordings absent) → preserve pulled output; do not retry.

Outputs: `analysis/deployment-probes/20260904T-nooa-join-public3-depth/`; notebook `deployment/kaggle/s0-nooa-join-public3-depth-notebook/`.
