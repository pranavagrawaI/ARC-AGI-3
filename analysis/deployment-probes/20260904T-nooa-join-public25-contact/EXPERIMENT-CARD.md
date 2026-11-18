# 20260904T-nooa-join-public25-contact

Question: Across all 25 public games, which ones does the frozen q3c join actually step within 8 official actions, and does any complete a level?

Posture: Execution. Portfolio scan on the incumbent join. Not a chassis change. Not q5.

Contrast: v2 three-game 16-action run (`ls20`, `s5i5`, `cd82`), which stepped all three and completed 0 levels.

Prediction: A subset of the 25 will show `grid_hash_changes >= 1`. Most will complete 0 levels at 8 actions. The ranking of games that move is the result, not a mean score.

Measure: pulled `official-launcher-result.json` fields `public_progress.games.*.grid_hash_changes`, `public_progress.games.*.max_levels_completed`, `public_progress.missing_games`, `solver_sessions.games.*.submitted_action_names`, `timed_out`. Kernel ERROR with partial recordings is still a result if those fields exist. `recording_action_input_ids` remaining `0` is not "no play."

Cap: one private Phase A push of a new kernel `pranavagrawai/prototype-nooa-join-public25-contact`; internet off; no competition rerun; no retry; RTX PRO 6000; launcher 21600 s; worker 5400 s; session timeout 28800 s; 8 actions/game; 25 public games; GPU-hours ≤ 8.0. Concurrent with `20260904T-nooa-join-public3-depth` only.

Stop: success if ≥1 game has `grid_hash_changes >= 1` or IPC `submitted_env_actions >= 1`. Futility if the worker dies before any `actions.jsonl`. Abort on wrong GPU or competition rerun. Timeout with a nonempty `public_progress.games` is a result, not a retry.

Invalid if: chassis metadata is not `q3c-mdfiles-after`; more than one push; q5 files are in the join zip; the probe still treats `action_input.id == 0` as no play; the kernel is a competition submit.

Result map:
- positive (some games step; rank them) → next GPU hours go to games that moved. Do not deepen games with 0 grid changes.
- negative (0 games step) → contact does not generalize past the v2 three. Do not spend the next card on 80-action repeats of the other 22.
- timeout with k < 25 games → treat the k as the sample; do not retry 25×8.
- ambiguous (recordings absent) → preserve pulled output; do not retry.

Outputs: `analysis/deployment-probes/20260904T-nooa-join-public25-contact/`; notebook `deployment/kaggle/s0-nooa-join-public25-contact-notebook/`.
