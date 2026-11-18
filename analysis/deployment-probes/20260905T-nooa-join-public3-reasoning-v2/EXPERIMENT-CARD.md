# 20260905T-nooa-join-public3-reasoning-v2

Question: With the GAME_OVER Solver episode closed, does thinking-on Qwen on the frozen q3c join complete a level in 80 official actions on ls20, s5i5, or cd82?

Posture: Execution. One-factor successor of `20260905T-nooa-join-public3-reasoning`. That run died at argparse: `--r8-l3b requires --request-timeout-s 180`, notebook had 300. No play.

Contrast: v1 launcher `--request-timeout-s 300` (invalid). Depth v1: thinking off, GAME_OVER continuity crash. Same games, 80 actions, chassis `q3c-mdfiles-after`, R8-L3B, thinking-on extra_body, GAME_OVER resync.

Prediction: At least one game reaches `max_levels_completed >= 1`, or all three stay at 0 with thinking-on held and no continuity crash.

Measure: pulled `official-launcher-result.json` fields `public_progress.games.*.max_levels_completed`, `solver_sessions.games.*.episode_count`, `timed_out`; kernel `result.json` `error` must not contain `turn does not match the last public observation` and must not contain `requires --request-timeout-s 180`. Thinking-on: `ARC_LLM_EXTRA_BODY` contains `enable_thinking":true`.

Cap: one private Phase A push (kernel version 2 of `pranavagrawai/prototype-nooa-join-public3-reasoning`); internet off; no competition rerun; no retry; RTX PRO 6000; launcher 21600 s; worker 7200 s; session timeout 28800 s; `--request-timeout-s 180`; 80 actions/game; games `ls20,s5i5,cd82`; `ARC_LLM_MAX_TOKENS=8192`; `ARC_LLM_TIMEOUT_SECONDS=300`; GPU-hours ≤ 8.0.

Stop: success if any game `max_levels_completed >= 1`. Futility if play completes and all three stay at 0 with thinking-on confirmed. Abort on argparse timeout mismatch or competition rerun. Timeout with recordings is a result; do not retry.

Invalid if: chassis is not `q3c-mdfiles-after`; extra_body has thinking off; `--request-timeout-s` is not 180; more than one push; q5 seed in the zip; continuity crash recurs.

Result map:
- positive (level ≥ 1) → next is a thinking-off control with the same GAME_OVER fix, own card.
- negative (0 levels, thinking-on held, no continuity crash) → thinking-on did not rescue this chassis at 80. Next is representation, not 160 actions.
- argparse timeout mismatch → the v1 defect did not ship fixed; stop.
- GAME_OVER continuity crash recurs → resync did not ship; stop.
- timeout / truncated → score completed games; do not retry.
- ambiguous → preserve pulled output; do not retry.

Outputs: `analysis/deployment-probes/20260905T-nooa-join-public3-reasoning-v2/`; rebuilt notebook in `deployment/kaggle/s0-nooa-join-public3-reasoning-notebook/`.
