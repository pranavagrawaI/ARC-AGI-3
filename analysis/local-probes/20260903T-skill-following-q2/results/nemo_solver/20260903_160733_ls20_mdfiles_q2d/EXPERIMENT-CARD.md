# 20260903_160506_ls20_mdfiles_q2d
Question: with thinking disabled, does qwen/qwen3.8-27b execute the 51-line interactive-game-solver procedure on live ls20 turns?
Posture: verification of a skill-following gate; diagnosis of the q2c transport failure is closed
Contrast: q2c (thinking on, killed EmptyContentError) vs q2d (thinking off). Same skill, game, sandbox, visual-off, model.
Prediction: compute_rate ≥ 0.50; if thinking-off only repairs transport, encode/predict helper and turn_contract may still fail
Measure: analysis/local-probes/20260903T-skill-following-q2/score_q2.py against this run dir; PRECOMMIT.md gate. Also report output tokens per committed env action (not the gate).
Cap: 40 turns, 120 env steps, $6 OpenRouter from remaining 8.99, 4h wall, --agent-turn-timeout 300
Stop: scorer PASS/FAIL after ≥15 turns; abort on spend ≥ $6; abort if EmptyContentError dominates with zero agent submissions
Invalid if: EmptyContentError still dominates; reasoning_tokens return on this hatch; game source referenced in a cell (VOID); fewer than 15 turns (INCONCLUSIVE)
Result map: PASS → B2/WikiSkill card becomes writable under its own holdouts; FAIL on compute_rate → host-side forcing, not better skill prose; FAIL only on helper persistence → notes not code; INCONCLUSIVE → do not interpret; tokens/action still >> 644 with a clean turn contract → tempo, not skill text, is the next constraint
Outputs: this directory (events.jsonl, ipc/actions.jsonl, traces/, team_nemo/shared/, launcher.err); scored by score_q2.py
