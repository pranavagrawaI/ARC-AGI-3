# m2-b1-smoke-1
Question: Does the B1 notebook run end to end on Kaggle, and do its three mechanisms fire under live traffic?
Posture: operational qualification before spending 4.4 GPU-h on the replicate pair. Not a performance test.
Notebook: solvers/m2_b1/kernel/m2-b1.ipynb, built and CPU-qualified (qualify.py: 10 checks, including the real-request pin replay), with two smoke-only edits from build_runs.py:
games restricted to lf52/s5i5/ft09 (as m2_evidence smoke-1), and the analyzer context window cut to 52K in the overlay cell so live trims happen within 870 s per game.
Prediction: COMPLETE; the log shows "M2 B1 installed"; artifacts/*_b1.json for all 3 games with trims_with_pin ≥ 1 in at least 2 games and fallbacks = 0; python_timeout_s = 60 and static_turn_rules = true;
no tool-call format failures beyond the m2_evidence smoke's level; no new exception type in the transcripts. Levels are not a criterion (m2_evidence smoke: lf52 2, s5i5 2, ft09 4; n = 1, and the window here is smaller).
Measure: kernel status; the log; artifacts/*_b1.json; transcripts grepped for Traceback types, "tool-call", "No user query", "context_overflow"; benchmark.json levels (descriptive only).
Cap: 1 session, 1 RTX PRO 6000, hard timeout 2,400 s; 870 s per game, concurrency 3 (inherited non-submission settings); no submission. Counts against the experiment ceiling of 6.0 GPU-h (launch.py).
Stop: the session ends at its cap. No retry under this card.
Invalid if: not on an RTX PRO 6000; "M2 B1 installed" missing; any game missing from benchmark.json.
Result map: all predictions hold → launch pair-1 unchanged. The pin never fires (trims_with_pin = 0 everywhere) → diagnose before pair-1, which would then test only the other two items. Crash, "No user query", or new exception types caused by B1 → fix locally, re-qualify, and run a second smoke under a new card. Format failures clearly above m2_evidence smoke → set ARC3_STATIC_TURN_RULES=0 for the pair and record that change.
Outputs: solvers/m2_b1/experiment/smoke-1/{launch-attempt.json, launch-response.json, status-history.jsonl, output/, RESULT.md}.
Authority: Pranav, 2026-10-10: "you will start building b1". GPU budget: this week's 30 h quota; another agent (marlin-reduction replays) is also using it.
