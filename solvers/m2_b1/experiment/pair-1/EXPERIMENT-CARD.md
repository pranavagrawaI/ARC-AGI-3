# m2-b1-pair1
Question: At the public-25 hidden-equivalent budget, does B1 (m2_evidence + level-opening pin + sandbox `time` and 60 s tool calls + static turn rules in the system prompt) beat the dfranzen control?
Posture: evaluation of a bundle. Each part was expected to be worth 2–5%; only the bundle is sized to be measurable. A result supports the bundle, not any one part.
Design: two independent B1 replicates launched together (b1-r1, b1-r2: byte-identical notebook, different kernel ids), compared against the existing full-cap runs with identical settings:
controls c1 (2026-10-01: 107 levels, mean 52.48) and pc (2026-10-08: 96, 40.11); m2_evidence pt (2026-10-08: 105, 45.99).
Post-smoke change (before freeze): the system prompt now states the enforced 60 s tool limit instead of a hard-coded 30 s (smoke-1/RESULT.md). The fixed build passed the 10 CPU checks; it was not GPU-smoked.
No fresh control is run. Identical-run noise is sd ≈ 11 levels per pass, and the hidden draws (README in analysis/hidden-draws-20261010) carry the control series.
Prediction: B1 replicate mean +5 to +15 levels above the control mean (101.5) and +0 to +10 above m2_evidence. Most likely INCONCLUSIVE by the levels rule below.
Mechanism: trims_with_pin > 0 in ≥ 15/25 games per replicate; fewer "Tool timed out after 30s" (none at 30 s should exist); `time` ImportErrors 0; tokens on final unsolved levels at or below m2_evidence's 1.3M.
Measure (primary first):
1. Mean over the 2 replicates of total levels_completed and of mean final_score (output/benchmark.json), against the control mean (c1, pc).
2. Per-game: the B1 replicate mean against the control mean, sign counts; the two replicates against each other (a noise estimate for this arm).
3. Explanatory only: artifacts/*_b1.json counters; artifacts/*_evidence.json; Traceback types; tokens on final unsolved levels.
Cap: 2 sessions launched together, 1 RTX PRO 6000 each, hard timeout 9,000 s each (5.0 GPU-h worst case); 7,254 s per game, concurrency 25, priority gate 10 streams, 1 pass, no action cap; no submission under this card.
Experiment ceiling 6.0 GPU-h including smoke-1 (launch.py enforces it from budget.json).
Stop: sessions end at their caps; no retry of either replicate. If one push fails, the other still runs, and the result is then n = 1.
Invalid if: a replicate is not COMPLETE or lacks a game; its log lacks "M2 B1 installed"; its solver config is not max_runtime_s_per_game=7254, concurrency=25, "priority gate active: 10 concurrent streams"; a different machine shape.
Result map:
- B1 mean ≥ control mean + 15 levels AND score ≥ +8 (POSITIVE): B1 becomes the hidden-draw candidate (b1-r1 v1 is submittable as is), alternating with the control.
- B1 mean ≤ control mean − 10 levels (NEGATIVE): drop B1 as a bundle; attribute from the counters; keep m2_evidence as the candidate.
- Otherwise (INCONCLUSIVE): report the deltas and counters. B1 still enters the hidden-draw series once, because the hidden draws are the better instrument; claim nothing about benefit from public-25.
Outputs: solvers/m2_b1/experiment/pair-1/{b1-r1,b1-r2}/{launch-attempt.json, launch-response.json, status-history.jsonl, output/}; RESULT.md; budget.json.
Authority: Pranav, 2026-10-10 ("you will start building b1"); this week's 30 h quota.
