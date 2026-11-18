# m2-evidence-pair1
Question: At equal execution settings and resource budget, does the m2_evidence treatment raise the competition score of the dfranzen incumbent on the public 25 games?
Posture: evaluation. Try to break the treatment; score is primary, and mechanism counters only explain a result.
Contrast (same-week fresh pair, launched together, same weights, drafter, SGLang, sampling, GPU type, games, priority scheduler and caps; the notebooks differ only by the overlay install cell and a title cell, asserted by build_runs.py):
| Arm | Notebook |
|---|---|
| control | analysis/m2-winners-20261001/runs/control/dfranzen-m2-control.ipynb, unchanged (outputs cleared) |
| treatment | solvers/m2_evidence/kernel/m2-evidence.ipynb = control + overlay (m2_basics fixes, m2_memory data API and cached views, append-only evidence record re-shown after context trimming and at level start) |
Post-smoke change (before freeze): record_evidence became keyword-only after smoke-1 showed scope passed positionally into interpretation (smoke-1/RESULT.md). The fixed build passed CPU qualification and the real-bundle install gate; it was not GPU-smoked.
The treatment is a bundle. A positive result supports the bundle, not the evidence record alone; attribution is follow-up work.
Prediction: treatment mean final_score above control by 0 to +10 points (control r1: 52.48 mean, 107 levels), i.e. most likely INCONCLUSIVE. Mechanism: evidence entries in ≥ 10 of 25 games; the treatment spends fewer generated tokens on final unsolved levels than control (1.4M in control r1); NameErrors fall.
Measure (primary first):
1. Mean of output/benchmark.json game_runs[].final_score over all 25 games, per arm; Δscore = treatment − control.
2. Total game_runs[].levels_completed; Δlevels. Per-game sign counts of both.
3. Explanatory only: artifacts/*_evidence.json (entries, revisions, reshows per game); generated tokens on the final unsolved level (analysis/metacog-transfer-20261008/stuck_tokens.py); python tool error counts by type from transcripts; levels at 100 min (levels_at_time.py) against controls r1/r2 for context.
Cap: 2 sessions launched together, 1 RTX PRO 6000 each, hard timeout 9,000 s each (5.0 GPU-h worst case); 7,254 s per game, concurrency 25, priority gate 10 streams, 1 pass, no action cap; no money; no submission. Experiment ceiling 6.6 GPU-h including smoke-1 and any failed start (launch.py enforces it from budget.json).
Stop: Sessions end at their caps. No retry of either arm under this card. If one push fails, the other arm still runs; the pair is then invalid and is charged.
Invalid if: either arm is not COMPLETE or lacks a game in benchmark.json; the treatment log lacks "M2 evidence installed" or the control log contains it; either log's solver config differs from max_runtime_s_per_game=7254, concurrency=25, "priority gate active: 10 concurrent streams"; the arms ran on different machine shapes; one arm started > 6 h after the other.
Result map (identical controls differ by sd ≈ 11 levels; score noise is not yet measured):
- Δscore ≥ +10 AND Δlevels ≥ +10 (POSITIVE): no submission yet. Confirmation needs another treatment-vs-control pass. If the remaining ceiling cannot hold one (≥ 5.0 h worst case), ask Pranav for budget; otherwise write a new card. Then attribute (memory API vs evidence record).
- Δscore ≤ −10 AND Δlevels ≤ −10 (NEGATIVE): keep the incumbent. Diagnose from traces (record use vs per-game loss, tokens, errors). No further GPU in this direction without a new decision.
- Otherwise (INCONCLUSIVE): report Δscore, Δlevels, sign counts and the mechanism counters; claim nothing about benefit. The remaining allowance (6.6 h − charged) may fund one targeted follow-up chosen from the observed uncertainty, under a new card, or stop.
- Treatment not delivered (evidence entries in < 5 of 25 games): read the result as a test of m2_basics + m2_memory only, whatever the score.
Outputs: solvers/m2_evidence/experiment/pair-1/{control,treatment}/{launch-attempt.json, launch-response.json, status-history.jsonl, output/}; budget.json.
