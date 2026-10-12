# model-serving wave-1 (2026-10-11)
Question: on the dfranzen control harness, does (E1) the Swift-1.5 Flash-Next checkpoint or (E2) 14 concurrent streams
with a 92K context complete more public-25 levels than the control?
Posture: two independent single-lever screens; each run is n = 1 against two historical control passes.
Design: each arm = the control notebook (`analysis/m2-winners-20261001/runs/control`, the one with request logs)
with only the lines in `build.py` changed, plus a fail-fast after server startup in both arms.
- E1 `wave1-e1-swift-20261011`: MODEL_DIR → a merged symlink view of `phuongncn/arc3-qwen38-swift-w4a16-autoround`
  (w4a16-a + w4a16-b). The drafter (albucino, the base MTP layer in INT4), FR-Spec map (same tokenizer, SHA-256
  checked), chat template (byte-identical) and all harness settings are unchanged.
- E2 `wave1-e2-c14-20261011`: streams/MAXREQ/cuda-graph max 10 → 14 (+ a bs-12 graph), analyzer context 128K → 92K,
  CTX 136K → 100K, drain 58K → 40K, mamba slots 60 → 84.
Controls: c1 (2026-10-01) 107 levels / mean 52.48; pc (2026-10-08) 96 / 40.11; control mean 101.5 / 46.30. Identical-run
sd ≈ 11 levels per pass.
Prediction: E1 tokens per solved level −30 to −60%; levels −5 to +25 (most likely inconclusive at n = 1). E2 gen tok/s
+10 to +20% at similar acceptance; levels +0 to +10 (inconclusive by levels).
Mechanism checks (read before levels):
- E1: the log shows "Swift view: 95 files from 2 parts" and READY; accept length ≥ 2.3 (drafter still matches); median
  generated tokens per level, on levels solved by both E1 and c1, below c1's.
- E2: "priority gate active: 14 concurrent streams"; time-weighted gen tok/s over run-minutes 30–90 vs control's 704
  (same estimator on both serve logs); acceptance within ±0.2 of 2.67; #retracted ≈ 0; KV usage p90 < 0.95.
Measure (primary first): total levels_completed and mean final_score (output/benchmark.json) vs the control mean;
per-game sign counts vs control mean; explanatory: tokens per level, serve-log throughput and acceptance.
Result map:
- E1 ≥ control mean + 15 levels AND score ≥ +8: POSITIVE → E1 becomes the hidden-draw candidate and the base for E2.
- E1 ≤ control mean − 10 levels: NEGATIVE → drop Swift (record the mechanism readings).
- E1 otherwise, with the token mechanism present: INCONCLUSIVE → a second E1 pass is the next spend.
- E2: adopt concurrency 14 if the throughput mechanism holds (≥ +15% tok/s, no KV thrash) and levels ≥ control mean − 5;
  drop it if throughput is < +5% or levels ≤ control mean − 10.
Invalid if: a run is not COMPLETE or lacks a game; the server never reached READY; the log lacks the arm's marker
line; the machine shape differs. A run that fails at server load is a negative result about loading, not about the
lever.
Cap: 2 sessions launched together, 1 RTX PRO 6000 each, hard timeout 9,000 s each (5.0 GPU-h worst case); 7,254 s per
game, concurrency 25, 1 pass. Experiment ceiling 5.5 GPU-h (launch.py enforces it from budget.json). No submission under
this card; no automatic retry.
Outputs: wave-1/{e1-swift,e2-c14}/{kernel/, launch-attempt.json, launch-response.json, status-history.jsonl, output/};
budget.json; RESULT.md.
Authority: Pranav, 2026-10-11 ("Let's do E1 and E2, let's see what we can get"); this week's 30 h quota.
