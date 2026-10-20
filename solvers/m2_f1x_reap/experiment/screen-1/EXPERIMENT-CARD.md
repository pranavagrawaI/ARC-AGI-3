# m2-f1x-reap screen-1 (2026-10-20)

**Question:** Does F1x run on REAP448 serving at 14 streams, with F1x's 116K input, and play no worse than the F1x/B1 screens?

**Why:** F1x (hidden D10 30.88) and REAP448 + c18 (hidden D11, 2026-10-20) are separate lineages, and `incumbent/check.py` fails both. This is the merge: one stack carrying every retained piece. Approved by Pranav on 2026-10-20 ("I agree with this plan, start implementing"). The plan: build the merge now, screen it against F1x, and draw it about 10-22, once D11 and the 10-21 F1x draw have both come back valid.

**Arm:** `screen-1/merge` = `m2-f1x-reap14-screen1` v1, built by `solvers/m2_f1x_reap/build.py`. It is every F1x v1 cell, with REAP v1's serving cell swapped in at F1x's context (CTX (116+16+8)K, 14 streams, 84 Mamba slots) and REAP's late-READY limit gated to screens. The tests prove the merge minus this change is F1x, and the merge minus F1x's harness is REAP.

**Why 14 streams:** F1x on the unpruned pool (1,012,736 tokens, 10 streams) loads KV at p50 0.62 and p90 0.81, about 82K tokens per stream at p90. REAP448 freed 51% more pool (1,373,568 at 108 Mamba slots; about 4.2K tokens per Mamba slot). At 84 slots the expected pool is about 1.47M, and 0.81 × 1.47M / 82K ≈ 14.4 streams. At 16 or 18, the p90 load would exceed 0.9 and 1.0.

**No paired control.** The arm is read against three B1-lineage 60-minute screens: f1x 76, b1ctl 72, b1ctl 65 (mean 71). Same-code runs show no run-level effect (F = 1.08), so one arm against a three-run mean has less noise (≈1.15 σ) than a fresh pair (≈1.41 σ), at half the GPU.

**Measures (`analyze.py`):**
- Levels at the 60-minute stop.
- Mechanism:
  - `pruning/target.json`: 448 target experts, complete.
  - `pruning/draft.json`: 512 drafter experts.
  - Kernel log shows "priority gate active: 14 concurrent streams".
  - Pool `max_total_num_tokens` exceeds F1x's 1,012,736.
  - Decode running requests reach 14.
  - Aggregate gen tok/s over minutes 0–60 is at least 1.10 × F1x's 697.
  - KV full-token usage p90 ≤ 0.90.
  - At most 5 retraction events.
  - Zero NameErrors on preloaded modules.
  - out16k applied.
  - 25 games, none crashed.
- Recorded, no rule: per-stream tok/s, Mamba usage, READY time, tokens per action at minute 30.

**Pre-registered rules:**
- **PASS:** every mechanism check holds and levels ≥ 71 − 7 = 64. The merge v1 is queued for the first free day ≥ 2026-10-22, with `requires: [D11, D12]`. `daily_draw.py` submits it only when both draws are COMPLETE with a score above 0. If either is invalid, the day falls back to F1x. While either is pending, the day waits until 20:00Z, then falls back.
- **MECHANISM_FAIL:** no queue entry. Inspect before any rerun. REAP stays requalifying.
- **NO_HARM_FAIL** (mechanisms hold, levels < 64): no queue entry. One merge arm can't separate REAP from 14 streams, so the next step is a decision, not an automatic rerun.

**Prediction:** about 830 aggregate tok/s, the c14 level on a pool sized for it. KV p90 between 0.78 and 0.86. Levels within the B1 range (65–76). No level gain is claimed. A 60-minute horizon can't show a breadth effect, and the hidden run is where aggregate throughput pays.

**Cap:** one launch, hard cap 6,000 s, one GPU slot. Play is 3,600 s plus at most 1,300 s to READY. Admission requires quota used + 1.67 h ≤ 30 h and a free slot. No retry except for a proven infrastructure failure. This experiment makes no submission; the queue entry is the only output that reaches the draw.

**Known risk carried by D11, not this arm:** REAP v1's late-READY limit is fatal in competition mode. In 1 of 32 dfranzen-family kernels, weight load alone took over 1,200 s, so D11 has roughly a 3% chance of dying on a slow disk. The merge gates that limit to screens.
