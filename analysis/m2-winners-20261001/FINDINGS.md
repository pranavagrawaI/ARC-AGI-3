# Milestone 2 winners vs our stack (2026-10-01)

Sources pulled here: `{dfranzen,lordhansolo,sirikilohit}/` (notebook, metadata, `code.py`, kernel outputs in `out/`).
dfranzen's full source + writeup: `~/projects/arc-agi-3-reference/dfranzen-m2/WRITEUP.md` (github da-fr/arc-agi-3-solution).
Reproduce the tables: `python3 action_matched.py dfranzen lordhansolo sirikilohit` (levels = `score` field of action events).

## Leaderboard (2026-10-01)
Tufa Labs 50.65 · Yi-Chia Chen 40.80 · **dfranzen 27.89** · the last dance 24.54 · **lordhansolo 23.84** · **rellik13/sirikilohit 22.53** ·
... · keithtyser 12.09 · ... · **us: KF 2.96** (P = 56712962 still PENDING).

## Same scorer, so the numbers compare directly
`final_score` in their benchmark.json == our `score.json`: per level, weight = level index, times an action-efficiency
factor vs `base_actions_per_level`.

| run | games | wall/game | levels | mean score | efficiency* | actions/level |
|---|---|---|---|---|---|---|
| dfranzen demo | 10 | **25 min**, 10-way | **40** | **36.56** | 0.98 | 33 |
| our P2 on the same 10 | 10 | 132 min, 25-way | 37 | 33.89 | 0.93 | 49 |
| our P1 on the same 10 | 10 | 132 min | 31 | 22.33 | 0.92 | 72 |
| our KF on the same 10 | 10 | 132 min | 27 | 18.95 | 0.91 | 54 |
| lordhansolo demo | 25 (11 never admitted) | **15 min** | 25 | 5.28 | 0.98 | 35 |

*efficiency = score / score-if-every-level-were-at-human-efficiency. **Efficiency is not the gap; level depth is.**

GPU share per game: dfranzen's demo ≈ 25 min × 1/10 = 2.5 GPU-min; ours ≈ 132 min × 1/25 = 5.3 GPU-min (≈ the hidden run's
540 min / 110 games = 4.9). **dfranzen beats our best pass on the same games using about half the GPU per game.**

Action-matched (our levels after the same per-game action count as theirs): dfranzen 40 vs P 30–33 vs KF 25;
lordhansolo 25 vs P 17 vs KF 12. So per-action quality is about 1.3–2× ours. Aggregate actions/min: dfranzen 53, P 42, KF 26.

## Claims in our record this contradicts
1. **"fp8 KV is impossible on Flash-Next"** (kf-is-kv-bound): wrong scope. All three winners run `fp8_e4m3` KV
   (SGLang Pennyroyal v2.5.x; vLLM main e975732). The `QSA requires BF16 KV` error is specific to Keith's pinned vLLM build.
2. **Arm G "12× board costs ~30% actions, don't re-run"**: dfranzen reports 10× upscale as his *biggest* image win. Confound:
   Duck counts an image by its base64 length (~8× too high, per siri M74 and dfranzen §2), so on our 32K context big images
   force trims. dfranzen fixed the accounting (image tokens from dims/patch layout) and runs a 128K window. G tested the image
   against a broken budget, not the image itself.
3. **R (ACTION7 + tolerant WM parser) null**: dfranzen got his best result by turning the world model *off*
   (`ARC3_MEMORY_SECTIONS=off`) with large context, and exposes ACTION7 as `UNDO` (helped on the competition set; not isolated).
4. **"Duck sliding window breaks cross-turn prefix"** (arm P): solved by hysteresis trimming (128K window, drain to ~57K) plus a
   Mamba-checkpoint retention patch. 93% of prompt tokens come from cache.
5. **KF "5 running / 20 waiting, KV-bound"**: dfranzen caps admission at 10 streams so that returning games find their prefix
   resident. A priority scheduler (writeup §3) gives freed slots to games with high expected score gain.

## What each winner changed vs Duck 7652836 (the parts that matter)
- **dfranzen**: Intel W4A16 AutoRound + Albucino MTP drafter (frees VRAM vs NVFP4), SGLang Pennyroyal 2.5.3 + prefix-cache and
  prefetch patches, fp8 KV, 128K/12K out, block trim; 10 streams + priority scheduler P=(A+B)·C (all 110 games started; 10 admitted);
  yield on 2,048 generated tokens instead of on seconds; persistent model functions; UNDO; frame_diff; animation frames as REPL
  variables; 10× images + diff images + death frame; game-over and level-transfer prompts; no-op/stale/terminal guards.
- **lordhansolo**: own "Improved TAAF" bundle (not Tufa's), vLLM main e975732, own mixed NVFP4/FP8 quant, MTP 3 + 32k draft
  vocab, 127K window / 81.5K soft target, 14 seqs, reasoning_effort xhigh, upscale 4.
- **sirikilohit**: Keith's NVFP4 on SGLang Pennyroyal 2.5.0, fp8 KV + 48 GB host tier, 57K→45K watermarks, 16 slots / 28 games,
  ACTION7 fix, multi-request turns, UCB one-pool scheduler, solved-level memory, ported Wang/Ludvig harness (board 11.5).

## Caveats
Their demos are n=1 on a subset (dfranzen used 10 games; lordhansolo had 11 games never admitted in 15 min). Board scores for
identical code spread about 2× (duck-score-noise-vs-shift). Neither caveat comes close to explaining a 9× hidden-score gap.
