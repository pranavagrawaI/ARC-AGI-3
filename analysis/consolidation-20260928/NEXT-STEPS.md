# Consolidation 2026-09-28: four survey threads → one plan

Inputs (all 2026-09-28):

| Thread | Agent | Output |
|---|---|---|
| "top positions are not open-sourcing" | Claude 85782cc2 | Tufa declined M2 disclosure (discussion 742801); M2 is an option, not the plan |
| "wait for milestone 2?" → K/KF → field survey | Claude 65b233b5 | KF 44 vs K 36 levels; KF submitted (ref 56645750); `duck-qwen38/runs/flash-next/FIELD-SURVEY-20260928.md` |
| incumbent survey (our runs) | Codex 01a0e8bb | `analysis/incumbent-survey-20260928/REPORT.md` |
| community survey (the field) | Codex 01a0e8bc | `analysis/community-survey-20260928/REPORT.md`, board #482/#483 |

## State

- Incumbent **KF** = `duck-flash-next-kf2` v1: 44 levels, 21/25 games, local 10.15, one pass.
- Submission 56645750 **PENDING** at 16:50Z (submitted 15:46Z, 9 h cap). Our hidden scores so far: 0.67 → 1.47 → 1.58.
- Board: leader 28.34 (Yi-Chia Chen), Tufa 27.29, #5 20.53. Top recipes are private; Tufa will not publish for M2.
- GPU week Sat 09-26 → Sat 10-03 00:00Z: about 6.5 h used, **about 23 h left**. Two sessions at a time. One submission per day.

## The finding that reorders the lists: KF is KV-bound

Checked in `out-kf2/vllm-openai-server.log`:

- `kv_cache_memory_bytes` = 5 GiB → 105,202 KV tokens (3.21× at 32K).
- Steady state: **Running 5, Waiting ~20, KV usage 88–99 %**. `max_num_seqs=8` is never reached. Codex's metric: median 3 running / 22 waiting; 87 % of request latency is queueing.
- GPU is full: model 81.8 GiB, then the autotuner OOMs with 42 MB free after the KV reservation. There is no free VRAM for more KV.
- Time still binds on this stack: 13/44 levels came after minute 80, and 8 after minute 100.

Consequences:
1. **Scott's "c16" (`max_num_seqs=16`) cannot help by itself**, because the sequence cap is not the constraint. The field survey ranked it #3 for the wrong reason.
2. The lever is **KV tokens per GiB**: an FP8 KV cache fits about 2× the tokens in the same 5 GiB. This is a config change. An expert-pruned checkpoint would free VRAM, but it needs loader work, so it waits until next week.
3. **F1 image trimming works against throughput.** By correcting the estimator's 5–8× overcharge for images, it keeps *more* real tokens per sequence, so more KV is used per game. It trades throughput for retained reasoning. Test it after the KV lever, not before.

## Where the threads disagree, and the ruling

| Issue | Position A | Position B | Ruling |
|---|---|---|---|
| Wait for M2? | Claude 65b233b5 (early): wait | Claude 85782cc2: don't bank on it | **Don't wait.** Oct 1: a 10-minute check for teams below the top 3 that publish to claim the prize (`tools/milestone2_intake.py` is ready, uncommitted). |
| Scott's 5.19 | Field survey: from F1/F3 | Codex #483: animation+sheets ON, upscale 12 | **Codex is right** (source audited). 5.19 does not attribute to any one fix. |
| KF hidden-score forecast | Field survey: ~4–4.6 via another solver's local/hidden ratio | Codex: invalid calibration | **Invalid.** Wait for 56645750's score; it is the only KF calibration. |
| Isolate every fix? | Both Codex reports: one change per arm | — | **No for defect repairs.** With delta sd ≈ 4 levels and ≤2 passes, a +1–2 level fix is invisible alone. Bundle the repairs that cannot be net-negative. Use trace mechanism checks for attribution. Isolate only the *bets*. |
| Top lever | Field survey: F1+F3; Codex-incumbent: ACTION7/parser; Codex-community: image accounting | — | **KV capacity (FP8 KV) is the bet with the most upside. The repairs go in as one bundle.** |

## Arms (all vs KF: public-25, 1 pass, 7920 s, same image/datasets, fresh kernel id each)

| Arm | Change | Mechanism check (must pass before levels count) | GPU |
|---|---|---|---|
| **R** repair bundle | ACTION7 dispatched as the real engine action (no UNDO alias taught); tolerant world-model label parser | ACTION7 executed > 0 in ar25/bp35/lf52/sb26/sk48/su15; unavailable action still rejected; parser fill ≈ 465/465 on replay (now 188/465) | ~3 h |
| **S** KV lever | `kv_cache_dtype=fp8`, `max_num_seqs` 8→16, same 5 GiB | Server loads (abort in ~15 min if not); median running > 5; aggregate gen tok/s up; tool-call parse errors not up | ~3 h |
| **C** control | KF again, unchanged | First same-stack noise estimate: replaces the 3.6-era sd ≈ 4 | ~3 h |
| **RS** | R + S (if S passes its mechanism check) | as above | ~3 h |
| **RSI** image package | RS + F1 (image trimming, token estimate measured, not a flat 120) + `MULTIMODAL_UPSCALE` 4→8 | real image tokens measured; running count not below RS | ~3 h |

Order: **R ∥ S** (Tue) → **C ∥ RS** (Wed) → **RSI** (Thu). That is about 15 h, leaving about 8 h in reserve.

Submission rule: submit the best arm only if it beats mean(KF, C) by ≥ max(5, 2 × |KF − C|) levels and its mechanism checks pass. Otherwise KF stays. Each submission needs Pranav's go.

## Not now (and why)

- Critic / questioner / self-question: four nulls (11 v 14, 21 v 21, 43 v 45, 74 v 72).
- NOOA lineage: 0 levels.
- 16K context: it cuts evidence to buy concurrency, and FP8 KV buys the same concurrency without cutting anything. Use it as a fallback only if S fails to load.
- Pruned REAP k448 checkpoint: separate MTP bank and loader qualification needed; next week's serving lever if S wins and we want more.
- Prediction-checked batches (OY1/Retrodict), PRO-LONG retrieval, SB26 level-boundary record: real ideas, but they need architecture work. Keep the SB26 and LS20 cells as replay fixtures.
- Animation (p = .92, +17 % tokens), LoRA/SFT (no replicated gain), GAME_OVER note retention (policy, not a defect; Codex's staleness warning holds).

## What could reverse this plan

The pinned dev vLLM build (`0.1.dev20073+g8e685d198`) may reject FP8 KV for this hybrid GDN/QSA model, or FP8 KV may damage long-context tool use. S's first 15 minutes and its parse-error rate show which. If it fails, the fallback lever is F1 plus a 24K context, measured the same way.
