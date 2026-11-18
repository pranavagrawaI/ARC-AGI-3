# Speed screen — card (frozen before push, 2026-09-30)

Plan: `~/projects/arc-agi-3/analysis/kv-lever-scope-20260930/NEXT-PLAN.md`. Kernel `pranavagrawai/duck-flash-next-speed-screen`.
Authorization: the user said "execute the final plan" (2026-09-30).

## What runs

The notebook is P1's, byte-identical for cells 0–10: P's serving profile, the k448 pruning plugin, the S0 server,
and the ARM P guard. No games are played. `speedbench.py` replays P1's 25 final prompt snapshots as 25 concurrent
Duck-like sessions:

- same system prompt, tool, and sampling as Duck;
- sliding-window trim at the start of each turn to ≤ 20,500 tokens;
- append-only within a turn, with at most 2 requests per turn (P: 1.9).

Each config runs a greedy check (4 prompts × 2 attempts, T=0, 128 tokens), then 180 s of warmup, then 600 s of
measurement in 5 windows of 120 s.

| config | change vs P |
|---|---|
| S0 | none: the server started by P's setup |
| S1 | `TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION=1` |
| S2 | S1 + `TAAF_VLLM_ENABLE_PREFIX_CACHING=1` (vLLM selects mamba `align`) |
| S3 | base + MTP k. Base is S2 if S2 ran and its cached greedy outputs match S1 no worse than max(1, S0-vs-S1 mismatches); else S1; else S0. k=2 if the base's third-position acceptance under load is < 0.35, otherwise k=4, with cudagraph capture 60. |

## Decision rule (P+ selection), fixed now

1. **Integrity (per config).** A config fails this gate if any of these holds:
   - it did not run;
   - the pruning guard failed;
   - its intended flag is absent from the server's non-default args (`index_share_for_mtp_iteration`,
     `enable_prefix_caching`, `num_speculative_tokens`);
   - it had more than 1 % client errors after warmup;
   - for S2, and S3 built on S2: the greedy check fails the S3 rule above.
2. **Speed.** Metric: median over the 5 windows of generation tok/s (verified output) and of completed requests/s.
3. **Choice of P+.**
   - Take the integrity-passing config with the highest median gen tok/s.
   - If a measured config with a strict subset of its changes is within 3 % on gen tok/s, choose that one instead
     (fewer changes wins ties).
4. **Screen passes if both hold for P+:**
   - its median gen tok/s and median requests/s are each ≥ 1.10 × S0's;
   - ≥ 4 of its 5 windows exceed S0's median gen tok/s.
5. **If the screen passes:** build P+ = P notebook + P+'s env overrides (nothing else changes) and run the
   validation card (two full passes). **If it fails:** keep P and run no validation this week.

## Costs and known limits

- About 2.2 GPU h: ~19 min setup, 3 restarts of ~17 min, and 4 × ~14 min of load. Quota before push: 8.48 h.
- The replay is a proxy. Generated text drives acceptance and length, and the snapshots are end-of-game
  contexts. Differences between configs are the signal; absolute numbers are not comparable to P's live
  384 tok/s.
- The prefix hit rate here reflects Duck's current sliding window as modeled: within-turn appends plus the
  shared system prompt.
