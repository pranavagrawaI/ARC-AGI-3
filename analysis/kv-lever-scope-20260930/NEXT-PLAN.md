# Speed plan after P — final (2026-09-30)

This merges two plans: the "remaining speed experiments" section of SCOPE.md (MTP index sharing first) and the
P-log profile (prefill interference first). Nothing has been launched.

## Grounds

| Fact | Value | Source |
|---|---|---|
| Decode lost to concurrent prefill | 25 % (slope −0.024 gen tok/s per prompt tok/s, P1 = P2) | out-p{1,2}/vllm-openai-server.log, 10 s intervals, running ≥ 7, waiting > 0 |
| Decode with no prefill running, 8 requests | 549 tok/s, vs a run mean of ~380 | same logs, prompt < 1,500 tok/s |
| Decode-only scaling with running requests | 3: 314, 8: 549, 10: 654 tok/s | same logs, P + KF |
| Request shape | 21.3K prompt, 1.5K generated, 1,981 requests, 242 preemptions | vllm-metrics-final.prom |
| Shared system prompt | ~3.4K tokens (12,683 chars), identical in every game | out-p1/prompts |
| MTP3 acceptance by draft position | 76 / 58 / 45 % | spec_decode_num_accepted_tokens_per_pos |
| Index sharing in P | off (`mtp_index_share_for_iteration: false`); pinned source supports it | vllm-setup-provenance.json; llm_base_proposer.py:580 |
| Prefix caching | off; target and MTP reject only `mamba_cache_mode=all` | nvidia/model.py:617, nvidia/mtp.py:372 |
| Duck history trimming | drops the oldest block once over budget, so across turns only the system prompt and tools are a stable prefix | tool_agent.py `_trim_messages_for_context` |
| Server start | ~16 min (12:43 → 12:59 in P1; PLE offload is the long pole) | out-p1 server log |
| Quota | 8.48 GPU h; resets 2026-10-03 00:00Z to 30 h | live check |

The other plan corrects mine: prefix caching also shares live KV blocks. A 3.4K-token system prompt shared by 8–12
concurrent requests saves ~24–37K of 252K KV tokens, about 10–15 % more capacity, on top of the prefill saving.

## Decisions

1. **Screen with a serving benchmark, not a gameplay probe per knob.** Each server start costs ~16 min. The
   benchmark replays P's recorded prompts at sustained demand. It rebuilds within-turn requests from message
   boundaries in each game's snapshot, so the requests share the system prompt exactly as live play does.
2. **Screen configs in one session, run in this order**, each with ~10 min of measurement:
   - S0 = P (a matched baseline on the same GPU)
   - S1 = P + index sharing
   - S2 = S1 + prefix caching (`align`)
   - S3 = S2 + the better MTP length. Pick k=2 or k=4 from S0–S2 acceptance at 8+ running: k=2 if the
     third-token acceptance falls below ~35 % under load, otherwise k=4.
   - Record `nvidia-smi` peak memory in every config. That answers the 4,096-batched-tokens question (headroom
     for a bigger KV pool) without a config of its own.
3. **Screen gates.**
   - Correctness: greedy output on 4 fixed prompts at concurrency 1 matches between S1 and S2 (prefix caching
     must not change outputs under QSA/PLE).
   - Speed: throughput gain ≥ 10 % over S0 in verified tok/s and completed requests/s, repeated across windows.
   - Liveness: the worker logs show each flag was actually enabled.
   - The stack that clears these gates becomes **P+**. Budget ≈ 2 GPU h.
4. **Validate P+ this week.** Run two full passes (~5–6 GPU h) against P. Every P+ change is speed-only, so the
   rule tests for non-regression plus the mechanism, not a new level threshold.
   - Mechanism: busy-window generation ≥ +10 % over P's 384 tok/s.
   - Levels: mean ≥ 48.5 (P 54.5 minus one per-pass sd of ~6).
   - Adopt P+ if both hold; otherwise keep P. Freeze this in a card before launch.
5. **Next week (30 h), changes to inputs or weights, each as its own arm against the incumbent:**
   - Chunked history trimming: when over budget, cut to ~70 % at once. This makes the prefix stable across
     turns, so prefix caching can reach up to ~+33 %, and the shorter average context frees KV.
   - k384 prune: recover the mask from the routers of sh0wie/Qwen3.8-Flash-Next-REAP-384-bf16, not from its
     `reap_kept_experts.json`. Frees ~7.9 GiB. k448 already cost SP80 and TR87.

## Not doing

- Dynamic speculative decoding: current vLLM docs note CUDA-graph limits, so fixed lengths come first.
- MTP off: frees 4.86 GiB but loses ~2.8× tokens per step. At 0.75 slot elasticity it loses on paper.
- Raising max_num_seqs alone: the median is 8 of 12, so the limit doesn't bind.
- FP8 KV: rejected by QSA.

## What could reverse this

- Prefix caching in `align` mode fails, or changes outputs, under QSA/PLE. S2 is then dropped and the trimming
  arm loses its main payoff.
- The replay benchmark misses live-play dynamics (preemption, turn yields). The gameplay validation catches that.
- Index sharing lowers draft acceptance by more than it saves. S1 measures this directly.
