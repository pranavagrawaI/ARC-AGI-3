# baseline-cost-20260906 — prefill dominates long context; batching helps at Actor lengths

`pranavagrawai/baseline-cost-20260906`, pushed 15:38:53Z, terminal **COMPLETE** 16:24:16Z, ~21 min
of run against a 5400 s runner cap. RTX PRO 6000, model verification `matched`, lifecycle clean
(`containment_passed true`, `root_returncode 0`, `term_only true`, `forbidden_count 0`).
`VERDICT.json`: `MEASURED_REVIEW_REQUIRED`, `promotion: false`.

## Context sweep — cost growth is prefill, not decode

256 output tokens, three warmed repeats, alternating length order. Repeats vary by <0.1 %.

| input tokens | median decode tok/s | median e2e | median prefill | prefill share |
| --- | --- | --- | --- | --- |
| 1,024 | 44.52 | 5.83 s | 0.101 s | 1.7 % |
| 10,240 | 41.39 | 7.28 s | 1.112 s | 15.3 % |
| 21,279 | 38.11 | 9.42 s | 2.725 s | 28.9 % |
| 95,000 | 25.34 | 34.50 s | 24.426 s | **70.8 %** |

Decode degrades gently (44.5 → 25.3 tok/s across a 93x input increase). Prefill grows from
negligible to dominant. At the real Actor prompt length (21,279) prefill is already 29 % of request
time. Queue time was ~0.2–0.4 ms throughout — nothing waited.

Per the card's result map, growing predecode cost with comparatively flat decode points at the
prefill/cache boundary, and the retained `trace-1024` and `trace-21279` are where that inspection
belongs. Both traces are present with GPU kernel events (3.5 MB and 7.0 MB).

## Continuation — historical prefix reuse is real

| request | prompt | cached | e2e | output |
| --- | --- | --- | --- | --- |
| `natural-13` (cold) | 21,279 | **0** | 5.35 s | 100 |
| `continuation-successor-13` (after seed 12) | 21,279 | **18,432** | 2.41 s | 57 |

86.6 % of the successor's prompt was served from cache. `continuation.json` correctly warns that
`prompt_lcp_tokens` (19,708) is not the cache-hit claim — the 18,432 figure comes from response
metadata, as the card requires.

**The e2e difference is not a clean speedup measure**: the two produced different output lengths
(57 vs 100 tokens) under natural stopping. The defensible statement is the cache-hit fraction, not
a 2.2x latency claim.

## Batching — real, and server-side confirmed

Four distinct captured prompts, identical set at each concurrency, two repetitions with reversed
order.

| concurrency | wall (rep 0 / rep 1) | aggregate tok/s | GPU-s per completed request |
| --- | --- | --- | --- |
| 1 | 30.15 s / 30.01 s | 34.0 | 7.52 |
| 2 | 18.58 s / 17.59 s | 55.1 / 58.2 | 4.65 / 4.40 |
| 4 | **10.86 s / 10.93 s** | 94.3 / 93.7 | **2.71 / 2.73** |

C4 cuts finite-set elapsed time by **64 %** versus C1, reproducibly across reversed order. That
clears the card's ">=10 % repeated reduction" bar.

**The card requires observed server overlap, not client concurrency.** Extracted from
`metrics.jsonl` (723 samples), `sglang:num_running_reqs` within each batch window:

| set | max running | mean running |
| --- | --- | --- |
| C1 | 1.0 | 0.81 |
| C2 | 2.0 | 1.43–1.55 |
| C4 | **4.0** | 2.57–2.62 |

The server genuinely ran 2 and 4 requests concurrently. Corroborating: per-request queue time rose
from ~0 at C1 to 1.7–3.1 s at C2/C4, and per-request decode fell from 38–41 to 30–32 tok/s — the
signature of shared decode, not serialization.

## What this does not establish

- **No deployment change.** `promotion: false`. Per the card, throughput gain warrants considering
  host/deadline qualification, not selecting it.
- **The batching set bypasses the host.** It is isolated HTTP concurrency against the server, not
  qualification of the packaged host's admission path. The `decide()` semaphore was not exercised,
  and separate local evidence shows it does not serialize requests anyway.
- **Prefix contamination within the finite set is present and expected**, e.g. case-5 saw 8,192
  cached tokens at C1. The card says reuse within a set is observed, not disabled. Same-game
  captured prompts share more prefix than independent games would.
- **The 95k input is mechanical stress**, not a natural Actor context.
- **Traces are not saturation evidence.** No hardware counters; kernel-duration share is not
  bandwidth or arithmetic saturation.
