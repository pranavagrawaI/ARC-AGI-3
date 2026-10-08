# Serving headroom: completed, intended comparison unresolved

No qualified serving advantage was established. The notebook completed all 13 batches on one actual RTX PRO 6000 Blackwell Server Edition (97887 MiB), producing 127 fixed-length responses and 455168 tokens. Collected 300 output files; every one of the 298 files declared in the runtime hash manifest matches. Startup succeeded after 539 seconds. The server completed the final block and was stopped at approximately 2026-10-08 10:08:24 UTC. The collector observed COMPLETE after this host resumed at 14:45 UTC.

| Condition | Whole-batch tokens/s | Maximum observed running | Maximum observed queue |
|---|---:|---:|---:|
| N10B | 681.4 | 10 | 0 |
| N10C | 684.8 | 10 | 0 |
| N16A submitted | 514.6 | 12 | 7 |
| N16B submitted | 577.0 | 12 | 4 |

These are **post-run descriptive whole-batch rates**, calculated as raw response completion-token sum divided by recorded batch wall time. They include admission, prefill and finishing tails. The larger batches delivered approximately 15–25% fewer tokens per second than the two warm controls, but they used six additional prefixes and had cache misses; this is not an isolated decode-speed comparison or a gameplay result. N10A also had a cache miss and delivered 629.1 tokens/s. Both warm controls satisfy the 90% cache gate; neither N16 batch does. Zero requests were retracted, which does not imply absence of queueing or cache eviction.

## Two implementation failures

1. **Wrong counter for in-flight throughput.** The pinned runtime increments `sglang:generation_tokens_total` in `observe_one_finished_request`, alongside prompt/cache counters. The driver required positive generation while every request in the submitted batch was still pending. The raw events reproduce zero generation-counter delta in those intervals, even though completions later report the full output. All frozen plateau measurements are invalid. A stored U=0 is a measurement failure, not zero model throughput. The same condition guarded `/start_profile`: zero profile requests were emitted, so no native CUDA traces were collected. This is not a Torch/CUDA graph profiler failure. The mock/planted tests did not model the actual counter's completion-only semantics.

2. **Requested concurrency did not equal effective concurrency.** The launcher requested 16 and `/server_info` still exposes `max_running_requests=16`. Its internal state explicitly reports `effective_max_running_requests_per_dp=12`. Server log line 60 states: the Mamba pool has 60 slots and requires five slots per request, capping concurrency at 12. Line 104 confirms effective maximum 12. Raw metric samples never exceed 12 in the N16 batches. The probe therefore never tested sixteen simultaneously running requests. Prelaunch verification checked the flags, but missed this resolution rule.

The frozen card, source and automatic no-superiority decision are unchanged. The automatic RESULT.md table's invalid zero U values should be interpreted using this diagnosis. Operational completion does not establish comparison validity or an expert critical-path fraction.

## Reproduced seam and next decision

The exact pinned `_calculate_mamba_ratio` and `resolve_max_num_reqs` methods were extracted and executed without altering their code, with the recorded overlap/extra-buffer settings: ratio five, cache60 -> cap12. With fixed hypothetical KV capacity, cache80 -> cap16. That removes this admission clamp only; a larger state pool spends VRAM and can reduce KV capacity, so it is not a qualified configuration or a promised speedup.

Before another GPU run, exercise the actual metric semantics and reject an effective-cap mismatch. The runtime provides `sglang:realtime_tokens_total` with a `mode` label and a decode update path; any future reducer must select decode explicitly rather than sum prefill modes. Trigger a short native profile independently of completion-counter progress. First establish the incumbent's actual kernel breakdown; only then test a larger Mamba/KV envelope if it can retain the intended contexts and cache behavior.

Retain the incumbent. No model promotion, competition submission or relaunch was made. Expert optimization upside and genuine N16 decode throughput remain unmeasured.

Primary artifacts: `raw/headroom/server.log`, `raw/headroom/server-info-before.json`, all block requests/responses/events, runtime hash manifest, `collection.json`, `audit.json`, and `POST-RUN-AUDIT.json`. Raw output remains preserved locally under the ignored `raw/` directory.
