# FLARE stage-3e smoke: result (kernel flare-smoke-3e-20261020, notebook 15:56–16:24Z), FLARE CLOSED

All images were replaced by a fixed placeholder, so prompts were prefix-stable. Flare healthy after
145 s; functional 6/6; 0 errors.

| S | responses in 480 s | U_resp | U_overlap | mean latency | cached fraction |
|---|---|---|---|---|---|
| 10 | 180 | 518.3 | 500.8 | 22.5 s | 0.941 |
| 14 | 168 | 523.2 | 505.6 | 36.4 s | 0.913 |

- The snapshot diagnosis from 3d is confirmed: with stable prefixes the cache hit rate rose from
  0.37–0.42 to 0.91–0.94.
- **r = 1.05 (U_overlap 500.8 vs 477.6; U_resp 518.3 vs SGLang's 518.2).** This is Flare's best case:
  its cache working, no image tokens, the shortest prompts, all biases in its favour. It only ties the
  incumbent SGLang serving full images. Throughput is flat from S=10 to 14 (~500), so concurrency
  can't buy more.
- Card rule: r < 1.10 → **FLARE closed**. With images, which dfranzen's harness uses, it is 0.45 (3d),
  because of the 4-image limit plus request-boundary snapshots.

The SGLang same-workload reference (sglang-keep0-ref-20261020) gave no data. It landed on a **Tesla
T4**, not the RTX PRO 6000 its metadata asked for, while smoke 3d held the RTX. SGLang then died at load
(cudaHostRegister of the PLE table failed), after ~3.5 h of the session. It isn't needed for this
verdict.

What would reopen FLARE: a Flare build with token-granular (or message-boundary) prefix snapshots and
a higher image limit, together with evidence of better than 1.3× decode on long-context agent traffic.
There's no such evidence today.
