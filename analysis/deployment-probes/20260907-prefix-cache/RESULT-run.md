# prefix-cache-chunk-abba-20260907 — all four arms measured; the run failed on a log line

`pranavagrawai/prefix-cache-chunk-abba-20260907` (note: the **title** slug, not the metadata id).
Pushed 22:35:56Z, terminal **ERROR**. All four ABBA arms completed with 148 files and 37 result
files each. The failure is a post-hoc classification artifact, not a measurement failure.

## What actually failed

`FAILURE.json` says only `RuntimeError: Lifecycle failed`, raised at `actor-replay.py:213` after the
**fourth** arm finished its work. The cause is one teardown log line in arm 3:

```
[2026-09-06 23:17:39] INFO:     Waiting for background tasks to complete. (CTRL+C to force quit)
```

classified `forbidden_unclassified_log_line` under rule `default_deny_unmatched_nonempty_line`. It is
a benign uvicorn shutdown message that is not in the teardown allowlist.

The teardown itself was clean. Arm 3 records `containment_passed true`, `root_returncode 0`,
`term_only true`, `quiet_window_valid true` — only `log_classification_allowed` is false. Arms 0, 1
and 2 pass every check with `forbidden_count 0`.

Deny-by-default on teardown lines is the right posture; this is one unlisted benign line, and the
fix is to classify it, not to loosen the rule.

## The assessor cannot process real output

`assess.py` never ran on Kaggle (the runner raised first). Running it locally against the downloaded
evidence fails twice, independently:

1. `path.glob('batch-*.json')` matches **36** files per arm, not 4 — it also catches
   `batch-captured-1-case-0-request.json` and every other per-case file, so
   `len(batches)!=4` raises `ValueError('Incomplete arm')`.
2. Arm 3's `lifecycle.json` status is `failed`, so `raise ValueError('Unqualified lifecycle')` would
   fire regardless.

The 18 local tests passed because their fixtures contain only the summary files. This is a real gap
between fixture layout and run layout, and it would have blocked assessment even on a fully clean
run.

## Chunk 2048 vs 4096 — measured, and below the decision bar

ABBA order 2048/4096/4096/2048. Repeatability is excellent: same-config arms agree to ~0.1 %.

| workload | A0 2048 | A1 4096 | A2 4096 | A3 2048 |
| --- | --- | --- | --- | --- |
| captured C1 | 30.09 s | 31.13 s | 31.14 s | 30.19 s |
| captured C4 | 11.60 s | 11.00 s | 11.00 s | 10.70 s |
| 95k C1 | 134.11 s | 130.05 s | 130.08 s | 133.99 s |
| 95k C4 | 103.56 s | 98.87 s | 98.87 s | 103.52 s |

4096 is ~3 % faster on 95k C1 and ~4.5 % faster on 95k C4; on the captured set it is ~3 % *slower*
at C1 and mixed at C4. **Nothing approaches the card's 10 % bar**, and the sign flips with workload.
No configuration is selected; `promotion` stays false.

Resolved policy was identical across all four arms — same `mamba_backend`, `mamba_radix_cache_strategy`,
`uses_mamba_radix_cache`, and every CUDA-graph setting. The card's concern that changing chunk size
might change graph shape or Mamba policy is **not** borne out here; only `chunked_prefill_size` differs.

## The substantive finding: cache reuse is quantized to chunk size

| pair | condition | 2048 arms | 4096 arms |
| --- | --- | --- | --- |
| `normal_long_12_13` | cold | 9.44 / 9.45 s, cached 0 | 9.46 / 9.46 s, cached 0 |
| `normal_long_12_13` | predecessor-warmed | 7.19 / 7.06 s, **cached 18,432** | 7.49 / 7.49 s, **cached 16,384** |
| `normal_short_0_1` | cold | 7.49 s, cached 0 | 7.50 / 7.51 s, cached 0 |
| `normal_short_0_1` | predecessor-warmed | 6.64 s, cached 8,192 | 6.62 s, cached 8,192 |
| `mechanical-95000` | cold | 34.71 / 34.70 s, cached 0 | 33.49 s, cached 0 |
| `mechanical-95000` | predecessor-warmed | 32.44 s, **cached 18,432** | 31.52 s, **cached 16,384** |

Cached tokens land on chunk multiples: 18,432 = 9 × 2048 and 16,384 = 4 × 4096. **The larger chunk
retires less of the same shared prefix** — 2,048 fewer tokens on both long pairs — because reuse is
quantized to chunk boundaries. On `normal_long_12_13` that shows up in latency too: 7.06–7.19 s at
2048 versus 7.49 s at 4096, the one place where the two effects point the same way.

Predecessor warming helps consistently and independently of chunk size: 9.44 → 7.19 s (~24 %) on the
long captured pair, 7.49 → 6.64 s (~11 %) on the short pair, 34.7 → 32.4 s (~7 %) on 95k mechanical.

## Limits carried forward

Fixed at 256 output tokens, so these are equal-work comparisons; natural output and schema results
stay separate. The 95k inputs are mechanical repetitions, not natural contexts. Real
summary-boundary request bodies remain absent and nothing was fabricated. Captured cases 12/13
returned unoffered tools in the source capture — HTTP 200 is transport success, not good Actor
behavior. No gameplay, no deadlines, no host qualification, no promotion. MTP stays parked.
