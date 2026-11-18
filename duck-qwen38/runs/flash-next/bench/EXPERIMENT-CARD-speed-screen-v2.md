# speed-screen-v2
Question: Does any stack of speed-only serving changes on P raise sustained serving throughput by ≥10 % on a
Duck-shaped replay without changing what the model computes? The candidates are MTP index sharing, prefix
caching in align mode, and MTP length.
Posture: evaluation (try to break the candidate). This retries speed-screen-v1, which was INVALID; see "Why v2" below.
Contrast: S0 = P is the baseline, run in the same session on the same GPU. The stacked configs are:
- S1 = S0 + `TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION=1`.
- S2 = S1 + `TAAF_VLLM_ENABLE_PREFIX_CACHING=1` (vLLM selects mamba `align`).
- S3 = base + MTP k. The base is S2 if its prefix-cache logprob check passes, else S1, else S0. k=2 if the
  base's third-position acceptance under load is < 0.35, otherwise k=4 with cudagraph capture 60.

Everything else is P's frozen notebook, cells 0–10 byte-identical: k448 pruning, 12 GiB KV, 12 seqs, 8,192
batched tokens, MTP3.
Prediction: S2 beats S0 by ≥10 % in generation tok/s, because removing re-prefill frees the ~25 % of decode
that P's live logs show prefill consuming. S1 is within ±3 % of S0. The S3 direction is unknown.
Measure: `speed-screen.json` under `configs.<S>.load.windows[*]`: `gen_tok_s`, `requests_s`, `prefix_hit_rate`,
`accept_rate_by_pos`, `running_mean`, `kv_usage_mean`, `gpu_mib_max`. Raw request records and 5 s
`/metrics` samples are in `bench/raw-<S>.json`; server logs are in `bench/server-<S>.log`. `configs.<S>.logprobs`
and `decisions` hold the correctness check. The verdict comes from `bench/assess_screen.py`, applying the rule
in `CARD.md` (v1) unchanged: integrity, then median over 5 windows, then a subset preference within 3 %; pass
needs P+ ≥ 1.10 × S0 in median gen tok/s and in median requests/s, with ≥ 4/5 windows above S0's median.
Cap:
- One Kaggle GPU session; kernel `pranavagrawai/duck-flash-next-speed-screen`, version 2.
- Expected ≈1.4 GPU h (v1: 1.42 h), hard ceiling 2.0 GPU h. Quota is 7.06 h before push, and validation needs
  ≈5.05 h.
- Load is 180 s warmup + 600 s measurement per config. No games, no submission, $0.
Stop:
- Success or futility: the notebook finishes S0–S3 and tears the server down. There is no early stop on a good
  or bad number.
- Abort: if the kernel is still RUNNING 2.0 h after it starts, stop it in the Kaggle UI. The CLI has no
  cancel verb.
Invalid if:
- S0 fails integrity: it did not run, the pruning guard failed, client errors exceed 1 % after warmup, or
  windows are missing.
- Mean prompt tokens during measurement are < 15K. The replay would then no longer resemble P's ~21K live
  prompts, which is exactly v1's failure.
- `tokenize_fallback` exceeds 5 % of counts.
- A config that fails integrity is excluded from the choice; it does not invalidate the run.
Result map:
- Screen PASS: build P+ = P + the winning overrides (`bench/build_pplus.py`), write the validation card, and
  run two full passes this week.
- FAIL: keep P, run no validation this week, and record which component failed and why (no gain, flag not
  live, or logprob divergence). The chunked-trim and k384 arms move to next week's quota against P.
- INVALID: keep P. Do not retry this week unless the cause is again a driver defect proven offline; that
  would be speed-screen-v3 with a new card.
Outputs: `~/projects/arc-agi-3-reference/duck-qwen38/runs/flash-next/bench/out-v2/` (the downloaded kernel
output), `bench/execution/` (push log, status, quota), and `bench/out-v2/screen-assessment.json`.

## Why v2 (v1 post-run audit; the v1 card `CARD.md` is unchanged)

v1 (kernel version 1, 1.42 GPU h) returned INVALID: about 5,300 client errors against ~590 completions in every
config. Raw output is in `bench/out/`. The causes were driver defects, all reproduced and fixed offline against
`bench/mock_server.py`:

1. `/tokenize` undercounts the chat prompt by up to ~510 tokens, and `max_tokens = 32768 − est − 512`
   overflowed by one token.
   - Each 400 then dropped history, so sessions collapsed to ~6K-token prompts.
   - **Fix:** a 1.03·est + 1,024 margin, and on an output-length 400 a retry using the server's own prompt
     count. History is dropped only when the prompt itself overflows.
   - Sessions are pre-grown to the turn budget: the snapshots are 4–18K tokens because levels reset history.
2. T=0 output is not reproducible on this stack. The same config's repeat diverged after 10–50 characters on
   4/4 prompts.
   - **Fix:** next-token top-20 logprobs on 16 identical token-id prompts via `/v1/completions`.
   - S2 passes if, warm-cached against S1 cold, top-1 mismatches ≤ floor + 1 and mean |Δ top-1 logprob| ≤
     3 × floor + 0.02. The floor is the max of S1's own repeat and S0 vs S1.
   - A mock with a corrupted cache is detected.
3. `serving_setup.start_server` races: it reads `/proc/<pid>/cmdline` before `exec`. **Fix:** retry that
   specific error up to 3 times; the launcher terminates the stray child.

Offline tests are in `test_speedbench.py`, `test_orchestrate.py` (normal, prefix crash, and corrupted prefix
cache), and `test_assess_screen.py`.
