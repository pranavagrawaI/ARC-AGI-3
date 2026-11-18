# FLARE stage-3 smoke: addendum to SMOKE-CARD.md (frozen before launch, 2026-10-20)

Only the package route changes. 2a, 2b and probe 4 all died rebuilding 107 GB on local disk, while
probe 5's in-place reads survived. So the package is now a single file in the private dataset
`pranavagrawai/flare-gb202-package-20261020`, and Flare reads it from /kaggle/input. Everything else is
2b unchanged: shim, flags, trust fingerprint, memory watchdog, the six-row functional check, the
S=10/14/16 replay, the estimators and the SGLang M3A reference (U_overlap 477.6). The watchdog line
now also logs Flare's /proc read counters, so the load rate is measured.

## Route
- Upload: the laptop opened a Kaggle blob session (`../flare-dataset-20261020/session.py`). The CPU
  kernel `flare-gb202-upload-20261020` PUT the six verified part outputs into it, and its sha256 over
  the committed bytes must equal the LFS hash c3db5e1a…. The dataset was created from the blob token
  (`create_dataset.py`), and its description opens with the LIL-1.0 Attribution Notice. Private.
- Kernel `pranavagrawai/flare-smoke-3-20261020`, RTX PRO 6000, self-capped at 110 min. 21.9 h of weekly
  GPU quota was left on 10-19; this run costs at most 1.9 h.

## Flag change: prefix cache on (amended 06:06Z, before launch)
`flare run --help` says the prefix cache is disabled by default (`--prefix-cache-memory`), and 2a/2b
never turned it on. The SGLang M3A reference ran with the incumbent flags, i.e. with its radix cache
(predicted replay cache fraction 0.92, production 0.935). An uncached Flare would be measured
re-prefilling ~100K-token contexts on every turn, and that measures the configuration, not the
runtime. So stage 3 adds `--prefix-cache-memory 64G`; host RAM is PLE 26.8 GiB + 64 GiB, well under
the 160 GiB watchdog. If Flare won't start with it, there's one recorded fallback without the cache.
An r from that fallback is a lower bound: it can still call GO, but a KILL or PAIRED from it is
INVALID for the decision. The replay's `cached_fraction_block` is reported beside r.

## Decision: SMOKE-CARD.md's rule, unchanged
INVALID if Flare is never healthy (cause from the log, one retry if it can be worked around). KILL if
fewer than 3/6 functional rows parse, replay errors exceed 2%, or r < 1.10. PAIRED if 1.10 ≤ r < 1.30.
GO if r ≥ 1.30.

## Added: load time, reported but not part of the decision
- L = seconds from Flare start to health, with the read rate from /proc/<pid>/io. A single
  /kaggle/input stream ran at about 79 MB/s on 10-18, which puts a 107 GB sequential load near 23 min.
- Submission arithmetic: r_eff = r × (532 − L_flare/60) / (532 − L_sglang/60) over the 532-min hidden
  run. L_sglang comes from the control's startup logs before r_eff is quoted. If L_flare exceeds 40 min,
  the 2b wait limit, the result is INVALID (load time) and the next step is a load-path fix, not a pair.

## After a GO: the arm's base is F1x (added 06:10Z)
D10 F1x scored 30.88, a valid run (z ≈ 1 against the control mean of 27.8; D8's control scored 31.31).
That shows F1x is safe, not that it is better. F1x is now the fallback, so the arm is F1x with Flare
serving against F1x itself: the server is the only difference. The smoke still replays the 10-01
control traffic, because the SGLang reference was measured on that replay. The preload piece is
sandbox-only and carries over unchanged. The 16K output cap reaches the server: 116K in + 16K out,
with the CTX at 140K. Flare has no fixed context cap ("context is limited by the checkpoint and
available cache capacity", from the package metadata), and its KV pages trade against
--max-concurrency. So before the arm is built, the smoke's server log must show that 10 streams of
140K fit.
