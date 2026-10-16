# m2-f1 screen-1 (2026-10-16)

**Question:** Do the friction fixes fire, and do they avoid harming levels, against B1 in a matched 60-minute public-25 run?

**Posture:** Screen under the plan agreed on 2026-10-16. A piece is admitted if its mechanism visibly fires and it shows no harm (≤ 1 sd below the paired control). Score cannot see a gain this small.

**Arms:** `screen-1/f1` = B1 (the D5 kernel's notebook, sha da5bef…) + F1. `screen-1/b1ctl` = B1. The two notebooks differ only in F1's install cell and three config lines; `build.py` asserts this. Both arms stop 60 min after server startup, set by `soft_end` in the interactive branch only. Games keep their full-run per-game caps, so the first 60 minutes match a full run.

**F1 pieces (evidence: solvers/m2_f1/README.md):**
- **preload:** the sandbox preloads bisect, collections, copy, functools, heapq, itertools, json, math and statistics. In B1 pair-1, about 8 NameErrors per run came from forgotten imports.
- **out16k:** `LOCAL_ANALYZER_MAX_OUTPUT` 12K→16K, with window and server context each +4K, so the input budget is unchanged at 116K. In B1, 24–35 responses per run hit 12,288 tokens, almost all with no tool call.

**Absent by declaration (`incumbent/check.py --without`):** c14, server-fail-fast and reap448 are not in the B1 notebook that was drawn as D5. F1 is a pure delta on that notebook.

**Prediction:** F1 shows 0 NameErrors on preloaded module names (b1ctl shows a few). F1 shows some responses ending between 12,289 and 16,384 completion tokens with a tool call; b1ctl can't. Levels: no detectable difference.

**Measures:** `analyze.py` computes:
- Levels at the end of the run, summed from `benchmark.json` `levels_completed`. Both arms get the same 60 wall-minutes.
- Mechanism counts from `*_requests.jsonl`.
- Tokens and actions.

**No-harm rule (pre-registered):** F1 fails if `levels(f1) < levels(b1ctl) − 7`. 7 levels is 1 sd of a paired difference at minute 60. It comes from two replicate pairs, B1 79/88 and c14 78/73, giving single-run sd ≈ 5. 60-min control replicates will refine it.

**Mechanism rule:** both of the following, plus no crashed games and both arms COMPLETE:
- preload: zero NameErrors on preloaded module names in f1.
- out16k: at least one f1 response with 12,288 < completion_tokens ≤ 16,384, OR every f1 length cutoff at 16,384 (the cap applied).

**Result map:**
- **Both rules pass:** F1 enters probation on B1. It's queued for a hidden draw after the scheduler and mining candidates are known, and may stack with them.
- **No-harm fails:** split once. Rescreen preload and out16k separately against b1ctl (about 3 GPU-h). A half that still fails is dropped.
- **Mechanism fails:** inspect before any rerun.

**Cap:** one launch per arm, hard cap 6,000 s each, 2 slots. Admission requires quota used + 3.34 h ≤ 30 h, and both slots free. No retry except for a proven infrastructure failure. No submission from this experiment.

**Note:** b1ctl is also the first 60-min control replicate in the 10 h/week replication budget.
