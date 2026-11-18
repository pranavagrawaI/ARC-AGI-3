# Temporal expert locality: completed CPU probe

There is useful routing locality, but a half-size expert bank chosen from one earlier window misses 14–28% of later routed occurrences. Omitting those cold contributions changes layer-0 output by 33–35% normalized RMS. This does not establish a serving advantage or measure ARC quality.

The one capped local CPU run completed in 17.70 seconds. No GPU, model/API calls, Kaggle launch or download were used.

| Bank | Uniform random expected route coverage | Earlier-window bank on held-out windows |
|---|---:|---:|
| 128/512 | 25% | 41.25–70.75% |
| 256/512 | 50% | 72.00–86.25% |
| 448/512 | 87.5% | 92.00–96.25% |

The predeclared 256-bank >=90% screen failed in all six layer/window pairs. More frequent updates may help: a bank chosen from windows 8+16 covers 73%, 85.25% and 87% of window 32 routes in layers 0/24/47. A same-window lookahead oracle covers every routed occurrence with 256 experts in each window. That oracle uses future information and is not evidence for a deployable cache.

For layer 0, independently reconstructed full FP32 outputs agree with preserved GPU FP32 references at normalized RMS error 2.46e-7 and 2.56e-7. The omission diagnostic has normalized RMS errors 0.34555 and 0.32846; individual-row relative L2 errors reach 97.2% and 95.4%. This is meaningful cold contribution, not a harmless numerical rounding difference.

Interpretation: caching must still execute cold experts, as Strata does. Simply omitting an expert because it was previously cold is not supported by this probe. This is not REAP: there is no saliency calibration or rerouting among the surviving experts. It neither overturns Arm P nor tests the published Coder model. A full GPU already holds all experts, so the benefit of an offload system would need to come from a binding memory constraint or some other measured engine difference.

Evidence is in `EXPERIMENT-CARD.md`, `routing.jsonl` (all bank memberships and train/test count vectors), `RESULT.json`, two contribution NPZs, `source-hashes.json`, `verification.json` and `run.log`. The nine inputs are preserved real S10 speculative verification captures: 40 rows, top-10 routing, three layers and three nearby windows from one workload. Rows are correlated, and longer-term/game-specific locality remains unknown.

Reproduce in a fresh experiment directory because `probe.py` refuses to overwrite a run identity. The original invocation was:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 timeout --signal=TERM --kill-after=10s 900s /home/pranav/projects/arc-agi-3-reference/nooa/.venv/bin/python -B analysis/expert-locality-screen-20261011/probe.py
```

The next comparison that could establish an advantage is a matched full-engine replay and then gameplay: identical recorded ARC prompts, context/admission/output budget, measured effective concurrency, cold-start and warm-turn wall time, plus a paired test of action quality. When formats differ, describe it as a system-package comparison and preserve each quantization identity. Do not infer retained capability from latency, routing mass or a layer-output norm.
