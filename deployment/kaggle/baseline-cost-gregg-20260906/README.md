# Baseline cost investigation — Gregg revision

Locally prepared successor to `baseline-cost-20260906`. Original package and live solver sources were not edited. `SOURCE-PIN.json` identifies the source snapshot copied before this revision. No upload or GPU run is authorized by local qualification.

The package contains `baseline_cost.ipynb`, `kernel-metadata.json`, and `manifest.json`. It preserves the frozen Actor replay bootstrap, Qwen3.8-27B FP8 checkpoint, pinned engine and orderly lifecycle. Target remains offline RTX PRO 6000. MTP stays disabled.

## What changed

- Four-length cold-cache diagnostic remains 1024/10240/21279/95000 tokens, three repetitions. Graph/kernel warmups are separate from cache warming.
- Both short and 95k production-path traces are required to contain GPU kernel events. Profiling timings never enter the speed comparison; CUDA graphs remain enabled.
- Batching now measures two finite sets at concurrency 1/2/4, with two order-reversed repetitions per set. The first uses four distinct captured rendered requests; the second mechanically repeats each distinct prompt to 95000 tokens. The latter is a long-context cost diagnostic, not natural conversation or a quality test.
- Completion timing starts when the entire finite workload is ready, so executor waiting cannot disappear from deadline accounting. Reports also retain dispatch-to-completion time. The 180-second common-ready threshold is an explicit diagnostic reference to the deployed request timeout, not a game deadline or a guarantee of useful actions.
- Incremental batch event files preserve readiness, dispatch, completion and failure evidence even when a trial aborts. Already running requests may finish after admission stops.
- Missing, nonpositive or nonfinite server decode timing fails a fixed control. HTTP data arrival is labeled separately from generated-token timing.
- Natural requests preserve their captured bodies and stopping behavior. Replay-to-schema-checked timing is recorded separately from HTTP time; no tool is executed.

The 100-action solver run spent 39.5% of summed Actor request wall time above 64k input tokens, including 9.0% at or above 95k. This supports a long-history diagnostic; it does not establish context as the cause of that time. Reproduction: `analysis/competitor-transfer-20260906/performance_workload.py`.

## Request overlap is possible, but serving capacity is unqualified

A CPU-only probe used the unchanged packaged LocalSolver, worker and headless actor with actual NOOA and one loopback scripted model server. A successor observation woke game A's actor outside the `decide()` semaphore; game B then sent its request while A's remained in flight. Peak HTTP overlap was two, with separate session routing. Report: `analysis/competitor-transfer-20260906/runner-overlap/REPORT.md`.

This disproves treating the existing semaphore as a global request-C1 guarantee. It does not establish SGLang batching, KV capacity, GPU throughput, or real-game deadlines. This notebook controls its own HTTP request set and never changes the deployment semaphore. Real integration must count request ownership and overlap at the actual request boundary, including post-observation wakeups and summaries.

## Local qualification

```bash
python3 build.py
python3 build.py --check
python3 -m unittest discover -s . -p 'test_*.py'
```

Run in this directory. Tests exercise real loopback streaming HTTP, missing timing and truncation rejection, finite concurrency, common-ready deadline accounting, no replacement admission after a known failure, complete experiment branching, unchanged natural request bodies, and exact embedded notebook assets. They cannot qualify the target GPU, profiler availability, cache telemetry or runtime speed.

Read EXPERIMENT-CARD.md before any future launch. Download the whole output directory, preserving partial failures. `MEASURED_REVIEW_REQUIRED` is not a performance win or approval to alter a backend. Select one intervention only after reviewing timing, cache state, overlap, traces, and attainable wall-time savings.
