# Baseline cost diagnostic

Upload `baseline_cost.ipynb` with the accompanying `kernel-metadata.json`. It is tied to `arc-prize-2026-arc-agi-3`, requests RTX PRO 6000, keeps Internet off, and retains the frozen donor's model/runtime datasets. No launch is performed locally.

Run `python3 build.py --check` and `python3 -m unittest discover -s . -p 'test_*.py'` from this directory before upload. Do not reuse a changed package under this run identity after launch.

The notebook reuses the frozen Actor replay bootstrap, full model verification and orderly lifecycle. It measures baseline only; no sampling repair is necessary on this PyTorch non-speculative path. Natural Actor requests retain original bodies apart from model routing and evidence-return fields. Fixed controls explicitly force 256 tokens with temperature 1, top-k 20, top-p .95; they do not establish output quality.

Server-side profiler endpoints were checked against the pinned `0.5.19.dev591+g8e04f66a7` wheel. CPU/GPU traces profile the serving worker with CUDA graphs left enabled, separately from speed measurements. Profiling availability on Kaggle remains a target precondition; traces alone do not supply hardware counters or prove a resource limit.

## Deployment feasibility

The current local `deployment/kaggle/runtime/local_solver.py` defines `_MODEL_GENERATION_ADMISSION = threading.BoundedSemaphore(1)` and holds it in its model route. This notebook deliberately bypasses that host for isolated server measurements. Do not remove it based only on aggregate throughput. Independent-game ownership, result routing, deadlines and the exact packaged host must be qualified before integration. No real-game deadline or time-to-usable-decision claim is possible in this replay.

## Reading the result

Keep cache warming separate from graph/kernel warming. Context and batching trials flush only when no request is pending; continuation deliberately does not flush between seed and successor. Inspect actual cached-token metadata and sampled running-request metrics before interpreting either contrast. Same-game captured prompts can share more prefix than independent games.

The existing MTP fixed rate used tokenizer first-output-to-finished timing, not HTTP elapsed time. Both old fixed arms were greedy; their regression is not evidence that top-k/p renormalization caused the slowdown. This notebook makes the stochastic diagnostic policy explicit.

`MEASURED_REVIEW_REQUIRED` is not an optimization pass. Select a single follow-up only after checking validity and estimating attainable wall-time savings. No backend is selected automatically.
