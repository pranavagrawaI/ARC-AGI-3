# Local preparation — no target run

Prepared `deployment/kaggle/baseline-cost-20260906/baseline_cost.ipynb`, its experiment card,
offline competition metadata and deterministic build manifest. No upload, launch or submission.

Verification: 8 unittest checks pass; `build.py --check` passes. Tests cover all 48 explicit
request branches with controlled fixtures, preserved natural chat bodies, the same finite
batch workload, a barrier proving concurrent client execution, actual loopback HTTP streaming,
truncation rejection, dead-engine handling, profiler failure and stopping admission after an
unknown retirement. Notebook cells compile and the embedded runner matches local source.

The original fixed MTP controls are confirmed greedy from retained request JSON. Updated only
the result report's unsupported causal explanation; raw data and slower-MTP disposition unchanged.

Limits: Local tests do not execute the target model or establish GPU trace availability,
performance, kernel saturation, cache reuse, scheduler overlap or lifecycle on Kaggle. The
notebook checks for real GPU kernel events before continuing the main timing sweep. Resource
counters and a production-faithful system trace may still be needed to separate bandwidth,
arithmetic and synchronization. A torch trace with no counters is not a resource diagnosis.

The pinned local host serializes model admission. No host edits were made. These batching
inputs are distinct captured requests from one game, not a certified multi-game workload;
the context sweep is a mechanical stress test. End-to-end Actor deadlines and a useful action
remain integration gates. No speedup or deployment readiness is claimed by this local pass.
