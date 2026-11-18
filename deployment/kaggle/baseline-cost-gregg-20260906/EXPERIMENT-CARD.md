# baseline-cost-gregg-20260906

Question: What baseline request cost and finite-workload concurrency tradeoff justify one next optimization that improves deadline-qualified useful work?

Posture: Local preparation and qualification only. Prospective target experiment measures cost; kernel resource attribution and real Actor acceptance require the evidence specified below. No backend is presumed to be the bottleneck.

## Frozen comparison

Use the Actor replay's verified checkpoint, pinned engine, launch configuration and generation defaults. Baseline only: no MTP, sampler repair, backend mutation, graph disabling or host semaphore change. Natural requests preserve sampling and output limits exactly, including omitted limits. Record the resolved server configuration and model generation_config.json.

1. Obtain exact server-rendered IDs for captured cases 0/5/12/13, with natural stopping and cold prefix cache. Save every response, including invalid tools and short completions.
2. Repeat chronological cases 12 then 13 without a cache flush between them. Compare successor cache metadata against its cold natural request. This is historical-prefix reuse, not a claim that the new seed response authored the frozen successor.
3. Warm the four diagnostic shapes, flushing request cache separately. Require short and long serving-worker traces at 1024 and 95000 tokens, production CUDA graphs retained. Reject missing GPU kernel events. Do not use profile durations as unprofiled speed measurements.
4. Fixed-output diagnostic: 1024/10240/21279/95000 actual input IDs, three alternating-order repetitions, 256 generated tokens. temperature1/top-k20/top-p0.95, ignore_eos true. Inputs are explicit prefix truncations/repetitions; forced continuation is diagnostic only. Verify token counts and record effective start/end contexts.
5. Finite batching: four distinct captured rendered prompts, and a separately labeled four-prompt 95000-token mechanical set. For each set, identical requests at C1/C2/C4, two repetitions with reversed concurrency and workload order. Cold cache before each finite set; prefix reuse within it is observed, not disabled or assumed absent. This is a finite workload, not steady-state saturation or a representative distribution of independent games.

## Timing and cache contract

Record raw server metadata, total/cached/new input counts, stream receipts, actual output counts, client start/end, first HTTP data arrival, and terminal response. Missing cache fields remain unknown. Missing/invalid decode timing invalidates fixed controls.

The inherited server rate is `(N-1)/(tokenizer finished_time-first_token_time)`. It excludes pre-first-output work but includes tokenizer output handling; it is neither total HTTP throughput nor pure GPU decode latency. Use metadata's forward-entry/prefill-finished and queue timing only where actually exposed. Never relabel HTTP chunks as generated tokens. Retain unknown timing boundaries honestly.

Each finite workload has a common monotonic ready time. Save admission waiting, dispatch-to-completion and ready-to-completion per request; aggregate throughput is total generated tokens/whole-set elapsed time. Count completions within an explicit180-second common-ready diagnostic budget, without changing the observation watchdog to censor late completions. This is not actual game-level deadline qualification.

Natural replay records HTTP time and replay-to-schema-checked time. It does not reconstruct live context-render time, execute tools, or measure time to a usable game action. Those stages must be timed in the bounded real Actor validation after an intervention is selected.

Sample server metrics and GPU clocks/power/memory separately. Require observed server overlap for a batching claim; client concurrency is insufficient. Inspect waiting requests, KV use, eviction/recomputation and preemption evidence. Missing metrics cannot become zero events. Short metric sampling can miss overlap, so absent sampled overlap remains unresolved rather than proving serialization.

## Bounds and failure handling

Prospective one user-launched offline RTX PRO 6000 notebook. Runner5400seconds, outer runner5700seconds; bootstrap retains its inherited subprocess bounds. **At most72 explicit generation requests:** six natural, four16-token shape warmups, twelve256-token context controls, forty-eight256-token batch controls, two32-token traces; plus documented inherited server warmup/lifecycle probes. This increases the earlier48-request diagnostic count but does not increase its wall-clock cap. Long-context capacity or time may terminate the run before all contrasts complete; preserve partial evidence.

No game actions, tool executions, external paid APIs, competition submissions, automatic retries or automatic optimization. No launch during this local task. Natural observations retain900-second watchdog; fixed HTTP read timeout600seconds. These are collection bounds, separate from diagnostic deadline eligibility.

Abort on engine death, transport failure, unknown retirement, missing terminal response, truncation, fixed token mismatch, invalid server timing, manifest drift, invalid profiling or cap. Never admit replacement work after an observed failed completion. Existing in-flight work is handled by containment, not silently reused. Qualification is per contrast; a later failure does not erase earlier raw evidence.

## Decision rule and one intervention

Report medians/ranges and sample counts. A repeated >=10% finite-set elapsed-time reduction without failures warrants considering host/deadline qualification; it does not select deployment. Retain single-request latency and completion distributions alongside throughput. Timing-only improvement with unacceptable deadlines is not a useful win.

Use a production-faithful trace to locate costly stages. Kernel duration alone cannot distinguish bandwidth, arithmetic or poor small-shape efficiency. If that distinction changes the proposed intervention, acquire a small targeted counter/shape trace before selection. Do not import backend support from a different GPU or unpinned SGLang release.

Estimate attainable request/decision savings before implementing one change: with affected wall-time fraction f and component speedup s, predicted total speedup is 1/((1-f)+f/s), only where the fraction represents the critical path rather than summed overlapping operations. Candidate examples are supported attention-decode change, selected FP8-kernel change, graph/launch repair, or cache/request-stage correction. No candidate is preselected.

Stage2 is a separate bounded follow-up chosen from those results: one change, unprofiled matched retiming, then real Actor natural-stopping validation with unchanged reasoning/sampling policy and explicit deadlines. Full decision measurements include context rendering, request queueing, all model/tool rounds and the usable action. No claim that the present offline replay completes Stage2.

MTP remains parked. Its measured long-prompt loss is sufficient to retain baseline; cause remains unknown. Reopening requires matched speculation telemetry/trace, committed-token denominators and preserved sampling policy.

## Evidence

Outputs: `/kaggle/working/baseline-cost-gregg-20260906/` and inherited bootstrap records. Preserve request bodies, input-ID hashes, raw responses/timings, cache telemetry, server log, production traces, launch/model identities, and failures. Download to `analysis/deployment-probes/20260906-baseline-cost-gregg/raw/`. Do not rewrite original run status or raw data.

Useful-work acceptance, target profiling/counters, actual GPU overlap and game deadlines remain unqualified locally. The host CPU probe establishes HTTP overlap only; source semaphore presence is not a C1 control.
