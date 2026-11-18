# baseline-cost-20260906

Question: What baseline execution cost and finite-workload concurrency tradeoff justify one next optimization?

Posture: Diagnosis, not backend selection or gameplay qualification.

Contrast: Fixed native diagnostic at 1024, 10240, 21279 and 95000 input tokens (three warmed repeats, alternating length order); one captured chronological successor pair cold versus prefix-populated; same four distinct rendered Actor requests at concurrency 1, 2 and 4 in forward/reverse order. Baseline only, production CUDA graphs retained. No MTP, kernel/backend change or host semaphore change.

Prediction: Context-dependent cost or server batching may expose a material opportunity; neither is assumed. A >=10% repeated reduction in finite-set elapsed time without failures warrants host/deadline qualification, not deployment.

Measure: `*-stream.jsonl`, `*-result.json`, `batch-*.json`, `metrics.jsonl`, `server.log`, `trace-*`. Decode throughput is the pinned tokenizer's `(N-1)/(finished_time-first_token_time)`, not pure GPU time. Client monotonic admission-to-terminal response is a separate boundary. Never infer token timestamps from chunks. Record total/cached/new input from raw server metadata; missing telemetry stays unknown. A kernel-duration share is not bandwidth or arithmetic saturation. Do not sum overlapping kernel durations as wall time. Report medians/ranges, not unsupported tail percentiles.

Cap: One user-launched offline RTX PRO 6000 notebook; 5400 seconds runner, 5700 seconds outer runner; bootstrap has inherited bounded subprocess timeouts. At most 48 explicit generation requests (6 natural, 4 shape warmups, 12 context controls, 24 batch controls, 2 traces), plus documented inherited server warmup/lifecycle probes. Fixed controls 256 output tokens, warmups 16, traces 32. Natural requests retain their captured limits and a 900-second observation watchdog. Zero game actions, tool executions, submissions, paid APIs or automatic retries. No concurrent-game deployment changes.

Stop: Finish all measurements and orderly containment; abort on dead engine, transport failure, missing terminal output, truncation, manifest drift, profiler failure or cap. Do not admit replacement work after unknown retirement. Preserve partial evidence on failure. No automatic rerun or intervention.

Invalid if: Wrong GPU/model/engine, prompt count drift, fixed output count mismatch, missing server timing, missing trace for resource diagnosis, unexplained cache state, failed lifecycle, server concurrency unobserved for batching claims. Classify validity per contrast; a profiler failure does not erase earlier raw timings. Invalid tool responses remain recorded outcomes, not dropped samples.

Result map: Repeated context cost growth -> inspect the corresponding production trace; flat decode with growing predecode cost -> inspect prefill/cache/host boundary. Throughput gain -> qualify ownership and real deadlines before allowing host concurrency. No gain -> retain baseline and stop the candidate. Ambiguous -> retain baseline and specify one discriminating follow-up. No automatic optimization even on positive results.

Outputs: `/kaggle/working/baseline-cost-20260906/`, plus inherited bootstrap evidence. Download under `analysis/deployment-probes/20260906-baseline-cost/raw/` without rewriting originals.

Limits: Context inputs are explicitly mechanical prefix truncations/repetitions, not four natural Actor contexts; 95k is stress only. The finite set comes from one captured game, replayed independently; it does not represent independent games' full prefix distribution. Natural calls test schema validity and latency, not usable decisions or gameplay. The cached continuation uses frozen historical messages, not newly sampled output. No claim about summary quality. End-to-end Actor acceptance and hardware-counter resource attribution remain subsequent gates.
