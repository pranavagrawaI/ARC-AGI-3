# critic-live-v1

Question: Can a concise epistemic review at submission make the actual Actor reconsider its inference and then commit a live action, with measurable added latency?

Posture: Local feasibility and uptake pilot. No deployment promotion or Kaggle launch.

Contrast: Baseline, external critic, host-injected self-review skill, in that order. Same frozen v6 runtime, LS20 seven-action recorded prefix, retained notes through Turn 7 and host transition measurements. Fresh Actor context; historical full conversation and helper are not restored. Contingent controller off in every arm. One review at first attempted submission, not necessarily an otherwise valid submission. Skill arm gets no extra model; Actor handles review on its next call. Different sampled proposals may arise, so this is not a paired causal efficacy estimate.

Prediction: Review pauses commitment, Actor responds and resubmits, two new actions can execute. A useful review names a specific unsupported inferential move rather than correcting coordinates or supplying a game solution. An unchanged action can be justified; acknowledgement alone is not success.

Measure: requests.jsonl and responses.jsonl preserve full actual Actor and critic contexts and outputs. pilot_events.jsonl records proposed submissions, feedback and results. actions.jsonl records actual successor grids and level counts. Report observed inference revision separately from action changes and objective progress. Record each request elapsed time, cached prompt tokens when reported, total arm time, usage and cost. No hidden cognition claims. Critic max 384 output tokens; desired <=60-word Actor response. Report overruns, do not truncate evidence.

Cap: $1 cumulative conservative reservation; 25 requests total, at most 8 Actor requests per arm and 1 external critic request globally; 2048 Actor output tokens and 384 critic output tokens; reasoning off, coreweave/fp8 only, no fallback, zero retries. 45s provider timeout, 180s per arm. Two new environment actions per arm, seven replay actions per arm; preparation also replays seven actions locally without inference. At most one review per arm. No action tool execution by critic.

Stop: End each arm after two new actions or terminal. Abort entire card on transport/model error, timeout, exhausted admission cap, runtime drift, critic tool call, invalid critique JSON or reasoning-on output. No rerun under this ID.

Invalid if: Replay mismatch, future evidence exposure, omitted Actor context in critic prefix, action bypass during review, altered ordinary submission gates, leaked critic tool execution, unknown action provenance or budget violation. These invalidate the affected interpretation; preserve all observations. Missing cache metrics do not invalidate uptake but preclude a cache benefit claim. Starting from notes rather than complete historical context is declared scope, not exact historical resumption.

Result map: Specific criticism plus observable appropriate Actor revision and action => retain hook for a larger paired trial, not deployment. Criticism ignored or ritual acknowledgement => improve response contract before broader use. Factual policing or invented errors => revise critic, retain baseline. Skill equally useful in this small sample => prefer its lower mechanism cost provisionally. Incomplete/invalid => repair identified harness boundary under a new card. No level gain in two actions neither proves nor disproves architecture value.

Outputs: This directory. Source: experiments/probes/epistemic-critic-local/live_pilot.py and live_hook.py. Started manifest hashes runner, hook, critic prompt source, seed and card; runtime manifest checked unchanged before launch.
