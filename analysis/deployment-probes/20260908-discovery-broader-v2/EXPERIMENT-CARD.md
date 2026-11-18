# Discovery factorial v2 — bounded OpenRouter screen

## Question

With routine reasoning disabled, does a compact host-measured evidence capsule, one fixture-generated executable discovery demonstration, or both get the frozen Qwen 3.8 27B solver into a useful two-transition interaction loop more reliably or cheaply than the fresh common baseline?

## Posture

Bounded candidate screen. This can select a candidate for target-runtime qualification; it cannot establish Kaggle GPU performance, generalized gameplay ability, or a reasoning-on benefit.

## Contrast

- A: common v2 solver only.
- B: A plus the volatile measured-evidence capsule.
- C: A plus one stable fixture-generated directional discovery demonstration.
- D: A plus both treatments.

All arms use the same checkpoint endpoint, provider pin, temperature 1.0, top-p 0.95, top-k 20, 16,384-token completion ceiling, tools, prediction gates, world-model-optional instructions, two-action host cap, and reasoning explicitly disabled at both endpoint switches. Arm order is A/B/C/D because the prior failed series already used that order; this run is a functional screen, not an unbiased effect-size estimate.

## Prediction

D should reduce serial inspection calls before the first action and should use the first measured transition to choose a second action. B or C alone may capture most of the benefit. A cheaper arm with equal useful interaction is preferred over D.

## Measures

- Whether each arm commits two valid environment actions and receives their real transitions.
- Model calls, prompt/completion/reasoning tokens, model wall time, tool time, and finish reasons.
- Presence and exact factor assignment of rendered evidence/demo blocks.
- Time and calls to first action, experiment records, evidence capsules, executable procedures, helper failures, and level progress.
- Whether the second decision responds to the first measured transition, assessed from retained calls and actions rather than keyword mention.

TTFT is recorded as unavailable for this non-streaming transport; it will not be inferred from total wall time.

## Cap

One run directory; no reuse or relaunch. At most 32 physical transmissions per arm, USD 6 cumulative reservation, 900 seconds per arm, two environment actions per arm, 16,384 completion tokens per request, and no automatic forwarding retry. The provider is pinned to CoreWeave FP8 with fallbacks disabled and required parameters enabled.

## Stop

Stop before the next arm on source drift, treatment mismatch, reasoning-setting drift, provider failure, arm timeout, unresolved admitted transport, worker/cleanup failure, or budget exhaustion. A normal two-action completion ends an arm.

## Invalid if

Source/card identity is missing, a request lacks its assigned factor, reasoning is not explicitly off, request/result provenance cannot be reconciled, environment actions are not attributable to retained model calls, or an admitted request lacks a terminal transport outcome.

## Result map

- One treatment repeats a useful two-transition loop with fewer calls/time and no new failure: advance it with A to the local Qwen/SGLang Kaggle notebook.
- Several treatments are practically tied: select the simpler one.
- Factors improve tool use but not evidence-responsive action: retain the traces and diagnose the newly exposed inference/planning failure; do not claim gameplay gain.
- No treatment improves or the run is technically invalid: retain A and diagnose the concrete failure; no relaunch under this card.

## Outputs

Raw requests/responses, lifecycle and worker IPC, per-arm results, candidate/card hashes, and a separate post-run assessment under `analysis/deployment-probes/20260908-discovery-broader-v2/`. Reasoning text is not required to be retained because reasoning is disabled. No Kaggle launch or submission is authorized by this card.
