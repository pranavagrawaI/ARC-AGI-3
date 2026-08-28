# Unified one-action contact — 20260828T170100Z

Question: Can the current unified `trigger/control` Solver cross the complete live path from ARC observation through Qwen 3.8 27B judgment to one legal action, accepted successor, post-action observation, and clean finish?

Posture: bounded diagnostic contact; preserve failure; no adaptive retry in this run directory.

Contrast: none.

Prediction: the worker reports live/trigger/control/host, the provider returns usable native tool calls, the Solver commits exactly one host-legal action, the host records its successor, the worker acknowledges `action_result`, and the run finishes at `action_budget`.

Measure: `analysis/unified-live-v2/runs/contact-20260828T170100Z-ar25-s0-1a/summary.json`, host transcript, raw ARC recording, worker logs, `tool-trace.jsonl`, `llm-trace.jsonl`, decision checkpoints, Builder/Scribe traces, and workspace artifacts.

Cap: 1 ARC action; Actor maximum 25 tool rounds; Builder maximum 40 tool rounds if the level-start trigger fires; one post-action consolidation path; 4,096 requested output tokens per provider call; 900-second worker decision timeout; 1,800-second launcher wall time. Model `qwen/qwen3.8-27b` through OpenRouter. Public OpenRouter metadata retrieved 2026-08-28 reports $0.425/M prompt tokens, $2.55/M completion tokens, and a 1,000,000-token context window. There is no host-enforced dollar cutoff, so action, call, output-token, and wall-time ceilings bound exposure instead.

Stop: stop after one accepted action and successor, or at the first ARC, provider, protocol, sandbox, schema, trace, timeout, or cleanup failure. Do not reuse this run directory.

Invalid if: the source binding changes; metadata is not live/trigger/control/host; the route is scripted or not Qwen 3.8 27B through the configured OpenRouter gateway; more than one ARC action occurs; an accepted action lacks a successor/acknowledgement; exact LLM/tool evidence is absent; or execution exceeds a stated cap.

Result map: complete one-action path → retain integration validity for this source, route, game, seed, and single action only; provider failure before decision → preserve and repair route/configuration without an ARC retry until an offline probe separates the cause; Solver/protocol/sandbox/schema/trace failure → repair offline before a new generation; legal action without complete successor evidence → treat the run as invalid and repair host evidence closure.

Outputs: `analysis/unified-live-v2/runs/contact-20260828T170100Z-ar25-s0-1a/`.

## Source and route binding

- Preflight: `analysis/unified-live-v2/runs/preflight-20260828T165910Z-ar25-s0-0a/` — passed with zero actions and zero provider calls.
- Worktree: `/home/hipra/projects/arc-agi-3-tycho-nooa-unified`
- Branch: `refactor/tycho-nooa-unified`
- HEAD: `fa190780ac20dd50b3b45a9ec7b63abbfda8bd79`
- Tracked patch SHA-256: `e2af7ff9083e8fbf9493234bcefecde86cf963aca861a16caaae56cec1bb68ae`
- Module disposition SHA-256: `37e5a9f24b8624f812ece3d7c51ea18783d2aa1d857d34098ca069f1d3be7672`
- Environment file: repository `.env`, mode `0600`; credential values omitted.
- Gateway: `https://openrouter.ai/api/v1`
- Model: `qwen/qwen3.8-27b` (`qwen/qwen3.8-27b-20260814` in the retrieved public metadata).
