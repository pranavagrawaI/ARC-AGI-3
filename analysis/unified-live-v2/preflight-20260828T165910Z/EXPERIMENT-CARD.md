# Unified live preflight — 20260828T165910Z

Question: Can the current unified Solver cross ARC authorization, environment reset, host-to-worker startup, live-client construction, and clean finish without making a provider call or ARC action?

Posture: cheapest discriminating contact; preserve failure; no adaptive retry in this run directory.

Contrast: none.

Prediction: `ar25` seed 0 starts; the worker reports `mode=trigger`, `treatment=control`, and `sandbox_runtime=host`; the host stops at `action_budget` with zero actions; no provider call is made.

Measure: `analysis/unified-live-v2/runs/preflight-20260828T165910Z-ar25-s0-0a/summary.json`, `transcript.jsonl`, worker stdout/stderr, workspace metadata, and absence of LLM/tool traces.

Cap: 0 ARC actions; 0 provider calls; 900-second worker timeout; 300-second launcher wall time; no model spend; one ARC environment creation/reset contact.

Stop: stop on clean `action_budget` finish or the first ARC, protocol, worker, configuration, or cleanup failure. Do not reuse this run directory.

Invalid if: the source binding changes; metadata is not trigger/control/host; any decision/provider call or ARC action occurs; the run uses a scripted client; or required host evidence is absent.

Result map: clean zero-action finish → authorize preparation of a separately bound one-action card; ARC 401 → replace the ARC credential, preserve this failure, and use a new run generation; worker/configuration failure → repair offline before another ARC contact; ambiguous or missing evidence → preserve and stop.

Outputs: `analysis/unified-live-v2/runs/preflight-20260828T165910Z-ar25-s0-0a/`.

## Source binding

- Worktree: `/home/hipra/projects/arc-agi-3-tycho-nooa-unified`
- Branch: `refactor/tycho-nooa-unified`
- HEAD: `fa190780ac20dd50b3b45a9ec7b63abbfda8bd79`
- Tracked patch SHA-256: `e2af7ff9083e8fbf9493234bcefecde86cf963aca861a16caaae56cec1bb68ae`
- Module disposition SHA-256: `37e5a9f24b8624f812ece3d7c51ea18783d2aa1d857d34098ca069f1d3be7672`
- Environment file: repository `.env`, mode `0600`; credential values omitted.
- Model route configured but not contacted: `qwen/qwen3.8-27b` through OpenRouter.
