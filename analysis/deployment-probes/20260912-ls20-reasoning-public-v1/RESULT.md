# LS20 reasoning-on public run v1 — result

Status: INVALID. Kaggle kernel version 1 ended `ERROR`; do not interpret this as a solver-quality result and do not promote reasoning-on.

Kernel: https://www.kaggle.com/code/pranavagrawai/arc3-ls20-reasoning-v1-20260912

## What happened

The requested NVIDIA RTX PRO 6000 Blackwell Server Edition was allocated. The offline wheelhouse and pinned model verified, SGLang became request-ready, runtime v11 reached the official launcher, and Qwen reasoning was observably active.

The solver completed six Actor requests, all ending in `execute_python`, without submitting a game action. The seventh request exceeded the 90-second HTTP timeout and failed at monotonic 785.012. This aborted the official launcher before it could produce a game result.

Teardown contained the process tree, but the lifecycle gate also failed because the timed-out model request was still decoding immediately after SIGTERM. The log classifier recorded one in-flight request line and three unclassified post-termination lines. This is secondary to the request timeout.

## Measurements

- Actor requests admitted/completed/failed: 7 / 6 / 1; lifecycle reconciles.
- Executed actions: 0.
- Levels completed: unavailable; no completed game record exists.
- Critic calls: 0.
- Reasoning: present on all 6 completed calls; 7,574 reasoning tokens out of 9,025 completion tokens (83.9%).
- Completed-request wall time: 232.597 seconds total; 38.766 seconds mean; 36.113 seconds median.
- Longest completed request: 87.797 seconds. The next request timed out at 90 seconds.
- Model evidence: complete `request_messages` and `tool_results` on all 6 completed calls.
- False host-event phrase candidates in completed Actor outputs: 0.
- Terminal-class predictions: 0, because no action was submitted.
- Run result elapsed: 639.018 seconds.

Historical LS20 reasoning-off reference (`20260912-objection-closure-public-v2`, runtime v10): 88 completed calls, 6 actions, no failed requests, mean request latency 6.866 seconds, and mean completion length 199.2 tokens. This reasoning-on run averaged 38.766 seconds and 1,504.2 completion tokens: 5.65× mean latency and 7.55× mean completion length. The comparison is descriptive and includes runtime v11's telemetry-only observability change, although that change was locally qualified as behavior-preserving.

The card predicted 2–4 actions and a clean stop at the 600-second admission boundary. Instead, the run made zero actions and failed on a per-request timeout. The experiment is invalid by its precommitted timeout and cleanup criteria. It still supplies operational evidence that unrestricted reasoning is incompatible with the current 90-second request contract: reasoning consumed most generated tokens and prevented even the first action.

Kaggle quota changed from 30.00 GPU hours remaining before launch to 28.90 afterward. Kaggle therefore accounted 1.10 hours, exceeding the card's 1-hour cap even though the captured run lasted about 11 minutes. No retry was launched.

## Evidence

- `PRELAUNCH.json`: frozen launch prerequisites and authorization.
- `AUDIT.json`: derived audit; `valid: false`, reasoning and lifecycle checks preserved.
- `raw/qualification/result.json`: immutable run result, SHA-256 `66f28a87e91521da5371a1044ab8fdee05eae0f46af76fc634e5dfdce41f5977`.
- `raw/qualification/solver-workspaces/ls20-9607627b-oq_f267r/ipc/model_calls.jsonl`: six completed calls, SHA-256 `d5942cd921c977ed52cbf52d353d713640a68f3b8a3fa32e9a9ea781968bcfec`.
- `raw/qualification/solver-workspaces/ls20-9607627b-oq_f267r/ipc/request_lifecycle.jsonl`: reconciled 7/6/1 lifecycle, SHA-256 `c94f15e9a40e922663ab9793f94caf70e07e9b9ca383a7c1415c7d8f4947e0ea`.
- `raw/qualification/teardown-log-classification.json`: post-SIGTERM classification.
- `live-log.txt`, `monitor.out`, `final-status.txt`, and `postrun-quota.json`: external status evidence.

## Decision

Keep reasoning off. Do not retry this kernel under the closed card. If reasoning is tested again, bound its token budget or move it behind an explicit, sparse escalation policy and validate that intervention locally before authorizing another public run.
