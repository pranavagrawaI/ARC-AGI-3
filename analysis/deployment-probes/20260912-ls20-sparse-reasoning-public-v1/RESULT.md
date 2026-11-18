# LS20 sparse-reasoning public run v1 — result

Status: VALID. Kaggle kernel version 1 reached `KernelWorkerStatus.COMPLETE`; the runtime result and every final verification check passed. The run completed zero levels, so it is not evidence of solving capability or grounds for promotion.

Kernel: https://www.kaggle.com/code/pranavagrawai/arc3-ls20-sparse-reasoning-v1-20260912

## Reasoning result

Routine reasoning was disabled in both the launch configuration and every observed Actor and critic response. All 55 routine calls recorded zero reasoning tokens and no reasoning content.

The existing selective mechanism was retained: the Actor could call `deep_investigate(...)` at most once, with a 2,048-token cap, to issue a separate reasoning-enabled request. The Actor never selected it. No `deep_investigations.jsonl` or `deep_outcomes.jsonl` record was produced, and all 63 observed tool calls were `execute_python`.

This distinguishes the mechanism from the invalid prior run. The prior package enabled reasoning globally on ordinary Actor calls; this package kept ordinary calls off and left reasoning only behind the explicit sparse tool. In this live session, the resulting behavior was entirely reasoning-free because that tool was not invoked.

## Measurements

- Official result: `passed`; external kernel status: `COMPLETE`.
- Stop: clean `model_admission_budget` boundary after 602.56 seconds.
- Levels completed: 0.
- Requests admitted/completed/failed: 55 / 55 / 0; lifecycle reconciles.
- Routine requests: 49 Actor and 6 critic.
- Reasoning: 0 tokens on all 55 calls.
- Actor latency: 593.25 seconds total; 12.11 seconds mean; 10.79 seconds median.
- Actor output: 17,288 completion tokens; 352.82 mean.
- Policy batches: 6.
- Actions submitted: 10; executed by the launcher: 8; skipped by the host after a prediction mismatch: 2.
- Sparse reasoning calls: 0.
- Tool calls: 63, all `execute_python`.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition.
- Kaggle accounting: 1.10 GPU-hours before, 1.37 after; delta 0.27 hours, within the 1.5-hour cap.
- Shutdown containment and teardown classification passed; zero forbidden teardown entries.
- No retained `.env` file and no retry.

Against the historical reasoning-off runtime-v10 LS20 session, this single run executed 8 actions in 49 Actor calls versus 6 actions in 80 Actor calls: 6.13 versus 13.33 Actor calls per executed action. That is a descriptive stochastic comparison, not a causal improvement claim. Both sessions completed zero levels.

## Interpretation

The corrected configuration restored operational throughput and avoided the global-reasoning timeout failure. It does not live-validate the sparse reasoning path because the Actor did not invoke it. Local qualification proved that `deep_investigate` overrides the routine off configuration with a bounded reasoning-on payload; live usefulness remains untested.

The binding gameplay problem remains the Actor's inspection loop rather than reasoning: 63 Python-tool calls and 49 Actor requests produced 8 executed actions and no level completion.

## Evidence

- `AUDIT.json` / `audit.py`: immutable offline audit; all 14 validity checks passed.
- `VERIFIED.json` / `verify.py`: 10 final checks passed, including external COMPLETE status, zero routine reasoning, quota cap, source hashes, no retry, and no retained `.env`.
- `raw/qualification/result.json`: SHA-256 `9a04898db5f64f57da5bb97e4412466c11defea0bda0f5a94f26f3f20ebd505e`.
- `raw/qualification/solver-workspaces/ls20-9607627b-jz1v9k2m/ipc/model_calls.jsonl`: SHA-256 `b8fa8a726a81a4bdc81e39c6ce3e2e67231cf23c3c0f0b039add50b03c79a874`.
- `raw/qualification/solver-workspaces/ls20-9607627b-jz1v9k2m/ipc/request_lifecycle.jsonl`: SHA-256 `5a0fb379247af752d78e421d7eb305f8b69b2c90d0baf6dc05cfa1be007cea4d`.
- `PRELAUNCH.json`, `push.log`, `final-status.txt`, and `postrun-quota.json`: launch authorization and external-state evidence.

## Decision

Keep routine reasoning off. Preserve the sparse activation as an available mechanism, but do not claim live efficacy until a session actually invokes it. No additional kernel is authorized by this card.
