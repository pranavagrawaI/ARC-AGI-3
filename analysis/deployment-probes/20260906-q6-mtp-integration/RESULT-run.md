# q6-mtp-integration-20260906 — same EAGLE/FlashInfer crash; zero attributable actions

`pranavagrawai/q6-mtp-integration-20260906`, pushed 12:36:40Z, queued ~21 min, RUNNING 12:58:02Z,
terminal **ERROR** 13:19:05Z. Probe `elapsed_s` 1194.3, inside the 4800 s session cap.

**The card's success criteria are not met.** Required: model requests reconcile, at least one valid
committed environment action, no integrity/lifecycle failure, and actual MTP work established.
Actions committed: **0**. Solver model calls recorded: **0**. No MTP work was ever performed.

## Identical root cause to the replay, reached independently

The server came up fully — model API ready at 150.0 s, `Uvicorn running` 13:01:29, `The server is
fired up and ready to roll!` 13:01:34 — and the runtime-owned warmup **succeeded**
(`chat_completion_accesses: 1`, `successful_chat_completion_accesses: 1`, 4.1 s).

Nine seconds later, on the first real solver request, the scheduler died:

```
sgl_kernel/sampling.py:59   top_k_renorm_probs(probs, top_k)
flashinfer/jit/core.py:109  raise RuntimeError("FlashInfer requires GPUs with sm75 or higher")
[13:01:43] SIGQUIT received. It usually means one child failed.
```

Same misleading message, same real cause as `actor-mtp-20260906`: `top_k_renorm_prob` is not
precompiled in this wheel, FlashInfer attempts a JIT build, and `check_cuda_arch()` fails offline on
this sm120 device. Two independent runs, two different harnesses, one fault.

**The warmup surviving is the informative detail.** It passed; the first solver request killed the
scheduler nine seconds later. The likely explanation is that the warmup does not use the model's
`top_k: 20` default while real solver traffic does — but this run did not retain the warmup body, so
that is inference, not established fact.

## Everything downstream is a consequence, not a separate fault

With the scheduler dead, the worker had nothing to answer with. The launcher spent its 900 s
pending-worker window and then failed:

```
TimeoutError: worker timed out waiting for decision
failure_classification: pre_request_failure:official_launcher:RuntimeError
request_path_status: invalid
```

The worker itself died trying to report the error upward — `BrokenPipeError` in
`nooa_worker.py:60`, writing to a stdout whose reader was already gone. That is teardown noise, not
an additional defect. Most of the 1194 s elapsed was the worker window expiring against an engine
that had already crashed.

`backend-facts.json` records `cuda_kernel_exception_observed: false` — correct: this was a Python
JIT-build refusal, never a CUDA kernel fault.

## What this run does and does not establish

- **It does not test the frozen prediction-feedback solver.** Zero solver model calls means the
  candidate never ran. Nothing here counts for or against it, and the invented-tool question is
  untouched.
- **It does not measure MTP.** No speculative work occurred, so there is no speed or accept-length
  evidence of any kind.
- **It does establish that the engine delta is unusable as configured.** The pilot's only real
  finding is that shallow MTP plus the official launcher's real sampling cannot start work in this
  offline environment — the same conclusion the replay reached, now confirmed against the actual
  solver rather than a replay corpus.

Per the card's Stop/fail list an engine crash is a stop condition, correctly honoured. No automatic
relaunch was performed.

## Consequence for the isolated engine result

The 58.6% MTP win in `20260905-isolated-engine-v3` was measured with `temperature 0, top_k -1`.
Both of today's runs show that as soon as the real workload's sampling (`temperature 1.0,
top_k 20, top_p 0.95`) is used, the EAGLE verify path requires a kernel this environment cannot
build. That gain is not currently reachable by either the replay corpus or the live solver, and
the engine-v3 disposition — "select for a later matched-workload comparison" — should be read
against this constraint before any further MTP work is scheduled.
