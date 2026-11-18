# actor-mtp-20260906 — baseline arm complete; MTP arm cannot run under the Actor's sampling

`pranavagrawai/actor-mtp-20260906`, pushed 12:38:26Z, queued ~27 min, RUNNING 13:05:48Z, terminal
**ERROR** 13:14:51Z. Genuine RTX PRO 6000. Stopped far inside the 7200 s cap.

**The baseline arm completed all 14 replays. The MTP arm died on its first generation.** Per the
card this is a stop condition correctly honoured, and it yields no selection: there is no paired
MTP measurement to compare.

## Root cause: EAGLE verify needs a FlashInfer kernel that cannot build offline

The MTP server started cleanly — `Uvicorn running` 13:14:16, `The server is fired up and ready to
roll!` 13:14:18 — and the scheduler raised one second later, at 13:14:19, on the first request:

```
eagle_worker_v2.py  verify -> run_eagle_verify -> eagle_sample
eagle_utils.py:791  target_probs = top_k_renorm_prob(...)
flashinfer/jit/core.py:109  raise RuntimeError("FlashInfer requires GPUs with sm75 or higher")
```

The message is misleading. This GPU is sm120, comfortably above sm75. The real failure is that
`top_k_renorm_prob` is not precompiled in this wheel, so FlashInfer tries to **JIT-build** the
sampling module, and `check_cuda_arch()` fails in the offline Kaggle environment. The server then
took `SIGQUIT` and became a zombie, which is why the runner reported the downstream
`LifecycleObservationError: target root lacks the exact R8_L3B_OWNER marker` — the observer read an
empty `/proc/1539/cmdline` from a `state: "Z"` process. Containment sent SIGKILL to the pgid and
recorded `observer_contract_invalid`. That lifecycle error is a symptom, not the cause.

## Why this did not appear in the isolated engine run

The engine v3 comparison, where MTP won by 58.6%, ran **greedy**:

| | engine v3 (MTP won) | this replay (MTP died) |
| --- | --- | --- |
| temperature | 0 | 1.0 |
| top_p | 1 | 0.95 |
| top_k | **-1** | **20** |
| do_sample | — | true |

Greedy decoding never calls top-k renormalization, so EAGLE verify never touched the missing kernel.
This replay preserves the Actor's real sampling via `--sampling-defaults model`, and the mounted
`generation_config.json` — verified equal to the capture — carries `top_k: 20`.

**So the engine v3 MTP result does not transfer to the Actor's actual sampling configuration in this
offline environment.** That is the substantive finding of this run, and it is exactly what the card
built the replay to expose: the synthetic-prompt speedup was measured under conditions the real
workload does not use.

## Baseline arm: 14/14 replayed, provenance intact

All 14 cases came from the sealed capture (`20260906-actor-capture`, attempts 0–13), every response
`finish_reason: tool_calls`, every `reasoning_tokens: 0`, no `limited` truncations. Total natural
wall 104.1 s. Native diagnostics on the longest prompt: 38.29 and 38.73 tok/s over 1024 output
tokens. Baseline teardown clean — `containment_passed true`, `root_returncode 0`, `term_only true`.

These are single-arm numbers. They are **not** a speed result: the card requires paired arms, and
this decode rate is not comparable to engine v3's 45.1 tok/s, which used an 81-token prompt and
different sampling.

## The invented-tool behavior is stochastic — first direct evidence

Replaying identical bytes did not reproduce the capture's failures in the same places:

| case | schema errors in capture | schema errors on replay |
| --- | --- | --- |
| 0004 | 1 | **0** |
| 0006 | 0 | **1** |
| 0008 | 0 | **1** |
| 0012 | 1 | **0** |
| 0013 | 1 | **0** |

Three invalid in the capture, two on replay, at **disjoint positions**. Identical inputs, different
failures — which is what `temperature 1.0 / top_k 20 / top_p 0.95` predicts, and it rules out a
deterministic prompt-shape trigger.

This does not measure reliability. It is one sample per request in one pass, n=14, and the card
warns that precision is limited. What it establishes is the *character* of the fault: sampling
noise, not a fixed defect at particular history lengths. A rate estimate needs repeated sampling per
request, which this design deliberately does not do.

## Disposition

- **No MTP selection.** No paired data. Baseline retained by default, as the card requires for
  mixed or absent results.
- **The blocker is environmental, not a solver or package fault.** MTP under non-greedy sampling
  needs either a FlashInfer build with precompiled top-k sampling kernels, or a serving path that
  avoids `top_k_renorm_prob`. Neither is a change this card authorizes.
- **The 14 baseline pairs are retained** and are legitimate reference outputs for the same corpus.

No automatic relaunch was performed.
