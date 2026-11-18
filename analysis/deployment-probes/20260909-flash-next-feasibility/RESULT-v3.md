# Flash-Next feasibility — version 3 result

> **Current verdict: operationally invalid; PASS withdrawn.** The server log
> contains an OOM, which the frozen card invalidates without a recovery exception.
> The original PASS and resulting next-action claims below are superseded.
> The fixtures also fail to cover the reported live prompt-size tail.
> See the [v3 audit](../20260909-experiment-checks/reviewed-final/flash-v3.json) and
> [coverage check](../20260909-experiment-checks/reviewed-final/coverage.json).
> Claims below that prompt disambiguation was proven to cause the changed
> outcome are also superseded: these runs did not isolate that causal effect.
> Detailed corrections and preserved observations remain below.

## Original account (superseded where contradicted above)

`pranavagrawai/flash-next-feasibility-20260909` version 3,
**`KernelWorkerStatus.COMPLETE`** at 12:35:54Z after 546 s of a 2700 s cap.
`verdict.json` was written, no `failure.json` exists, and cleanup succeeded.
Evidence in `raw-v3/` (v2's evidence is preserved separately in `raw/`).

## Verdict: pass

The card's prediction — "server becomes ready within 1800 seconds and completes
all eight requests with usable output, including offered Actor tools" — is
**confirmed in full**. The Result map therefore reads **Pass → prepare separate
long-context and MTP-off configuration, then matched 27B replay and bounded
gameplay.**

The card is byte-identical to the one v1 and v2 ran under (`afd59f1e…`), as are
the Actor fixtures and both donor files. Only `probe.py` changed across the
three attempts.

## All eight requests

| Case | TTFT s | Total s | Prompt tok | Completion tok | Reasoning tok | Effective tok/s | Finish | Tool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text | 0.63 | 0.65 | 18 | 2 | 0 | 3.1 | stop | — |
| tool | 0.82 | 0.92 | 284 | 26 | 0 | 28.4 | tool_calls | `submit_number` |
| json | 0.49 | 0.55 | 33 | 11 | 0 | 20.0 | stop | — |
| actor-case-0 | 1.74 | 2.35 | 10,559 | 107 | 0 | 45.4 | tool_calls | `execute_python` |
| actor-case-1 | 0.87 | 1.28 | 11,228 | 72 | 0 | 56.3 | tool_calls | `execute_python` |
| actor-case-5 | 1.38 | 1.84 | 17,578 | 78 | 0 | 42.5 | tool_calls | `execute_python` |
| actor-case-12 | 1.60 | 3.55 | 21,130 | 282 | 0 | 79.4 | tool_calls | `execute_python` |
| actor-case-13 | 1.61 | 2.03 | 21,257 | 79 | 0 | 38.9 | tool_calls | `execute_python` |

**All five retained Actor cases returned a schema-valid, offered
`execute_python` tool call.** That is the card's actual question, and it is
answered affirmatively. `actor_returned_tool` is `True` on all five.

Every request returned zero reasoning tokens and empty reasoning text, so
`chat_template_kwargs: {enable_thinking: false}` suppresses the channel on this
stack across real Actor prompts, not just trivial ones.

The `json` case now returns `{"value": 4}` and passes the unchanged exact-output
gate. That gate was never the defect; the v2 prompt was.

## Infrastructure

| Fact | Evidence |
| --- | --- |
| GPU | `NVIDIA RTX PRO 6000 Blackwell Server Edition`, 97,887 MiB, 106 samples over 526 s |
| Peak GPU memory | 96,628 MiB of 97,887 — a 1.3% headroom margin |
| Model | `Qwen/Qwen3.8-Flash-Next-NVFP4`, `max_model_len` 32,768 |
| Runtime | `vllm-0.1.dev20073+g8e685d198-bf09729e` |
| Server ready | 410.06 s (v2: 450.07 s), against an 1800 s budget |
| Cleanup | `stopped: true`, `survivors: []`, `port_open: false`, identity matched |

## Native MTP, now on a real sample

`spec_decode` counters moved 0 → 222 drafts, 0 → 665 draft tokens, 0 → 441
accepted tokens, across 657 completion tokens. That is a **66.3% draft-token
acceptance rate** over a meaningful sample, unlike v2's 32/32 over 39 tokens.

Treat it as a measurement of this profile, not a speed claim: the eight requests
are heterogeneous, prefix caching is disabled, MTP depth is fixed at 3, and the
card explicitly forbids a comparative performance conclusion. Effective
throughput above is completion tokens over total latency, not decode rate.

## Real Actor prompt sizes

The retained Actor prompts measure 10,559 to 21,257 tokens. Against
`max_model_len` 32,768 that leaves 11.5k tokens of headroom at the largest,
before the 1024-token output cap. This is the first direct measurement of Actor
prompt cost in tokens rather than characters, and it sharpens the long-context
question the Result map sends us to next: the donor's pinned 32K is not
comfortable for the largest retained prompts once output and growth are
included.

**The 32,768 is not a setting this probe chose, and not a property of the
hardware.** It is `ANALYZER_CONTEXT = 32_768`, a hardcoded constant at
`serving_setup.py:129`, passed to `--max-model-len` at line 2322. The probe sets
nothing about context, and there is no `TAAF_*` environment override for it —
unlike the nine tuning values, which are all env-driven. So the Result map's
next step, "prepare separate long-context configuration", means **patching
Keith's donor constant**, not flipping an environment variable. That is a
materially larger change than the tuning profile has needed so far, and it
lands against a 1.3% GPU memory margin.

The ceiling is specific to this Flash-Next NVFP4 profile and does not transfer
to the 27B SGLang path, which serves `context_len=262144`.

## What this run still does not establish

Unchanged from the card: no 64K/100K support, no prefix-cache reuse, no MTP-off
comparison, no matched 27B comparison, no ARC solve quality, no gameplay, no
promotion. No tool was executed — `execute_python` calls were recorded and
discarded. Peak memory at 98.7% of the card leaves little room for a larger
context without retuning the 5 GiB KV reservation.

## The three fixes, judged after the fact

| Fix | Did it matter in this run? |
| --- | --- |
| `probe.py:84` `reasoning.strip()` | No — zero reasoning tokens throughout. Dormant, still correct. |
| `probe.py:98` prompt disambiguation | **Yes** — this is the abort that killed v2. |
| `probe.py:100` record instead of abort | No — all five Actor cases happened to return tools. |

Two of three were not needed by this run's outcome, and that is worth stating
plainly rather than claiming three saves. The line-100 fix in particular would
have made no difference here: every Actor case returned a tool. It remains the
right change — `tool_choice: "auto"` permits prose, so the gate contradicted its
own request and the next run may well hit it — but this run does not vindicate
it. It got lucky, and a fix that is right for a reason the evidence has not yet
exercised should be recorded as such.

## The whitespace-reasoning hazard has never actually been observed

Correcting the standing of the evidence for the `probe.py:84` fix, prompted by
the peer session that made the same repair.

The only demonstration that an empty `<think>\n\n</think>` produces a truthy
whitespace `reasoning_content` is `gate_test.py` — a stream **I constructed by
hand** to probe the assert. It is a synthetic construction, not an observation
of this server. No run has produced it.

Evidence pointing the other way is real and larger: 8 streaming responses here
and 106 non-streaming thinking-off calls in the peer's cancelled v1 arm all
report `reasoning_tokens` 0 with empty reasoning. So on this pinned stack the
hazard has not appeared on either surface.

That does not make the fix wrong. Streaming `reasoning_content` deltas and a
non-streaming `reasoning` field are different surfaces and may differ, and
`assert not reasoning.strip()` costs nothing. But it should be recorded as a
**defensive change against a hazard neither session has observed**, not as a
fix validated by evidence. Citing a stream one wrote oneself as if it were a
measurement is the error to avoid here.

Revised tally of the three fixes: one mattered (the `json` prompt), one is
dormant, and one guards a case that may not occur on this path at all.

## Memory accounting for a long-context configuration

Correcting my own framing above. "1.3% GPU margin, so there is little room to
grow context" conflated two different things, and the arithmetic separates them.

Measured on this run:

| Quantity | Value | Source |
| --- | --- | --- |
| GPU total | 97,887 MiB | `nvidia-smi` samples |
| Peak used | 96,628 MiB | 106 resource samples |
| Free at peak | 1,259 MiB (1.23 GiB) | difference |
| Model weights | 81.8 GiB | `Model loading took 81.8 GiB` |
| KV pool (pinned) | 5.00 GiB | `TAAF_VLLM_KV_CACHE_MEMORY_BYTES` 5368709120 |
| **KV capacity** | **105,202 tokens** | `GPU KV cache size: 105,202 tokens` |
| Worst-case concurrency at 32,768 | 3.21x | `Maximum concurrency for 32,768 tokens per request: 3.21x` |

**KV is consumed per actual token, not per `max-model-len`.** So raising
`ANALYZER_CONTEXT` is close to free in memory as long as prompts do not actually
grow: `max_num_batched_tokens` stays 8,192 and the cudagraph capture size stays
32, neither of which is derived from the context. What a raise actually costs is
*worst-case* concurrency — 105,202 / 65,536 = **1.61x at 64K**, down from 3.21x.

Preserving 3.21x at 64K is the thing that does not fit: it needs a 10 GiB KV
pool, +5 GiB, against 1.23 GiB free. That is the real blocker, and it is a
blocker on the *pool*, not on the context ceiling.

Today's actual concurrency is better than the worst case. At the largest
measured Actor prompt of 21,257 tokens the pool holds 105,202 / 21,257 ≈ **4.9
concurrent requests**, and a context raise would not change that number at all.

So the corrected position: a 64K ceiling is affordable on this GPU *provided
real prompts stay near today's 10.5-21.3k*, and unaffordable the moment they
approach the new ceiling. The long-context question is therefore not "does 64K
fit in memory" but "do our prompts actually need to grow, and if they do, how
much concurrency does the solver require".

### The donor constant is five coupled sites, not one

Verified against `raw-v3/donor-serving_setup.py`:

| Line | Use |
| --- | --- |
| 129 | `ANALYZER_CONTEXT = 32_768` |
| 140 | `NO_CHUNKED_MAX_NUM_BATCHED_TOKENS = ANALYZER_CONTEXT` |
| 360-362 | raises if `TAAF_VLLM_MAX_NUM_BATCHED_TOKENS` < `ANALYZER_CONTEXT` |
| 2322 | `--max-model-len` |
| 2935 | exported `LOCAL_ANALYZER_CONTEXT_WINDOW` |

Lines 140 and 360 sit on the **chunked-prefill-disabled** path only. This
profile runs `enable_chunked_prefill: true` with `max_num_batched_tokens` 8,192,
so both are inert here — but they mean a long-context variant that also disables
chunked prefill for diagnostics would raise its batch cap to the new context and
start rejecting batched-token overrides below it. Line 2935 also leaks the
constant into the harness environment, so a raise must stay consistent across
server and client or the two disagree about the window.

Found by the peer session cross-checking this write-up; the memory arithmetic is
mine.

## The margin was already exceeded once, silently

Found by the peer session reading this run's server log. Verified here:

```
12:34:17  reserved 5.0 GiB memory for KV Cache ... skipped memory profiling
12:34:22  [AutoTuner] Tuning trtllm::fused_moe::gemm1: 20/21
          CUDACachingAllocator.cpp:3933] memory allocation failed with OOM
          on device 0 while trying to allocate 511705088 bytes
          (free: 42336256, total: 101973950464)
12:35:14  server ready
12:35:14  first request
```

An allocation of 488 MiB failed with **40.4 MiB free**. The caching allocator
absorbed it, the autotuner skipped that tactic and continued, and the run went
on to COMPLETE. Nothing surfaced.

**This corrects the headroom number above.** "Free at peak 1,259 MiB" is derived
from 5-second `nvidia-smi` sampling and is a peak-of-samples, not a minimum. The
instantaneous trough was 40 MiB — roughly 30x smaller, and it is the trough a
configuration change has to survive, not the sampled figure. So the honest
statement is that this profile runs with **essentially no instantaneous margin
at startup** and about 1.2 GiB during steady-state serving.

The two are separable, which keeps the long-context conclusion intact but
narrows it. The trough occurs during flashinfer autotuning, whose scratch
buffers scale with hidden size and batch tokens, not with `max-model-len`.
Raising `ANALYZER_CONTEXT` alone therefore does not enlarge this allocation. But
anything that touches startup memory — a larger KV pool, a bigger
`max_num_batched_tokens`, a larger cudagraph capture — is operating against
40 MiB, not against 1.2 GiB, and would likely fail where this run survived.

### Did the skipped tactic affect the 66.3% MTP acceptance?

Almost certainly not, and the reasoning is worth recording because the peer
correctly declined to guess.

The skip cost the autotuner one candidate *kernel selection* for a MoE GEMM.
That is a throughput property: a slower kernel may have been chosen for that
shape. MTP acceptance is the fraction of draft tokens the target verifies, which
is determined by agreement between the draft and target distributions — a
property of weights and sampling, not of which GEMM tactic computes them.
Different tactics differ in floating-point accumulation order and could perturb
logits marginally, but that is a second-order effect with no mechanism to move
acceptance by a meaningful amount.

Two things bound the concern further. The log shows `Skipped 4 unsupported
tactic(s)` on *every* autotune step, so skipping is routine and the autotuner is
designed to tolerate it; the OOM is one more skip, anomalous in cause rather
than in kind. And this run has no control, so nothing could be attributed even
if the effect existed.

Where it does land is **throughput**: the effective 38.9-79.4 tok/s figures may
understate a fully-tuned configuration. The card already forbids a comparative
performance claim, and this is a second, independent reason that prohibition is
correct rather than merely cautious.

## CORRECTION: the verdict is not licensed by the card

An independent analysis (`INDEPENDENT-ANALYSIS-astra.md`) disputed the pass. Three
of its claims were checked against the artifacts and all three hold.

**1. Both v2 and v3 contain an OOM, and the card invalidates on OOM.**
`raw/vllm-openai-server.log:488` records the same failed allocation as v3
(511,705,088 bytes, 42,336,256 free) at 12:07:09. `RESULT.md`'s "none of the
Invalid-if conditions occurred ... no OOM" is **false**, and so is the equivalent
premise under which v3 was recorded as a pass.

The card's Invalid-if clause reads: "Wrong model/runtime/GPU identity, missing
request/response accounting, timeout, **OOM**, failed cleanup, or missing terminal
stream/usage." It does not distinguish a recovered allocator failure from a fatal
one. Reading in an exception for "recovered" is the exact mirror of the
over-implementation defects found earlier today: `probe.py:84` and `:100` added
stops the card never stated, and carving out an OOM the card *does* state is the
same error inverted. The standard has to apply in both directions.

**Corrected verdict: operationally invalid under this card.** Result map branch:
operational failure → retain 27B, preserve the failure, no retry under this card.
The eight responses, their latencies, the MTP counters and the identity evidence
remain valid observations and should be retained — but they do not license a
feasibility conclusion under the card as written. A successor card should decide
prospectively whether a recovered allocator failure invalidates.

**2. "About 1.2 GiB during steady-state serving" is unsupported.** 1,259 MiB is
the all-run sampled minimum, and it occurs during startup. Only three resource
samples fall after `ready_epoch`, and they show 5,187 / 3,027 / 3,027 MiB free.
No sample establishes an instantaneous serving minimum. The startup trough
(40.4 MiB) and the sampled peak are different instruments at different moments,
and neither is a steady-state figure.

**3. Setup was 514.1 s, not 410 s.** `ready_seconds` 410.056 is server readiness;
`setup_elapsed_seconds` is 514.107, including 103.938 s of runtime extraction.
v2's equivalent is 551.686 s against 450.067 s readiness. Both are inside the
1800 s budget, but "setup cost 450 s" understates it and is not a controlled
comparison between versions.

Also noted and correct: `raw/flash-next-feasibility-20260909.log` now holds v2's
execution trace, not v1's. Downloading v2 into the same directory overwrote the
v1 papermill log, so the missing-kernelspec failure is quoted in `STATUS.md` but
is no longer independently verifiable from retained artifacts. v3 was downloaded
to `raw-v3/` precisely to avoid repeating that, one version too late.

## The fixtures were not representative, and 32K is already breached

The long-context section above asks "do prompts need to grow". Measured against
the live workload, that question is already answered and the answer invalidates
the framing.

The contingent-controller run on the same day recorded 136 live Actor calls on
`ls20`. Their prompt sizes, verified from its `model_calls.jsonl`:

| | off arm | on arm |
| --- | ---: | ---: |
| Actor calls | 61 | 75 |
| Minimum | 12,664 | 13,138 |
| Median | 21,312 | 21,874 |
| p90 | 30,621 | 30,378 |
| **Maximum** | **38,644** | **38,565** |
| **Calls exceeding 32,768** | **5** | **3** |

**Eight of 136 live Actor calls (5.9%) exceed the donor's pinned 32,768 ceiling,
after only twenty actions.** Prompts grow with trajectory length, so a longer run
makes this worse, not better.

This probe's five retained Actor fixtures measured 10,559 to 21,257 tokens —
approximately the *median* of the live distribution and none of its tail. They
were drawn from `prefix-cache-20260907/corpus.json`, an older configuration, and
`build.py` selected them by index (0, 1, 5, 12, 13) rather than by size. So the
feasibility probe sampled the easy half of the workload and never presented the
server with a request that would have failed.

Two consequences:

1. **The card's fixture selection is a defect, not a limitation.** "Five unique
   retained Actor cases" reads as coverage; it was a convenience sample from a
   stale corpus with no size criterion. A serving-feasibility probe whose inputs
   cannot reach the configuration's limit cannot establish serving feasibility.
   Any successor must select fixtures against the live distribution — at minimum
   the observed maximum, not the median.
2. **The long-context work is not optional tuning.** The earlier section framed a
   64K ceiling as affordable "provided real prompts stay near today's
   10.5-21.3k". They do not. The ceiling is already breached by the real
   workload, which puts patching `ANALYZER_CONTEXT` ahead of the KV-pool
   arithmetic rather than beside it.

Found by the contingent-controller session from its own run, which sampled the
live distribution this probe was built to characterise and did not.
