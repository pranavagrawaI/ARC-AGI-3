# mtp-sampling-repair-20260906 — the repair works; MTP is 30% slower than baseline

`pranavagrawai/mtp-sampling-repair-20260906`, pushed 14:03:26Z, no queue, terminal **COMPLETE**
14:18:41Z. ~15 min against the 7800 s cap. `VERDICT.json`: `MEASURED_REVIEW_REQUIRED`, 28 requests.

**Both gates passed. The measurement then went against MTP.** Per the card's result map this is
"runtime/quality failure -> retain baseline": the >10% paired-latency improvement required to select
MTP for integrated qualification did not occur, and the sign is reversed.

## Gate 1: native kernels correct on target hardware

```
device: NVIDIA RTX PRO 6000 Blackwell Server Edition   capability [12, 0]
renormalization_backend: sgl_kernel precompiled native   flashinfer_jit_disabled: true
top_k_max_absolute_error: 7.45e-09      top_p_max_absolute_error: 1.49e-08
```

`patched_sha256 4dae159c…` is byte-identical to the patch I reproduced locally on an SM120 RTX 5060
before launch, where errors were 1.12e-08 / 1.49e-08. The diagnosis was right: the precompiled
kernels existed all along and only the wrapper's FlashInfer preference was blocking them.

## Gate 2: stochastic MTP genuinely executed

No crash. `temperature 1.0, top_k 20, top_p 0.95` unchanged, and the EAGLE verify path ran:

```
spec_verify_ct 18   spec_accept_rate 0.944   spec_num_correct_drafts 17/18
```

On the matched fixed controls, `spec_verify_ct 536`, `spec_accept_length 1.910`. **The crash that
killed both of today's earlier runs is fixed.** That is a real result and it is what this run was
built to establish.

## The measurement: MTP is slower

Matched 1024-token native diagnostics, the only clean decode comparison:

| arm | fixed-1 | fixed-2 | vs baseline |
| --- | --- | --- | --- |
| baseline | 38.288 tok/s | 38.287 tok/s | — |
| shallow-mtp | 26.921 tok/s | 26.910 tok/s | **−29.7 %** |

The two repeats are close within each arm; this does not exclude fixed run-order or other
systematic effects. Across the 14 natural cases, baseline generated 3573 tokens in 121.0 s
(29.53 tok/s), versus MTP 3174 tokens in 150.6 s (21.07 tok/s), **−28.6 %**. Those totals mix
different output lengths and are not an independent equal-work decode comparison.

Correction after inspecting the retained `fixed-*-request.json` bodies: the fixed controls use
temperature 0, top-k -1 and top-p 1. Their slowdown cannot be attributed to stochastic top-k/top-p
renormalization. Accepted length was 1.91, but the cost of drafting, verification and state handling
has not been isolated. The cause of the regression remains unresolved.

**Per-case natural latency measures request experience, not equal-work execution speed.** Under stochastic
sampling the arms produced very different output lengths for the same input — case 0005 gave 159
baseline tokens against 656 MTP tokens — so per-case wall-clock deltas (median −13 %, mean −65.6 %)
measure how much text each arm happened to generate, not speed. The fixed controls exist precisely
because of this.

## Why engine v3 disagreed

| | engine v3 | this run |
| --- | --- | --- |
| fixed-control sampling | greedy (`temp 0, top_k -1`) | greedy (`temp 0, top_k -1`) |
| fixed-control input | short synthetic math prompt | long rendered Actor prompt |
| MTP result | **+58.6 %** | **−29.7 %** |
| baseline decode | 45.13 tok/s | 38.29 tok/s |

Both fixed comparisons were greedy. Context, request content, runtime configuration and run order
were not isolated across these runs. The synthetic win did not transfer; neither sampling nor
context has been established as the cause. Rates are derived from the server tokenizer's first
output to finished interval, not total HTTP elapsed time or pure GPU-kernel execution time.

## Output quality: clean, and stochastic again

All 28 responses: `finish_reason: tool_calls`, `limited: false`, `reasoning_tokens: 0`, prompt token
counts identical across arms, all 14 provenance records from the sealed capture.

**Zero tool-schema errors in either arm** — against three in the original capture and two in the
2026-09-06 baseline replay, at different positions each time. Three passes over identical bytes have
now produced 3, 2, and 0 invalid outputs. That further supports sampling noise rather than a fixed
prompt-shape trigger, and it still does not measure the rate.

Both arms shut down clean: `status passed`, `containment_passed true`, `root_returncode 0`,
`term_only true`.

## Disposition

- **Retain baseline.** MTP is not selected for integrated qualification; the card's threshold was a
  >10 % improvement and the observed effect is a ~30 % regression.
- **The repair itself is qualified and worth keeping.** It is the only reason stochastic EAGLE runs
  at all offline, and any future speculative work on this hardware needs it.
- **Do not rerun the integrated pilot for MTP.** Its premise — that MTP is faster — is now measured
  false for the tested long-prompt fixed control; natural stochastic traffic also showed no established gain.
- Fixed arm order, one sample per case, one sequence, and no run-order control limit precision.
  Nothing here certifies hybrid rejected-state correctness or gameplay.
