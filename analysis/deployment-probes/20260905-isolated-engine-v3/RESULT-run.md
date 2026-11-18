# Isolated engine v3 — shallow MTP is a large real decode win; tuned Triton is not

`pranavagrawai/qwen38-isolated-engine-20260905-v3`, terminal **COMPLETE**. Kernel wall ~660 s
against the 3600 s cap. Genuine `NVIDIA RTX PRO 6000 Blackwell Server Edition`, capability
`[12, 0]`. All three arms reached `status: measured`; `VERDICT.json` is
`MEASUREMENTS_RETAINED_REVIEW_REQUIRED`.

## Decode throughput

Steady decode measured server-side, first output batch to finished. Sampling temperature 0,
top_p 1, top_k -1, identical 81-token frozen input; the four request bodies hash identically
across all three arms.

| arm | fixed-1 tok/s | fixed-2 tok/s | vs baseline | natural e2e | natural tok |
| --- | --- | --- | --- | --- | --- |
| baseline (auto backend) | 45.134 | 45.133 | — | 19.476 s | 879 |
| tuned-triton (5x PR28552) | 46.725 | 46.725 | **+3.5 %** | 19.337 s | 903 |
| shallow-mtp (EAGLE) | 71.574 | 71.568 | **+58.6 %** | 11.859 s | 861 |

The two fixed repeats agree to within 0.01 % inside every arm, so the gap is not run-to-run noise.

**shallow-mtp clears the card's >10 % bar by a wide margin**, with no natural-latency regression:
natural per-token cost falls from 22.2 ms to 13.8 ms. `speculative_num_steps 1`,
`speculative_eagle_topk 1`, `speculative_num_draft_tokens 2` as specified, and MTP genuinely
executed — `spec_accept_length` 1.950 on both fixed requests (525 verifies each) and 1.970 on
natural. That is ~97.5 % acceptance of the single drafted token against a 2.0 ceiling; the 1.59x
realized speedup against a 1.95x theoretical one is the draft forward's overhead.

**tuned-triton is rejected for this prompt.** +3.5 % is under the bar. The tuning did take effect —
`selected-config-log.json` shows all five shape-gated W8A8 block-FP8 configs loading at capture
(N=16384, 5120/6144, 34816, 5120/17408, 14336), and the baseline arm logs none. As the card
predicted, this is not a clean Triton A/B: baseline auto-selects its own backend, so +3.5 % is the
tuned-explicit-Triton-vs-auto delta, not tuned-vs-untuned Triton.

## The one thing needing review: outputs diverge across arms

Greedy sampling did not produce identical token IDs.

| comparison | warmup | fixed-1/2 | natural |
| --- | --- | --- | --- |
| baseline vs tuned-triton | diverges at 48 | diverges at 48 | diverges at 48 |
| baseline vs shallow-mtp | **identical (64/64)** | diverges at 288 | diverges at 288 |

The card forbids silently accepting this, so: the divergence is benign. All three arms solve the
prompt and reach the same correct result, `\boxed{232792560}` = lcm(1..20), by the same prime-power
route. The reasoning prose differs only in wording after the divergence point. This is the expected
greedy tie-break sensitivity to FP8 GEMM numerics — a different kernel gives a different
last-bit logit and eventually flips one argmax. MTP staying bit-identical for 64 tokens and
diverging only at 288 is consistent with the same cause via the batched verify forward, not with a
speculation-correctness bug. No follow-up correctness check is owed on this prompt; a matched
workload comparison still is.

## What this does not establish

- **One synthetic reasoning prompt, single stream.** Not the frozen Actor corpus, no gameplay, no
  tool-call or structured-output correctness. Acceptance length on a math-reasoning prompt need not
  hold on ARC-AGI-3 Actor traffic, which is where it would have to pay off.
- **Startup times are not a comparison.** 203.5 s / 124.8 s / 100.7 s follows run order, i.e. page
  cache warming, not a candidate property.
- **No deployment promotion.** Per the card this is diagnostic only.

## Integrity

Model manifest `matched` across 81 files before any generation. All three arms shut down clean:
`containment_passed true`, `root_returncode 0`, `term_only true`, `forbidden_count 0`, every
teardown line classified as an allowed normal event. Zero games, actions, submissions or external
model calls. Under one GPU-hour on a single private allocation, no retries.

## Result-map disposition

- **shallow-mtp** → select for a later matched-workload and correctness comparison against the
  Actor corpus. Not a deployment decision.
- **tuned-triton** → retain baseline; reject this candidate for this prompt.
