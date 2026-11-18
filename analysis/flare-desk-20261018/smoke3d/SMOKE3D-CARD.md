# FLARE stage-3d smoke: card (frozen before launch, 2026-10-20)

## What 3c showed (flare-smoke-3c-20261020, notebook ~11:55–12:35Z)
Flare served. Health came 140 s after start, including `--verify-weights` (119.8 s): weights 72.4 GiB,
2,481 kernels, "Prepare execution" OK. The KV cache holds 1,282,048 tokens (18.8 GiB), the host prefix
cache is 64 GiB, and the limit is 262,144 tokens per request. The only attribute the driver rejected
was id 11, CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION, with value 0,0,0 ("no preference"), so
skipping it is a no-op. But every request got 400: "image count is empty or exceeds the configured
limit". /v1/models reports `max_images_per_prompt: 4`; the logged dfranzen requests carry a median of
24 images (max 41). The smoke's 400 fallback then stripped top_k and chat_template_kwargs, which were
not the cause.

## Change (one): the replay workload strips history images
dfranzen's harness already has `ARC3_HISTORY_IMAGE_KEEP=N` (images on the N most recent history
messages, older ones replaced by "[grid image omitted from history]"). With N=0, only the current turn
keeps its images, and a turn carries at most 3. Simulated over all 2,352 logged requests: max 3, none
over 4 (N=1 would put 0.8% over 4). The smoke applies exactly that transform to every replayed and
functional request, and it asserts none exceeds 4. Everything else is 3c. Any Flare arm must run with
ARC3_HISTORY_IMAGE_KEEP=0.

## Bias, and the decision rule adjusted for it (fixed now, before the result)
Stripping cuts ~24 images × ~400 tokens off a 72K-token mean prompt (~13%). The SGLang M3A reference
(U_overlap 477.6) was measured with full images. So r leans toward Flare, most likely by under 10% of
decode throughput.
- INVALID: Flare never healthy, or any route-B validity check fails (SMOKE3B-CARD.md).
- KILL: fewer than 3/6 functional rows have a `python` tool call with nonempty `code`, replay errors
  exceed 2%, or r < 1.10.
- SAME-WORKLOAD REFERENCE: 1.10 ≤ r < 1.45. Before any decision, SGLang must be measured on this same
  stripped workload.
- GO: r ≥ 1.45 (the old 1.30 bar plus a 10% bias margin). Next is the 60-min screen. Its arms must
  separate the runtime from the image window: F1x+KEEP0 on SGLang vs F1x+KEEP0 on Flare, with KEEP0's
  own effect against F1x measured separately.
- Reported, not decided: the S=14/16 results, cached_fraction_block, MTP acceptance if exposed, load
  time, the library stats.
