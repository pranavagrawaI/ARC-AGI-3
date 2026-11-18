# Marlin reduction replay preparation

Owner: codex-marlin-replay-owner. Local preparation only. No GPU launch or upload authorized.

Question: does existing FP32 scratch reduction beat the architecture-selected atomic path on identical real S10 target expert inputs?

1. Verify the exact v3 wheel and source members; never patch the local reference checkout or frozen baseline.
2. Add optional reduction override and capture tag to quant info and fused caller. Bind them only to GPTQ MoE kernels belonging to a ModelRunner with is_draft_worker=False. Compressed-tensor drafter receives defaults. Keep FP32 reduction enabled.
3. Capture first/middle/last target MoE layers. Insert device-to-device snapshots into each graph at the fused-call boundary. Flush after actual target verification graph execute at full occupancy10, preserving actual graph key, row count, routes, weights and replay ordinal. Select ordinals8/16/32 after an explicit capture arm marker (after prefix warmup). No warmup tensors qualify as real captures. Only capture graphs are instrumented; timings use isolated clean fused graphs later.
4. Replay baseline/treatment complete fused MoE calls, including routing alignment, activation, both GEMMs and final sum. Also report eager diagnostic timing. Alternate AB/BA order, synchronize outside each CUDA event interval, and preserve all paired observations. Run a pack/unpack sentinel before relying on an independent dequantized FP32 MoE oracle. TF32 disabled for the oracle.
5. Qualify numerical error, unmodified routing/input, poison/replay scratch safety and memory before advancing. No serving comparison inside this run. Serving comparison requires a separate immutable card, runtime pair and authority.

Pre-mortem is in PRE-MORTEM.md. Numerical/performance/cap decisions are in EXPERIMENT-CARD.md. No statistical or performance observation exists yet.
