# Pruning transfer: failure paths checked before launch

The failure to prevent is spending the six-hour allowance on a silently misloaded model, or declaring a speed gain while discarding gameplay quality or the retained c14 settings.

- Wrong expert correspondence -> wrong weights paired with router rows -> quality loss falsely attributed to pruning. High impact, detectable before GPU. All 48 routers now match byte-for-byte between pinned Intel and RadixArk; runtime checks those hashes again. Residual: quantized values differ across checkpoints, so historical efficacy does not transfer automatically.
- Loader remaps weights but not GPTQ scales/zeros -> invalid model. High impact, cheap offline detection. Exercise all 221184 actual indexed target tensors; retain 193536, drop 27648, align 48 routers, and preserve tensor objects. Live shape receipts and short requests then exercise the real runtime.
- Target config contaminates MTP -> lower acceptance or startup failure. Separate target metadata view and original draft preparation; runtime draft banks must remain 512. Source mutations are bounded by exact wheel member hashes.
- Kaggle splits HF shards -> false package mismatch. Observed 38 versus 17 files. Verify tensor-name inventory independent of shard layout and record the actual index hash; do not use filename equality as weight identity.
- More streams exceed real state/KV capacity -> OOM, retractions, or silently capped concurrency. Detect with server-info internal state, startup graph capture, 14/18 request batches, then full-horizon KV/retraction logs. Small requests do not prove long-context capacity; that remains a gameplay risk.
- Stochastic early score or throughput alone triggers promotion -> retained baseline regresses. Freeze paired score-first rule and identical horizon before launch. Preserve c14 on ambiguity; no hidden-score inference from one pair.
- Shared quota/slot races or ambiguous API response -> duplicate expensive runs. Fresh live admission, immutable one-attempt markers, six-hour cap reservation before push; unknown acceptance stops the sequence. No retry.
- Collector failure loses the decision -> improvement forgotten again. Read-only resumable collection, remote version/source guards, raw hashes, deterministic RESULT/retention artifacts and shared-board update. Failed evidence and c14 fallback remain intact.

Accepted residuals: one paired gameplay sample per arm; physical GPUs are the same model but not the same device; changed routing can affect rare tasks not sampled here; source-level/local tests cannot certify CUDA execution before the probe.
