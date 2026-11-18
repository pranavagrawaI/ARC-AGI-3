# MTP sampling repair — ready for qualification, not deployment

Upload sampling_repair.ipynb with kernel-metadata.json sources, attached to
arc-prize-2026-arc-agi-3. RTX PRO6000, private, internet disabled,7800s session cap.
No push or relaunch was performed during preparation.

The pinned EAGLE stochastic verifier imports renormalization directly from
sgl_kernel, independently of --sampling-backend pytorch. Its wrapper prefers
FlashInfer. The retained logs first report SM12.x requiring CUDA>=12.9; the empty
FlashInfer target set then produces the misleading SM75 error. Offline operation
alone is not the cause. The inventory includes a newer compiler, but compiler and
runtime headers have different versions, so compiler selection is not this repair.

This candidate opts into EXISTING precompiled top-k/top-p kernels in the pinned
sglang-kernel0.4.6.post1. install_sampling_patch.py verifies the original wrapper
SHA256 before adding the opt-in SGLANG_SPEC_NATIVE_RENORM switch. The default route
is unchanged without that variable. No tree-verification, acceptance, model,
tokenizer, top-k, top-p, temperature or reasoning settings are changed.

Execution order:

1. Existing offline bootstrap and hardware check.
2. Actual top-k20/top-p0.95 kernel check against numerical references, at full
   vocabulary width, with FlashInfer JIT disabled. Failure stops before model load.
3. Same14-request baseline/MTP replay as the previous notebook. Each arm's excluded
   64-token warmup body and response are now retained. MTP warmup must show a
   positive spec_verify_ct before the corpus starts. Omitted natural sampling
   fields remain omitted and resolve through the verified generation configuration.

The replay HTTP watchdog detects engine death promptly. Cleanup errors are saved
separately instead of masking the original generation failure. This does not yet
change the separate integrated pilot's outer worker; do not rerun that pilot until
the repaired replay establishes stochastic MTP on target hardware.

Local validation used RTX5060Laptop/SM120 with Torch2.13.0+cu130,
FlashInfer0.6.17 and sglang-kernel0.4.6.post1. Native kernels matched references.
No27B model was loaded locally. toolchain.py is retained diagnostic code for the
rejected compiler-selection route and is NOT embedded or used by the notebook.

Build/check: python3 build.py; python3 build.py --check.
Read EXPERIMENT-CARD.md. Keep raw outputs, including failure cases. The card makes
this a bounded qualification run, not automatic promotion of MTP.
