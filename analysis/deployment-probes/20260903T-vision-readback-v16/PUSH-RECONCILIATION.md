# V16 push reconciliation (immediate post-push)

## Push
- Kernel version **1** of `pranavagrawai/prototype-tycho-nooa-vision-readback`, pushed
  2026-09-03 (`kaggle kernels push`). New kernel; no prior version to displace.
- Local candidate: `deployment/kaggle/s0-vision-readback-v16-candidate`.
- Pre-push local hashes:
  - `s0_vision_readback.ipynb` = `18aa8d5e2de6b973f16f1abb212164bfa5e9a85705a8944273ae69b2c44da12e`
  - `vision_readback_probe.py` = `a52cdbbde7b2870205afdcf6ba2a72ab0f27899335b4877cd76ad2cdb4e5afca`
  - `exact_request_outer_runtime.py` = `be463cbc60b4569c5218ee4489150bf58f98442eff8928d8b678d20a63f805cd`
  - `frames.json` = `fd4031afec63975cd5bc98e8cc32445f54b510ac927aa486708447ca02b2d21d`

## Immediate remote readback (kernel version 1)
- Pulled remote source + metadata immediately after push, before interpreting anything.
- **All 4 notebook cell sources byte-identical to the frozen candidate.**
  - concat-source SHA-256 (local == remote) = `fe7a835662e9a00e2a56876a4a8ce5566352ef0c3d3d76734886ada5a34ee95c`
  - per cell: 1,296 / 461,953 / 7,869 / 1,655 chars, all identical
- Metadata: `enable_gpu=true`; `machine_shape=NvidiaRtxPro6000`; `enable_internet=false`;
  `competition_sources=[]`; `dataset_sources=[kaggle-t4-stack-2026-08-28,
  qwen3-8-27b-fp8-hf-snapshot, sglang-r8l3b-orderly-shutdown-runtime]`.
- No competition source is attached, so this kernel cannot produce a submission even by accident.
- Status after push: **RUNNING**.

## What is on the wire
- Cell 1 embeds the patched outer runtime (base64), the vision-readback probe, and the three
  pinned `ls20` frames.
- Cell 2 is the unmodified R8-L3B SGLang runtime install; the Solver wheelhouse is absent.
- Cell 3 refuses `KAGGLE_IS_COMPETITION_RERUN`, re-checks the frame asset's schema and count,
  then runs the outer runtime with `--r8-l3b --vision-readback-probe --vision-frames
  --vision-probe-timeout-s 5400`. It writes no `submission.parquet`.

## Status
Awaiting terminal state. On terminal: download output and source, confirm the measured
visual-token cost per render scale, the per-condition readback scores against the pinned frames,
the probe/server request-count reconciliation, and the R8-L3B lifecycle record, then write
TERMINAL-RECONCILIATION.
