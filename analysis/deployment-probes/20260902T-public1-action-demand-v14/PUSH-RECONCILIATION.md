# V14 push reconciliation (immediate post-push)

## Push
- Kernel version **12** pushed 2026-09-03 (`kaggle kernels push`).
- Notebook: `pranavagrawai/prototype-tycho-nooa-public-3-full-timing` (private).
- Local candidate: `deployment/kaggle/s0-public1-action-demand-v14-candidate`.
- Pre-push local hashes:
  - `s0_public3_full_timing.ipynb` = `b29258ba4e991fef08a508994f65fc1a7e55802ae1e3b9c042aba71c3fdaa1f0` (matches frozen `candidate-hashes.preupload.sha256`)
  - `kernel-metadata.json` = `915cc9c94c56980ed6c9a637d5b2402bdad4c46f968824e56a98f2f1b0d6e197`

## Immediate remote readback (kernel version 12)
- Pulled remote source + metadata immediately after push.
- **All 4 notebook cell sources byte-identical to the frozen candidate.**
  - concat-source SHA-256 (local == remote) = `084946e26897cbc522a867618ca656d70ac09bb17e7152ee083fc1f270f617ba`
  - kernelspec and language_info identical.
- Metadata material fields match intent:
  - wheelhouse dataset = `pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v11` (was v10 on V13)
  - `enable_gpu=true`, `machine_shape=NvidiaRtxPro6000`, `enable_internet=false`
  - `competition_sources=["arc-prize-2026-arc-agi-3"]`
  - `docker_image` = `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461` — **identical to the image V13 ran on** (no base-image drift)
  - Remaining diffs are Kaggle server-side only: `id_no`, `keywords:["gpu"]`, `code_file` rename, `dataset_sources` reordering.

## V11 wheelhouse dataset pre-launch check
- `pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v11` visible, private, `datasetId 11874096`.
- Local manifest hashes exact:
  - `wheelhouse-manifest.json` = `518f8407b0ab1234430fc0c167e2ecd62ce70318a62292f8397e1a7ea4ed3862`
  - `solver-source-manifest.json` = `22cd2034b4341606d1725f98e6552754d294d597f2ecfa620f8decb46ee98f4e`

## Local verification re-run (Linux)
- Focused Solver tests (`test_action_batch`, `test_llm`, `test_worker`) with `TYCHO_SANDBOX_RUNTIME=host`: **76 passed**.
- V14 candidate tests: **26 passed**.
- `git diff --check`: clean.

## GPU quota at launch
- 21 h remaining, resets in ~53 h. Card worst-case ~2.3 h.

## Status
- Kernel version 12 status after push: **RUNNING**.
- Awaiting terminal state; then download output + source and write TERMINAL-RECONCILIATION.
