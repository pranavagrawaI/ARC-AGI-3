# Invalid allocation; no engine measurements

Kaggle accepted version 1 with `machine_shape=NvidiaRtxPro6000`, but the executed
runner reported `Tesla P100-PCIE-16GB` (SM60). The hardware assertion stopped
execution before model discovery, model loading, server launch or any benchmark
HTTP admission. No FP8 or MTP arm ran. This is neither a candidate regression
nor an engine-speed result.

Evidence:

- `kaggle-output/isolated-engine-20260905/runner.log`: actual Torch device warning
  and assertion traceback at the RTX PRO 6000 gate.
- `kaggle-output/isolated-engine-20260905/FAILURE.json`: terminal AssertionError.
- `remote-source/kernel-metadata.json`: returned provider metadata still records
  `NvidiaRtxPro6000`, private/offline, requested dataset sources.
- Pulled remote notebook code cells exactly equal the locally frozen candidate
  cells. Different serialized notebook hashes are not a source-code difference.
- `kaggle-output/qwen38-isolated-engine-20260905-v1.log`: runner failure around
  notebook time 57 seconds; output conversion ends around 65 seconds. Queue and
  artifact-finalization time are not inference time or a measured GPU bill.
- Installed Kaggle request construction was checked before launch and again
  afterward: 3600-second timeout, requested RTX PRO 6000, no transport during
  checks. Deterministic package checks and local metric tests pass.

The exact reason requested and actual hardware disagree is unresolved. Historical
project notes also describe a P100 misallocation; those notes do not establish
the cause of this occurrence. Do not change CUDA, checkpoint or manifests to
make the P100 run. No retry was launched. A confirmed target-GPU allocation is
needed before a fresh, separately identified three-arm run.

Independent dispatcher status remains LOCAL_PACKAGE_QUALIFIED. This allocation
failure neither weakens those local process results nor promotes the dispatcher
to integrated Kaggle qualification. The tested polling/receive-gap investigation
stays closed.
