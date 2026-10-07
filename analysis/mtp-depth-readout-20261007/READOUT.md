# MTP sweep v1 readout, 2026-10-07

The notebook completed, but the serving experiment failed before measurement. The frozen analyzer independently reproduces `INVALID: baseline M3A missing`. No speculative-depth throughput comparison, profile, or ARC-score result exists.

## Observed failure

All six launches, in the authorized order M3A/M1/M2/M3C/M4/M3B, failed health startup in 42-50 seconds. Each log ends with the same `RuntimeError: unable to page-lock the PLE host table (320001536, 160)/torch.bfloat16: cudaHostRegister failed: cudaError.???`. A child then sends SIGQUIT, and SGLang kills the process tree, explaining the parent exit code -9. This is an initialization exception; the evidence does not establish an external OOM kill. The underlying page-lock failure remains unqualified.

The notebook GPU inventory reports **two Tesla T4 GPUs, each 15360 MiB**, and torch reports capability (7,5). The frozen card requires one RTX PRO 6000. Local metadata and pulled remote metadata both say `machine_shape=NvidiaRtxPro6000`. Inspection of the installed Kaggle SDK shows that `kernels_push` sets this request field; local request serialization retains it as `machineShape`. Therefore the evidence establishes a requested-versus-observed hardware mismatch, but does not establish why Kaggle assigned T4s. Setting the same field again is not a demonstrated repair.

`wrapt` is missing during sitecustomize startup, but the server continues into model initialization. It is an additional environment defect, not the demonstrated exception that kills the server. `nsys` is absent, so profiling was deferred.

## Identity and checks

- Kernel `pranavagrawai/mtp-depth-sweep-20261006`, version 1, ID 137460273. API status COMPLETE, no notebook-level failure message, checked 2026-10-07T12:23Z.
- Last failed launch finished at 2026-10-07T09:00:44Z. Notebook log ends near 422 seconds of elapsed time. This is log duration, not an accelerator-quota accounting receipt.
- Shipped code `28ef795f16f5165dc6c71426569bf7a16968d970`, prelaunch tag `mtp-depth-sweep-prelaunch-20261007`, manifest SHA `c21852dc33715d0efe6b4d62b70d7869040b859b4f1c2ec36ad1ea22e64b9713`.
- Runtime readback matches all 37 entries in the frozen manifest, with no mismatches. Recomputed verdict equals the saved verdict. Prelaunch verifier still passes and frozen task files remain unchanged.
- All 22 session output files were collected under `raw/`; saved remote notebook and metadata are under `remote-notebook/`. Explicit-version GetKernel pull returned 403; pulling without a version suffix succeeded, and all returned cell sources match the frozen notebook. The output SDK parses a version suffix but does not transmit it to ListKernelSessionOutput; only version 1 was authorized and accepted in this thread.

## Judgment and next useful action

This attempt gives no evidence for or against changing M=3. Preserve M=3 and the frozen experiment. Before any new full sweep, resolve accelerator selection and add an early runtime hardware assertion, then qualify one incumbent startup on the intended machine with page-lock diagnostics. A fresh GPU launch requires new authority; no retry or launch was performed during this readout.

## Competition eligibility correction

Pranav confirmed after this readout that RTX PRO 6000 access is restricted to the ARC-AGI-3 competition. Both frozen and retrieved remote metadata have `competition_sources=[]`; the notebook was not attached to the competition. This was a missed launch prerequisite. The prelaunch checks verified the declared machine shape without requiring the competition attachment, and the notebook printed actual hardware without asserting it.

This explains the hardware eligibility mismatch under the restriction confirmed by Pranav. It does not independently establish why CUDA host registration failed. Any replacement must attach the competition, verify the saved attachment, and abort before installation/model startup unless runtime inventory shows exactly the intended RTX PRO 6000. The failed freeze remains unchanged and no new launch is authorized by this correction.
