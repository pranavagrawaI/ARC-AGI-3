# Native renormalization candidate: local kernel PASS, target replay pending

Prepared deployment/kaggle/mtp-sampling-repair-20260906/sampling_repair.ipynb.
No Kaggle push, model run, game action or submission was performed. Failed donor
notebooks, cards and raw results were not modified.

## Diagnosis

The retained MTP replay server.log starts with `Failed to get device capability:
SM 12.x requires CUDA >= 12.9.` before the later SM75 exception. Inspection of
FlashInfer0.6.17 explains the chain: CompilationContext catches the SM120/compiler
version error and leaves TARGET_CUDA_ARCHS empty; check_cuda_arch subsequently
raises the generic SM75 message. A CPU-controlled test against the actual installed
FlashInfer reproduced that empty-target failure and the compatible-version branch.
The exact selected compiler path was not captured remotely; do not invent it.

The PyTorch sampler flag does not control this path: pinned eagle_utils.eagle_sample
directly imports sgl_kernel top_k_renorm_prob and top_p_renorm_prob for stochastic
verification. sglang-kernel0.4.6.post1 sampling.py prefers FlashInfer when importable,
but also contains wrappers for EXISTING precompiled native renormalization ops.
Thus missing native kernels was not established by the crash.

Source inspected: pinned SGLang wheel from failed-run bootstrap; exact
sglang-kernel0.4.6.post1 package; FlashInfer0.6.17 PyPI wheel SHA256
ab7ce3aec42faf2f6079aa840c2407ac00ecd277a0524cdb708a8ff4a64dad90.
Relevant files: flashinfer/compilation_context.py, jit/core.py, jit/cpp_ext.py;
sgl_kernel/sampling.py; sglang/srt/speculative/eagle_utils.py.

## Repair

Opt-in existing native implementations through SGLANG_SPEC_NATIVE_RENORM=1.
install_sampling_patch.py verifies version0.4.6.post1 and original sampling.py SHA256
80f01a3812ff7be617c169ab82828a09fa4f22969c42cdcaedf3a01c628bfdf8 before changing
only the optional FlashInfer selection block. The default remains unchanged without
the variable. Model manifest, installed wheel identities, tree verifier, acceptance
settings, temperature1/top-k20/top-p0.95, tokenizer and reasoning are not replaced.
The source overlay is a NEW engine candidate, bound separately in manifest/artifacts.

Rejected route: selecting installed CUDA13.3 passed architecture discovery but
local compilation exposed mismatched13.0 runtime headers plus an unsupported local
GCC16. No compiler/version guard was bypassed. The Kaggle package inventory likewise
lists CUDA13.3 nvcc and13.0 runtime, but its complete header layout was not captured.
toolchain.py remains diagnostic-only and is not embedded in the notebook.

## Local checks

- Actual GPU: RTX5060Laptop, SM120,8GB; no27B model load. Torch2.13.0+cu130,
  FlashInfer0.6.17, sglang-kernel0.4.6.post1, TVM-FFI0.1.11. These versions match the
  relevant failed-run inventory; the entire local environment is not a Kaggle clone.
- Native top-k20 then top-p0.95 on four248320-vocabulary probability rows PASS with
  FlashInfer JIT disabled. Against Torch probability references, maximum absolute
  errors1.1176e-8 and1.4901e-8. See local-kernels/sampling.json and sampling-patch.json.
  Timings include test operations and are NOT a kernel or MTP speed benchmark.
- Additional local uniform/point-mass checks preserve normalized finite probability;
  ties retain all128 uniform entries, point mass retains one. Not a complete
  distributional-equivalence proof against every FlashInfer edge case.
- Six local tests PASS: compiler diagnosis/selection controls, real FlashInfer
  empty-target reproduction, real subprocess-death HTTP-watchdog test, result/error
  propagation, corpus/notebook binding. The selected repair does not use the compiler
  selector tested for diagnosis.
- Generated successor runner passed the retained full14-case mocked replay harness:
  both arms complete, invalid outputs retained, four fixed diagnostics. Spec metadata
  is fixture data here, not evidence of local EAGLE execution.
- Deterministic notebook rebuild/check and focused Ruff checks PASS.
- Installed Kaggle CLI upload construction with transport mocked PASS: private,
  RTX PRO6000, offline, arc-prize-2026-arc-agi-3,7800s cap. Zero transport calls.

Initial local setup used cached Torch2.14, which did not load the pinned native
library ABI; corrected to2.13 before claiming kernel success. No unsupported ABI or
compiler overrides are shipped. Local test dependencies live in this probe's venv.

## Qualification order and remaining limits

The new notebook first runs native kernel checks BEFORE model loading. It then runs
the frozen baseline/MTP replay with the original omitted sampling fields intact.
Warmup request and response are retained; MTP must show positive spec_verify_ct.
The replay watchdog aborts on confirmed engine exit instead of waiting900s; cleanup
errors are retained separately from the primary generation failure.

This watchdog changes the replay, not the old integrated pilot's outer worker.
Do not rerun that pilot unchanged. Full stochastic EAGLE, rejected hybrid-state
correctness, tool reliability and gameplay remain unqualified. The bounded card
permits selection for a subsequent integrated candidate only after target evidence.
No inference speedup is claimed from this preparation.
