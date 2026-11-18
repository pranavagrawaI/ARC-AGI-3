# Local harness audit completed; remote containment remains unqualified

The fresh candidate executes the generated notebook, dedicated worker, assessor, rootfs builder, sealed runner and receipt audit locally. The final notebook passed all 16 synthetic containment cases with real chroot, identity/capability dropping, seccomp and resource limits. It also passed with conflicting cached notebook modules. No paid launch, provider request, model load, competition submission or git commit occurred.

This is finite local qualification, not proof that every bug is absent. The constructor is still a separate offline regression candidate, not a deployable integration of this runner.

## Final evidence

- `verification-r2/VERIFICATION.json`: successful full verification.
- `verification-r2/package-success/`: exact generated notebook, all 16 cases passed.
- `verification-r2/package-cached_modules/`: same notebook with four conflicting cached modules, all 16 cases passed. The notebook did not use the cached helpers.
- `verification-r2/package-loader_missing/` and `package-malformed/`: notebook exit 1, qualification false, zero cases executed, 16 unexercised. Process, worker, audit and qualification receipts remain separate.
- `verification-r2/local-real/`: actual local runtime facts and all 16 direct probe cases. The local GPU query observed an RTX 5060 Laptop GPU, not the requested RTX PRO 6000. The query parser rejected it as the target; `--local` deliberately does not claim target qualification.
- `verification-r2/package-parentkill-sealed/VERIFIED.json`: killing only the notebook killed and reaped its dedicated worker, assessor and already-sealed runner with SIGKILL.
- `verification-r2/BASELINE-REPRODUCTIONS.json`: bounded reproductions against untouched 160618Z source.
- `REVIEW.json`: independent review scope and exact reviewed hashes.
- `MANIFEST.json`: final file digests. `verify_manifest.py` recomputes them.

The full suite ran 53 mapped-root tests with no skips. The unprivileged run discovered 53 tests, passed 46 and skipped seven root-required checks. The untouched baseline suite passed 35 mapped-root tests. The constructor suite passed 46 tests and its command-line launch guard rejected `--execute`. Both frozen containment package verifiers passed without source changes.

The parent-death supplement first used process discovery before observing sealing and failed a signal assertion. The corrected measurement waits for the kernel-reported dropped UID, no-new-privileges and seccomp state before killing the notebook. The final result above passed. `verification-r2/package-parentkill/` preserves the initial unsuccessful measurement and is not positive evidence.

## Defect ledger

| Finding | Evidence status | Repair and regression |
| --- | --- | --- |
| Located NVIDIA binary could not resolve its runtime library under the notebook environment | Remote exit 12 was preserved in frozen 160618Z. A compiled synthetic `dlopen` executable independently reproduced exit 12 locally under the exact frozen notebook AST environment | `accelerator.py` supplies only bounded fixed trusted library directories to trusted diagnostics. The same compiled executable returns 0 after correction. No synthetic result is labeled real GPU evidence |
| Driver query output accepted malformed strings containing the expected GPU name | Source-supported permissive substring gate; malformed fixture exercises the repaired caller chain | Parse exactly one CSV row and four fields, validate target name, GPU UUID, driver version and positive MiB value. Reject missing, multiple, wrong-device, malformed, timeout and output-limit observations |
| Accelerator subprocess could accumulate unbounded output and collapsed prerequisite diagnostics | Source-supported old `capture_output` path | Query and independent library-load checks use bounded draining. Record paths, loader-directory evidence, exit/status/stdout/stderr and separate NVML/CUDA loading failures. No caller loader environment or credentials reach candidates |
| Directory FD and listener escaped cleanup when socket creation, bind or listen failed | Reproduced against frozen source. It reported filesystem cleanup complete while its directory FD remained open and listener close was omitted | Start ownership cleanup before acquiring either resource. Regressions inspect actual descriptor closure for create, bind and listen failures |
| A changed rootfs reached execution before a second admission hash check | Reproduced through frozen `main`, stopping at the execution call rather than executing the changed candidate | Revalidate the complete manifest immediately before launch. The changed source now fails before execution |
| Empty directories were outside the old exact allowlist, and post-run checks only compared listed files | Empty-directory acceptance reproduced; post-run omission source-supported | Validate the exact directory and file inventory plus hashes before and after each case. Added files fail. Preserve the executed receipt as failed if post-run validation fails |
| A descendant could survive when its session leader had already exited | Source-supported cleanup condition, exercised by a bounded disposable child regression | Always clean the invocation-owned process group, including when its leader has exited, and reap the leader |
| Runner startup could adopt a new parent before capturing its expected identity | Source-supported setup race | The trusted launcher passes the expected parent PID. Startup rejects a mismatch; UID dropping still re-arms PDEATHSIG. Existing real supervisor-kill test passes. The added package hierarchy also passed actual notebook-only SIGKILL testing |
| Generic audit accepted status from another target/version and ignored a failed status query | Independently reproduced by reviewer against frozen source | Require successful exact-version status, matching server identity/version and notebook bytes. Full audit regressions cover success and failure receipts, wrong target/version, return code, source drift, missing manifests, forged outputs/seals and zero-case false pass |
| Notebook completion did not require qualified receipts and exceptional startup lacked a durable package result | Source-supported prior bootstrap behavior | Persist package and worker lifecycle receipts. Run the receipt audit after the assessor. Nonqualified execution now raises in the notebook; `COMPLETE`, subprocess exit and qualification are not interchangeable |
| Notebook `sys.modules` cache could replace the freshly unpacked assessor/auditor | Reviewer reproduced a false `qualified=true` without an assessor or qualification directory in the first audit candidate | Notebook imports no candidate helper modules. A dedicated `-I -S` worker rehashes assets and checks their import origins; the assessor also starts in a fresh `-I -S` interpreter. Exact-notebook cached-module regression passed locally and independently |

The sealed runner retains its default-deny native x86-64 filter, x32 denial, cross-process `prlimit64` denial, no outside descriptors/stdin/environment, no host proc/devices, fresh trusted exec, irreversible UID/GID/group/capability drop, and no writable candidate rootfs. No security check was disabled to obtain a pass.

## Pipeline coverage and boundaries

1. **Packaging and bootstrap:** AST parse, exact bootstrap reconstruction, decoded asset equality, metadata contract and hashes reconcile. The package and helper scripts have a fresh identity. The notebook uses the actual `sys.executable`; local execution used `/usr/bin/python3`. Cached helpers cannot substitute for packaged helpers.
2. **Trusted prerequisites:** bounded executable discovery, independent NVML/CUDA loading, parsed device observation, OS/architecture/identity/capabilities, runtime import subprocess and stdlib-source presence are recorded together. The filter uses native BPF through libc, so `libseccomp` is not required. Loading `libcuda.so.1` alone does not establish a usable CUDA device or exclude a driver stub.
3. **Rootfs and execution:** every case uses fresh exec and the exact source-only stdlib allowlist. Required native extensions and shared objects load before chroot; the probe does not copy an interpreter, arbitrary host library tree, CUDA devices or site-packages into the rootfs. The real local `-I -S` startup and later JSON/runtime imports passed.
4. **Case coverage:** candidate and inspection each exercise benign execution, filesystem/future/verifier/proc/device access, inherited channels, denied syscalls and loopback, read-only state, wall timeout, output cap and forged output. Inspection alone receives its synthetic visible prefix. All 16 expected identities are independently audited; zero cases cannot qualify.
5. **Lifecycle:** direct-child reaping, exited-leader process-group cleanup, assessor death and notebook-to-worker-to-assessor-to-sealed-runner death were exercised. Acquisition faults and manifest drift prevent qualification. Atomic result replacement preserves completed case receipts. Abrupt notebook SIGKILL cannot guarantee a final lifecycle write; the death test establishes child termination, not durable completion or provider cancellation.
6. **Resources:** CPU, address-space, file-size, descriptor and process limits remain per-process controls. Process creation is denied. These checks do not establish aggregate kernel memory, PID, storage, GPU billing or provider-wide resource accounting. No fork or memory bomb was used.
7. **Constructor seam:** the 46-test suite exercised captured/mock-HTTP continuation, inspection, submission, accounting, audit and existing bubblewrap isolation. Its request contract still contains OpenRouter model/routing/reasoning fields, while `run.py` names a local HTTP endpoint and blocks command-line execution. Its executor dispatches `initialize`, `predict` and `assimilate`; this qualification runner dispatches synthetic `run(*args)`. Those paths are not wired together. Passing these separate suites does not qualify a constructor on Kaggle or a vLLM request contract. Model-only outcome criteria were not broadened.

## Remaining work, separated by kind

Target-only checks remain unexecuted on the requested Kaggle GPU image: actual NVML files and dependencies under the fixed directories, executable permissions and mounted-image ownership, reported and observed single RTX PRO 6000, exact target Python/stdlib/native-extension behavior, real chroot/group/UID/capability/seccomp support, all containment cases, parent-death behavior, invocation cleanup and write ownership. Provider image readback, exact remote version, terminal status, timeout enforcement, quota/billing and cancellation also require a separately authorized run. Local monitor exit is not remote teardown.

There is also a non-target-only integration gap. A deployable constructor still needs an explicit bridge for the existing method ABI and inspection inputs, an actually qualified local model-serving request contract, and a construction-only notebook. This audit did not build that broader feature or weaken isolation to bypass the gap. A successful local synthetic fixture does not authorize another paid launch.

Independent review found no further confirmed blocker in the bounded final delta. It independently reran the exact cached-module notebook fixture and verified hashes before and after. Its broader-suite statements distinguish owner evidence from independent execution. This is not an arbitrary-code security proof or an assertion that no further defects exist.

## Reproduce without paid calls

Run from the repository root. The full script requires the existing Linux user-namespace mappings, `bwrap` and `gcc`. Use a new output directory each time.

```bash
python3 -B analysis/kaggle-containment-audit-20260913T174113Z/verify_manifest.py
python3 -B analysis/kaggle-containment-audit-20260913T174113Z/verify_all.py /tmp/containment-audit-rerun
python3 -B analysis/kaggle-containment-audit-20260913T174113Z/hierarchy_death.py /tmp/containment-parentkill-rerun
```

`verify_all.py` never builds, launches or calls a provider. It verifies the existing package, so changed source requires a new explicit build and evidence run rather than silently validating old notebook bytes.
