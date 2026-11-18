# Target containment remains unqualified after a startup instrumentation abort

The package was built, reviewed through the existing Superset collaborator, tested locally, and pushed exactly once. Kaggle version 1 finished, but the probe stopped before any candidate or inspection execution. **This run does not provide negative or positive evidence about target chroot feasibility.**

- Kernel: https://www.kaggle.com/code/pranavagrawai/containment-q-20260913t150421z
- Exact version: **1**, confirmed by the returned launch receipt and `SERVER-VERSION.json`.
- Exact-version terminal status: **COMPLETE**, confirmed again in `status-exact-version-final.json`.
- Raw classification: `UNSUPPORTED_OR_QUALIFICATION_FAILED`.
- Assessed outcome: **PRECONDITION / INSTRUMENTATION ABORT; TARGET CONTAINMENT NOT QUALIFIED**.
- Error: `RuntimeError:TARGET_ACCELERATOR_NOT_OBSERVED`.
- Remote cases executed: **0**. No remote rootfs, chroot, identity drop, seccomp installation, parent-death arming, or generated-code execution occurred.
- No retry, model inference, model loading, competition submission, deployment, or git commit occurred.

## Why startup stopped

The frozen `probe.py` uses `shutil.which('nvidia-smi')`. The frozen notebook starts the trusted assessor with this explicit environment:

```json
{"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C"}
```

The remote `platform.json` contains `accelerator: null`. In this implementation, that means the binary was not discoverable on the assessor's PATH. The observed mount table maps `/var/lib/nvidia/bin` to `/opt/bin`, which that PATH omits. These facts support an instrumentation PATH omission. The exact `nvidia-smi` file and physical accelerator were not observed, so they are not asserted to exist or to be absent.

The GPU guard runs before temporary rootfs creation and before any sterile runner launch. `result.json` has an empty case list and `cleanup_complete: true`. Retrieval returned only `platform.json` and `result.json`, with no per-call rootfs manifests. The ancillary Kaggle log was discarded without reading it. See `POST-RUN-DIAGNOSIS.json` for the source-derived PATH and mount evidence.

The startup guard failed closed as designed. It also prevented the intended containment measurement. The frozen code and card remain unchanged; the omission was not repaired and repushed under the consumed identity.

## Provider and platform facts

The server returned `machine_shape: NvidiaRtxPro6000`, private mode, internet disabled, and empty dataset, model, kernel, and competition sources. This establishes the provider-reported configuration, not direct physical GPU observation.

The returned container image matches the image requested from the prior serving package:

```text
gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461
```

Observed runtime facts were Ubuntu 22.04.5 LTS, Linux `6.12.90+`, x86_64, glibc 2.35, and Python 3.12.13 at `/usr/bin/python3`. The assessor had real, effective, and saved UID/GID 0 and supplementary group 0. Its capability mask was `a80425fb`; the `CAP_SYS_CHROOT` bit was present. **A capability bit is not an exercised chroot boundary.** Container identity matching also does not establish identical host kernels or mounts across runs.

`--timeout 300` was sent through the official CLI. `PROVIDER-CONTRACT.json` independently exercises the installed CLI's request construction with the outbound service mocked and records `session_timeout_seconds: 300`. That proves the request, not independent provider timeout enforcement. No exact-session cancellation or user-stop mechanism was qualified.

The read-only monitor observed terminal status 32.768 seconds after the launch invocation. The assessor reported 0.008407 seconds before its startup abort. Displayed GPU quota changed from 1.37h used / 28.63h remaining to 1.38h used / 28.62h remaining. The **displayed delta is 0.01 GPU-hours**, below the 0.5 GPU-hour cap. The display is rounded and is not an exact per-job bill. The local monitor exited with code 0 after observing COMPLETE.

## Local implementation and property matrix

The fresh-exec runner starts without assessor data. It uses an exact, hashed, root-owned read-only rootfs; chroot and an inside cwd; closed stdin and extra descriptors; an allowlisted environment; zero real/effective/saved privilege; cleared ambient, bounding, permitted, effective, and inheritable capabilities; no_new_privs; and a native x86-64 syscall allowlist. Unknown calls, x32, networking, process creation, exec, process inspection, namespace changes, and privilege changes are denied. Python is already started before chroot. The rootfs contains selected stdlib source, the current call, candidate source, and an inspection-only synthetic prefix. No host library trees, proc, sysfs, devices, executables, writable scratch, or assessor data are copied.

Local tests use the same runner in a mapped user namespace, not bubblewrap. They exercise real chroot but do not establish Kaggle support.

| Property | Local evidence | Target evidence |
|---|---|---|
| Benign current-argument call | Passed in both modes | Not exercised |
| Inspection-only prefix; corpus, future, verifier excluded | Passed in both modes | Not exercised |
| Absolute path, `../`, symlink, openat directory-FD escape, cwd | Passed in both modes | Not exercised |
| Host proc, device, stdin, environment, extra FDs | Passed in both modes | Not exercised |
| Loopback, socket calls, ptrace, privilege and namespace changes | Passed in both modes | Not exercised |
| Fork, clone, subprocess, exec, x32, prlimit64 | Passed; separate self/other-PID regression passed | Not exercised |
| Wall and output limits; child reaping | Passed in both modes | Not exercised |
| Supervisor SIGKILL kills sealed child | Real mapped-root regression passed and reaped child | Not exercised, including arming |
| Forged PASS rejection; required seal receipt | Passed in both modes and negative tests | Not exercised |
| Manifest digest drift and unsafe rootfs rejection | Passed | Not exercised |
| Cleanup scoped to owned path and required for qualification | Passed | No rootfs was created; startup abort cleanup true |
| Target accelerator guard | Absent, wrong, and multiple-GPU fixtures rejected | Rejected missing binary observation before execution |

`LOCAL-ROOT-TESTS.log` records **30/30 tests passed**, including actual supervisor-kill termination and reaping. `LOCAL-TESTS.log` records **30 tests, 28 passed, two root-only skips**. `local-real-final-2/result.json` records **16/16 end-to-end cases passed**, eight cases in each mode, with cleanup complete.

These are finite synthetic checks, not proof against every native-code or kernel attack. Limits are per process. The design denies descendants and supplies no writable tree, but it does not establish aggregate cgroup kernel-resource accounting or a general hostile-code production service.

## Independent Superset review

The existing reviewer used workspace `e3401039-8f11-4473-821f-c114a2bfd55b`, terminal `2ade96aa-e767-4645-857e-e1e8a6773de1`. No new agent was created and the reviewer made no edits or launches.

The first review held launch for unrestricted `prlimit64`, child survival after supervisor death, missing seal acceptance in negative tests, cleanup gating, and unchecked manifest digest values. All were corrected before freeze. The reviewer independently reran the ordinary and mapped-root suites, including supervisor-kill, and resolved the hold for the exact final manifest:

```text
2648407397a6f1a25c13217e6914c099c0f8d24833442b9504669910b7bacdca
```

See `REVIEW-FINAL-RAW.json` and `REVIEW-RESOLVED.json`. `PRELAUNCH.json` covers 12 files, and the postrun audit reconciles their hashes unchanged. Review approval covered the bounded probe, not advance qualification of the remote platform.

## Artifacts and repeatable audit

Analysis root:

```text
/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/kaggle-containment-qualification-20260913T150421Z
```

Deployment package:

```text
/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/deployment/kaggle/kaggle-containment-qualification-20260913T150421Z
```

The key receipts are `remote/qualification/result.json`, `remote/qualification/platform.json`, `AUDIT.json`, `POST-RUN-DIAGNOSIS.json`, `SERVER-VERSION.json`, `status-exact-version-final.json`, and `quota-before.json` / `quota-after.json`. The package contains `build.py`, `qualification.ipynb`, `kernel-metadata.json`, and `BUILD.json`. All implementation changes are confined to the two new directories; prior probes and experiments were not edited.

From the repository root, rerun the local artifact audit without provider mutation:

```bash
python3 -B analysis/kaggle-containment-qualification-20260913T150421Z/audit_remote.py
```

The expected audit outcome is the startup-abort classification with `checked_cases: 0`, exact image readback, unchanged frozen hashes, terminal COMPLETE, and displayed quota delta 0.01h. An exit-0 audit validates that reported failure; it does not turn the probe into a containment pass.

A further target qualification needs a new identity, offline qualification of trusted GPU discovery under the actual sanitized environment, review, and renewed launch authorization. No model-backed run is permitted by this result.
