# Containment remains untested remotely: NVIDIA library-loading abort

**The binary-path fix worked, but this run still did not reach containment.** `/opt/bin/nvidia-smi` was found and executed. It exited **12** with:

> NVIDIA-SMI couldn't find libnvidia-ml.so library in your system. Please make sure that the NVIDIA Display Driver is properly installed and present in your system.
> Please also try adding directory that contains libnvidia-ml.so to your system PATH.

The unchanged target guard then stopped with `RuntimeError:TARGET_ACCELERATOR_NOT_OBSERVED`. This is a trusted startup/driver-library observation failure, not a demonstrated chroot failure or GPU absence. Physical accelerator identity remains **NOT_ESTABLISHED**.

## Actual run

- URL: https://www.kaggle.com/code/pranavagrawai/containment-q-20260913t160618z
- Exactly **one push, version 1**; no retries or model loading/inference.
- Exact-version status **COMPLETE**, independently read back. COMPLETE does not mean qualification passed.
- Required remote containment cases: **16; executed 0; passed 0; failed cases 0; not exercised 16**. The prerequisite failed before rootfs creation, child startup, sealing, or parent-death arming. Cleanup true means there was no created rootfs to remove.
- Requested and provider-reported shape: `NvidiaRtxPro6000`; not a direct hardware observation.
- Requested and provider-reported image match:
  `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`
- Observed runtime: Ubuntu 22.04.5 LTS, Linux `6.12.90+`, x86_64, Python 3.12.13 at `/usr/bin/python3`, assessor UID/GID 0. Image digest is provider metadata, not in-container attestation.
- Assessor elapsed **0.011919 seconds**; terminal observed **32.848903 seconds** after launch invocation. Foreground monitor exited 0 after COMPLETE; no active monitor remains.
- GPU quota before/after: **1.38h used / 28.62h remaining**, displayed delta **0.00h**, within the 0.5 GPU-hour cap. Rounded account-level display is not an exact per-job bill and does not prove zero consumption.

Mountinfo places NVIDIA binaries at `/opt/bin` and NVIDIA libraries at `/usr/local/nvidia/lib64`. The exact library file and a corrective loader setting were not tested. No further run occurred.

## Change and qualification

Created a fresh analysis identity and matching deployment directory; preserved the prior failed `20260913T150421Z` run unchanged. Only runtime change was trusted absolute NVIDIA executable discovery, usable-file checks, and actual execution/error receipts under the existing scrubbed environment. `runner.py` is byte-identical; AST comparison establishes every prior probe function except `snapshot` unchanged. The strict GPU guard and all previously reviewed containment/resource/cleanup behavior remain unchanged.

- Ordinary tests: **35 total, 33 passed, 2 root-only skips**.
- Mapped-root tests: **35/35 passed**, including real supervisor-kill termination and child reaping.
- Actual mapped-root end-to-end containment: **16/16 passed**, cleanup complete.
- Added five discovery regressions: exact scrubbed PATH with absolute `/opt/bin` dispatch, unavailable binary, non-executable file, execution errors/nonzero/timeout, wrong GPU. Fixtures are synthetic, not remote hardware evidence.
- Existing Superset terminal independently reran ordinary and mapped-root suites and approved the exact delta without blocking findings. No new agent or reviewer launch.
- All **12 frozen artifact hashes**, embedded assets, server-pulled notebook source, and retained remote receipt hashes reconcile. Official CLI offline request construction confirms `session_timeout_seconds:300`, exact shape/image, private, internet-off, and no attached sources.
- Separate post-run regression replays the actual exit-12 accelerator receipt and confirms rejection before rootfs creation. Frozen candidate/card/tests were not modified after launch.

Review manifest SHA256: `e799f370554d1dcf48408eccb043328a264ce03369e3c8d879563f46d28ae4b3`.

## Audit and evidence

Analysis root:
`/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/kaggle-containment-qualification-20260913T160618Z`

Deployment root:
`/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/deployment/kaggle/kaggle-containment-qualification-20260913T160618Z`

- `AUDIT.json`: frozen package, provider configuration, terminal status, quota reconciliation; zero checked containment cases.
- `POST-RUN-DIAGNOSIS.json`: independent failure audit, exact exit/output, case counts, source/readback/receipt reconciliation, failure-specific regression.
- `remote/qualification/{result,platform}.json`: actual allowlisted remote evidence. Only these two files were retained; ancillary log deleted without reading.
- `SERVER-VERSION.json`, `status-exact-version-final.json`, `RETRIEVAL.json`, `quota-before.json`, `quota-after.json`, `MONITOR-EXIT.json`: attribution/lifecycle/accounting evidence.
- `LOCAL-VERIFICATION.json`, `LOCAL-ROOT-TESTS.log`, `local-real-final/result.json`, `REVIEW-FINAL-RAW.json`: local and independent review evidence.

Repeat from the repository root:

```bash
python3 -B analysis/kaggle-containment-qualification-20260913T160618Z/audit_remote.py
python3 -B analysis/kaggle-containment-qualification-20260913T160618Z/postrun_audit.py
```

Both exit 0 for a correctly audited **failed prerequisite**, not a containment pass.

Versioned metadata pull `/1` returned 403. Read-only unversioned pull succeeded; provider `current_version_number:1` and exact notebook source comparison reconcile the readback with the sole version. Exact-version status/output retrieval succeeded.

## Limits and next action

No remote containment claim is supported. The single authorized attempt is consumed; stop, with no retry or inference. Any further correction requires separate qualification and launch authorization. Local synthetic success is not general hostile-code or kernel-security proof. The 300-second provider timeout was requested, not independently tested for enforcement; no exact-session cancellation was qualified. Notebook 180-second and assessor 150-second limits do not independently guarantee remote shutdown or billing. Actual terminal status was observed in this run.
