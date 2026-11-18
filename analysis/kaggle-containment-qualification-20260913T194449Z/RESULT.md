# Kaggle containment qualification: not qualified

**Exactly one push was made. Version 1 terminated with `ERROR`. No retry, image substitution, model loading, inference or competition submission occurred.**

Actual job: https://www.kaggle.com/code/pranavagrawai/containment-qualification-20260913t194449z (version 1).

## Concrete outcome

The corrected trusted NVIDIA loader worked on Kaggle: `/opt/bin/nvidia-smi` exited **0**, and independent `libnvidia-ml.so.1` / `libcuda.so.1` checks both returned `LOAD_OK`. The actual query reported:

```text
Tesla P100-PCIE-16GB, GPU-543c532b-c511-c675-a565-bf01208405e0, 580.159.04, 16384 MiB
```

Both the explicit request and server metadata named **NvidiaRtxPro6000**, but the observed GPU was a **Tesla P100**. The strict device check correctly rejected it with:

```text
RuntimeError:TARGET_ACCELERATOR_NOT_OBSERVED:ACCELERATOR_NAME_MISMATCH
```

| Remote case accounting | Count |
|---|---:|
| Required | 16 |
| Executed | 0 |
| Passed | 0 |
| Failed case receipts | 0 |
| Unexercised | 16 |

This is a failed target prerequisite, not a demonstrated containment escape or a successful target qualification. No case rootfs or sealed case runner was created. Remote sealing, parent-death arming, canary isolation and per-case cleanup remain **NOT EXERCISED**. The trusted assessor exited 2 and was reaped; the dedicated worker exited 2; package temporary-directory cleanup was confirmed. Startup failure receipts survived as intended.

## Provenance and deviations

- Server metadata confirms the exact requested image `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`, private mode, internet off and no data/model/kernel/competition sources. The digest is provider metadata, not runtime digest attestation.
- Kaggle derived the new URL slug from the title, changing requested `containment-q-20260913t194449z` to `containment-qualification-20260913t194449z`. The frozen launcher caught this after the sole successful push and stopped. No second push followed. Read-only retrieval reconciled the returned URL, actual version 1, server source and receipt identities. The original requested slug remains absent.
- The strict original target audit rejects the slug discrepancy with `AUDIT:STATUS_TARGET_VERSION_MISMATCH`; this rejection is retained, not relabeled a pass. The supplemental audit establishes provenance for the actual returned job separately.
- Kaggle reformatted notebook JSON. Raw source digests differ, but **complete parsed notebook equality and exact executable strings/embedded asset bytes match** the frozen notebook. Canonical JSON matches the prelaunch wire digest. Raw returned source and receipt digests remain preserved.
- The immutable audited 174113Z baseline still verifies all 268 files with manifest `dc21201f6b2dc584779d4261c572873aa310c333013e14ddc96d8189d10353ff`. New prelaunch manifest `264c9b833d1d5bbc12816fda185b1da56c57d9b77e1a73a3c7d9863d8538bc84` also reconciles. Frozen code/card/package were not repaired after launch.

## Local checks and resource accounting

Before launch, the fresh package passed 53 mapped-root tests with no skips, both actual generated-notebook normal/cached-module fixtures with all 16 cases each, fail-closed loader/malformed fixtures, and real notebook-only hierarchy SIGKILL/reaping. Four additional approval-gate tests passed after independent review found and resolved a missing affirmative-status check. These remain local evidence, not target GPU qualification.

The first durable exact-version terminal receipt was recorded 59.378 seconds after the launch-attempt timestamp. Assessor elapsed time was 0.102 seconds. GPU quota displayed 1.38 hours used / 28.62 hours remaining both before and after: displayed delta **0.00 hours**, not proof of zero billing. The 0.5 GPU-hour conservative bound showed no displayed overrun. Provider timeout 300 seconds was requested; no independent server timeout enforcement or exact-cancellation attestation is claimed. Final readback remains `ERROR`, no local owned launcher/monitor/collector process remains, and no active task-owned remote job is observed.

## Audit and next action

From the repository root:

```bash
python3 -B analysis/kaggle-containment-qualification-20260913T194449Z/audit_actual.py
```

Expected exit **2**, with `qualified: false`, exact returned version provenance and 0 executed / 16 unexercised cases. This is a successful audit of a negative qualification, not a successful qualification.

Key artifacts: `AUDIT.json`, `RESULT.json`, `SERVER.json`, `SERVER-RECEIPT.json`, `server-notebook.ipynb`, `launch.json`, `status-exact-version-final.json`, `RETRIEVAL.json`, `remote/qualification/{result,platform}.json`, `remote/package-{worker,lifecycle}.json`, `verification/VERIFICATION.json`, and `OWNED-PROCESSES.json`.

Stop here. The sole authorization is consumed. Investigating why provider shape metadata disagrees with actual hardware does not authorize another launch or weakening containment. No construction/inference result is established.
