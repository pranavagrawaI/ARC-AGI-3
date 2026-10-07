# Authorized prelaunch freeze

T3 source thread c07540ef-efa9-489a-9f6b-13cac5bad5d3 / Claude f6be8498-ddf9-43b1-bc28-93aeda775ac7, completed at Pranav's request on 2026-10-07.

Shipped replay code commit: 28ef795f16f5165dc6c71426569bf7a16968d970. Final prelaunch tag: mtp-depth-sweep-prelaunch-20261007. The tag also freezes the completed watcher, generated notebook, uploaded manifest and verification evidence. It is created before the watcher is restored.

Order: M3A -> M1 -> M2 -> M3C -> M4 -> M3B, all unprofiled. PROFILE_M3 follows separately if nsys and time permit. All three anchors are required for any adoption verdict. Alpha remains 0.523. 25 minute-40 games, 10 client slots, unchanged incumbent flags except speculative depth.

Qualification: 26 contract tests pass. Generated notebook null rehearsal passes occupancy, produces KEEP M=3 and has zero request errors. Planted HTTP +10% M2 / -8% M4 produces CLEAN WIN / CLEAN LOSS. Failed M4 is recorded once and final baseline B still completes. Headers, admission, handovers, six-launch order and depth substitutions were checked in actual raw events; compressed events are retained here. The inherited unscaled rehearsal is retained separately, with its INVALID WORKLOAD verdict.

Dataset: private pranavagrawai/mtp-depth-replay-data-20261006 version 2, pinned in kernel metadata. Expected manifest c21852dc33715d0efe6b4d62b70d7869040b859b4f1c2ec36ad1ea22e64b9713. Final readback receipt is evidence/dataset-readback.json. Notebook setup rejects mismatched manifest, commit or dirty code before loading the GPU server.

Operations: systemd user unit mtp-depth-launch-watch.service. Restoring it is already authorized. One push, 4 h notebook deadline, 4.5 h platform timeout, no retry. Wave-2 HOLD is checked against its raw readout hash. Existing occupied/unknown slots and incomplete quota block launch. Parsed timedelta usage avoids the SDK's malformed/lossy string serialization. The exact redacted notebook sentinel is ignored; other unknown identities block launch.

Runtime proof after tag/restore lives in ignored evidence/live-prelaunch-verification.json, evidence/launch-identity.json, evidence/quota-before.json, evidence/launch-attempt.json, evidence/launch-response.json and evidence/launch-watch.log. Those receipts reference the immutable tag commit. A missing or unresolved receipt is not successful execution and does not authorize another push.

Local qualification supports apparatus correctness only. GPU throughput, nsys/CUDA behavior, early EOS stability and live ARC behavior remain unmeasured until this experiment executes. No promotion or new scored run is authorized here.
