# SUPERSET_WORKER_BLOCKED — R10A-PUSH acceptance

One frozen-gate push executed. Provider receipt returned version 1 at https://www.kaggle.com/code/pranavagrawai/r10a-cd82-constructor-repair, not authorized pranavagrawai/arc3-r10a-cd82-20260916. Gate exit 2: push URL target/version mismatch. CONSUMED retained; launch.json absent. No retries, rename, patch, cancellation or submission.

Exact gate command/outputs: gate-push-receipt.json in this directory. Provider receipt: ../r10a-cd82-20260916/push-56e324f3e2004081979fd2f2a9c40e4f.json. Provider admission is reported, not certified against the authorized target.

First exact ApiGetKernelRequest.version_label=1 and exact status calls returned 404 for returned target, 403 for intended target. Evidence: returned-readback.json, returned-status.json, intended-readback.json, intended-status.json. Read-only account listing target-discovery.json shows the returned target. Privacy, attachments, notebook and runtime status unresolved at this stage. Recovery evidence stays separate from production control state.

Mechanical/scientific acceptance HOLD. Continue explicit-version read-only recovery/monitoring and exact allowlisted collection. No R10b run. This document is a worker status, not an approval or replacement launch record.
