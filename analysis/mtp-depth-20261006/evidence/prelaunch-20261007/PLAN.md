# Finish the authorized MTP sweep freeze

Scope: analysis/mtp-depth-20261006 only. Parent thread c07540ef-efa9-489a-9f6b-13cac5bad5d3; Claude session f6be8498-ddf9-43b1-bc28-93aeda775ac7.

1. Preserve the inherited rehearsal, then enforce M3A/M3C/M3B and the approved order.
2. Verify the revised analyzer with null, planted gain/loss, missing baseline and monotonic drift fixtures. Exercise the actual generated notebook with a scaled mock.
3. Commit code/card/workload/audit/config/test evidence without raw dataset logs. Build clean provenance from that code commit.
4. Upload one private dataset version, retrieve every shipped file at that version, compare hashes and bind the notebook to its manifest.
5. Commit generated notebook/manifests/readback, tag the prelaunch commit, then restore the single-attempt watcher. Verify the real service and launch receipt.

Pre-mortem: the sweep spends GPU time but cannot support a decision.
- Missing M3C is silently dropped and a candidate is promoted against two anchors. Observed in analyzer and cut logic. Prevent by requiring all three, flagging an incomplete sweep as unresolved, and protecting baseline C in the budget planner.
- Revised code is local but the server receives the earlier dataset. Observed stale readback/provenance. Prevent with version-specific download, matching local/uploaded hashes, and a notebook-bound expected manifest hash.
- The watcher expires or hangs and reports no useful state. Observed 401 and unbounded SDK requests. Refresh near expiry, use bounded network calls, preserve polling errors, and refuse launch on unknown slots/quota.
- A resumed launcher repeats a push. Preserve the existing no-retry policy with an exclusive launch-attempt claim.
- Mock profiling cannot establish CUDA behavior or an ARC score gain. Accepted: local tests qualify plumbing only; the real sweep remains the necessary measurement.

No change to alpha, model, server flags, score claims, or authorized compute cap. No GPU retry.
