# Corrected MTP retry plan

User authority: "do a retry", 2026-10-07, following confirmation that RTX PRO 6000 requires ARC-AGI-3 attachment. One new run, private kernel mtp-depth-sweep-retry1-20261007. Preserve original v1 card, code, receipts and dataset. Reuse verified replay dataset v2; modify only notebook identity, competition attachment, early runtime hardware gate and retry provenance. Keep all server flags, alpha, measurement and decision rules fixed.

## Pre-mortem

- Wrong GPU despite metadata: observed in v1. Add first executable cell that saves actual nvidia-smi inventory and aborts unless exactly one RTX PRO 6000 with >=90000 MiB. Test actual cell with T4, dual GPU, missing query and valid RTX fixtures. Remote metadata alone is insufficient; inspect runtime hardware after launch.
- Competition source omitted: observed builder defect. Require exact competition slug arc-prize-2026-arc-agi-3 in generated metadata and prelaunch verification; inspect saved remote metadata after acceptance.
- PLE startup still fails: unresolved lower-level cause. Keep incumbent flags fixed; preserve startup failures under original one-attempt rules, never claim GPU correction fixes PLE before a healthy launch.
- Replay or comparator accidentally changes: retain byte-identical dataset v2 and original sweep cell except adding retry provenance. Verify original tag and 37-file payload/readback before launch.
- Duplicate push/ambiguous response: reuse exclusive launch-attempt guard in separate evidence directory, one push, no automatic retry.
- Quota/slot/priority conflict: reuse fail-closed typed-quota, owned-kernel status and hash-bound wave-2 HOLD checks. 30h weekly cap, reserve 4.5h, fewer than two active sessions and no unknown status before push. No other run cancelled.
- Qualification ceiling: mocked hardware cell tests establish abort plumbing, not real accelerator allocation or PLE health. Real retry checks those conditions.

No paid diagnostic session precedes the authorized retry. M3A is its incumbent startup and workload qualification. Freeze and tag before push. Commit only retry and this thread's collected failure evidence.
