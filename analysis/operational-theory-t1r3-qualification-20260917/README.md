# T1R3 local qualification evidence

Provider/Kaggle calls: zero. Transport fixtures use loopback; real Actor cells
execute under unshare and bubblewrap with private read-only evidence namespace.
No independent review or launch authorization is self-authored.

`smoke.txt` covers untampered real-Actor completion and request parity.
`attacks.txt` records the first twelve host-drift fault injections.
`development-fixtures.tar.gz` preserves their complete raw fixture trees.
`qualified/` is the superseded 55-case qualification attempt: the source changed
while adjacent checks were running to cover same-inode relinking. Its source
stability gate rejects it; it does not authorize freezing.

`qualified-final/CHECKS.json` is the settled-source qualification record, including
exact commands, source hashes, exit codes, logs and raw-archive digest. Its package
suite has 56 cases: all 41 predecessor cases and 15 host-evidence cases. Adjacent
suites retain 56 transport, 18 Actor and 160 runnable current-runtime cases (290
in total). The same pre-existing assessment/packaging collection defects remain
excluded. Ruff, format, standalone replay and source-stability checks are included.

`qualified-final/raw-fixtures.tar.gz` retains complete raw fixture trees under
`fixtures/`: model requests/responses/receipts, host events, evidence snapshots,
canonical journals and worker reaping records. It preserves adversarial symlink
fixtures as links; regular archive members can be inspected without extraction.
Temporary fixture paths recorded in commands are removed after archiving.

The new tests cover states/pixels, transitions, catalog bytes, catalog object,
EVIDENCE.json, truncation, replacement, symlink, same-byte new inode and same-inode
hardlink restoration. Faults are injected before the first worker read, at a broker
read, after an Actor read, before reconciliation and at canonical publication.
They require hard failure, reaping, no completed RESULT, no canonical version and
no executed corruption sentinel. Existing tests cover clean completion, write
restrictions, background reader, helper proposal repair, accepted-helper hard stop,
ID ownership, accounting, isolation, deadlines and replay.

`PREDECESSORS-BEFORE.json` inventories T1, T1R1 and T1R2 including reviewer files.
`SUCCESSOR-CHANGES.json` records changed/new source hashes before freezing.
`COMPLETION.json` records final qualification bindings, all predecessor preservation,
manifest/card hashes and post-freeze verification. Independent T2R3 should verify
these and reproduce the fault injections before issuing its own PASS/HOLD.
These are mechanical local checks, not gameplay or treatment efficacy evidence.
