# T1R2 qualification evidence

No provider or Kaggle calls. All request dispatches use local loopback fixtures;
real Actor cells execute under the required unshare/bubblewrap isolation.

`before.txt`: three real Actor cases reproduced the original IntegrityError before
repair. `focused.txt`: recovery and hard-stop probes after repair; two assertions
initially confused a loader's helper listing with invocation. The final tests use
explicit execution markers and distinguish listings from calls.
`development-fixtures.tar.gz` preserves both development raw fixture trees.

`qualified/CHECKS.json` records final commands, source hashes, exit codes and log
hashes. `qualified/raw-fixtures.tar.gz` retains full raw pytest fixture trees,
including requests/responses, canonical lineage, rejection events and supervision.
Extract into a disposable directory; paths are under `fixtures/`. The temporary
paths recorded in commands have been removed after archiving.

The 41 package cases include all 34 predecessor checks plus seven held-boundary
cases: missing/unregistered/wrong-hash/stale proposed binding recovery, bounded
repair exhaustion, accepted helper replacement, and accepted helper removal.
Recovery takes three fixture model calls, retains reconciled phase on rejection,
accepts only valid versions 1–5, and does not invoke an invalid helper callable.
Accepted-state corruption still raises IntegrityError and reaps without RESULT.
The adjacent suites remain 56 transport, 18 Actor and 160 runnable runtime tests;
the same two pre-existing runtime collection defects are excluded as documented.

`PREDECESSORS-BEFORE.json` inventories every file in frozen T1 and T1R1, including
reviewer evidence. `COMPLETION.json` records the final preservation comparison,
qualification binding, card/manifest hashes and post-freeze identity verification.
`SUCCESSOR-CHANGES.json` records changed/new source files before freeze.

Independent T2R2 must issue its own manifest-bound PASS/HOLD. Qualification is
local mechanical evidence, not gameplay, truth or treatment efficacy evidence.
No authorization or review has been self-authored.
