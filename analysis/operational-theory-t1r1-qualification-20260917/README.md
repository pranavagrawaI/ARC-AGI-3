# T1R1 local qualification evidence

Provider calls: 0. All transport dispatches were local loopback fixtures. Real
Actor cells ran inside bubblewrap beneath the required unshare wrapper.

- `T1-BEFORE.json` inventories all 5,092 predecessor files, including reviewer
  evidence and generated predecessor artifacts. `T1-PRESERVATION.json` checks the
  entire inventory, beyond only the predecessor manifest.
- `development-fixtures.tar.gz` preserves the original 25-test pass, initial
  adversarial fixture failures (wrong output event selection and a prohibited
  pathlib import), and corrected eight-test pass, with raw files and logs.
- `final/` and `settled/` are superseded development qualification attempts.
  Source refinements occurred while their adjacent suites were still running.
  Their source-stability gate correctly rejects them; neither authorizes freeze.
  Raw test logs and fixture archives remain available. They are not approval.
- `qualified/CHECKS.json` is the final settled-source qualification record.
  It contains exact commands, exit codes, log hashes, source inventory and raw
  archive hash. `qualified-run.txt` contains runner output.
- `qualified/raw-fixtures.tar.gz` contains complete raw pytest fixture trees.
  Extract to a disposable directory to inspect requests, responses, receipts,
  PythonOutput events, canonical lineage, helper archives and supervision.
  Fixture paths embedded in commands/logs refer to now-removed temporary trees;
  the archive retains their relative paths under `fixtures/`.
- `qualified/replay.json` is standalone transcript-free replay of the unchanged
  reviewer helper attack. This is mechanical local evidence, not gameplay truth.
- `COMPLETION.json` binds the qualified source inventory to the final manifest
  and records freeze, identity verification and predecessor preservation.

No review or authorization has been self-authored. Independent T2R must inspect
and reproduce the boundary repairs, then issue its own exact-identity PASS/HOLD.
