# Operational Theory T1R3: local repair candidate, no launch

T2R2 held frozen T1R2 for undetected host backing-evidence corruption.
PREDECESSOR.json binds its card/manifest/review and full earlier chain.
`evidence_authority.py` pins all evidence bytes/digests/file identities and catalog
before worker startup. Every broker read verifies the entire set and returns the
verified buffer. `evidence_api.py` routes native state/grid, observations,
transitions, decision packets and the background reader through this broker.
`supervisor.py` checks before/after broker operations and after model responses;
`theory.py` checks again immediately before reconciliation and canonical publication.
A private empty read-only /evidence tmpfs prevents raw process reads from bypassing
the broker. Host data is never directly mounted into the worker. All 120 existing
host-operation slots include integrity read operations; other caps are unchanged.

The scientific question, recorded CD82 prefix, caps and interpretation remain
those in EXPERIMENT-CARD.md. This identity requires independent T2R3 review and
separate exact-identity T3 user authorization before any provider call.

Preserved T1R1 closures:
- `helper_authority.py`, `actor.py`, `supervisor.py`: host owns active treatment
  helpers. Read-only mount prevents direct writes; broker refuses replacement of
  a canonically bound name. New bytes need a new name and canonical revision.
  Checks run before broker operations, including decision preparation. Original
  helper AST rules remain. Registration must match active bytes.
- `evidence_api.py`, `worker.py`: both arms' unchanged getters and state-tail
  producer use verified broker reads. Host publishes only admitted observations.
  `/work/ipc` copies are non-authoritative scratch; their bytes never define refs.
- `prefill.py`, `client.py`, `transport.py`: allocate only owned NOOA prefill IDs
  at the exact pinned AST assignment before event rendering. No recursive text
  replacement. Provider IDs stay opaque; duplicates stop before tool execution.
  The allocator skips provider-owned future local names.

`test_adversarial.py` runs T2's helper code unchanged and transient evidence probe
through real CodeAct, tests both evidence arms and three provider ID collisions,
checks ordinary text, exact local linkage, duplicate rejection and allocator
collisions. `reviewer-fixtures.json` preserves the original attacks and findings.
The original opaque-provider test now returns a distinct ID on its second response;
its original opacity assertions remain, with true duplicate rejection tested
separately. The original 25 deterministic tests remain in `test_operational.py`.
Seven added cases cover four proposed-binding mistakes, bounded exhaustion and
two accepted-state corruption cases. `binding-review-fixture.json` preserves T2R-F1.

Mechanical authority does not certify truth, helper semantics, or why the Actor
chose an action. Arbitrary Python can calculate arbitrary values; accepted theory
refs still resolve only to the host catalog. No gameplay or treatment efficacy is
established by local fixtures.

From repository root, reproduce local qualification without credentials:

```bash
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/operational-theory-t1r3-20260917/verify.py --output /tmp/t1r3-review-unique
```

The runner uses `unshare --user --map-auto --map-root-user` for all pytest suites;
real Actor fixtures additionally use bubblewrap network/PID/mount isolation,
read-only source/evidence/helper mounts, dropped capabilities and read-only root.
It runs original plus adversarial tests, adjacent constructor transport and Actor
suites separately, 160 runnable current-runtime tests, Ruff and format, and the
standalone transcript-free replay CLI. Existing current-runtime collection defects
in test_assessment.py/test_packaging.py (missing prepare_candidate) remain excluded.
Raw requests, responses, receipts, canonical events and worker trees are retained
in one compressed archive in the external qualification directory; temporary trees
are removed after archiving. CHECKS.json records commands, source hashes and logs.

Freeze only after all checks succeed. Identity uses an explicit root file allowlist
plus pinned evidence, metadata, incumbent chassis and installed NOOA sources.
Mutable review/authorization/results, live outputs, caches and generated fixture
trees are excluded. A reviewer adding T2R3-REVIEW.json must not change manifest hash.

T2R3 handoff: independently verify identity.py, PREDECESSOR.json and qualification
CHECKS.json; reproduce attacks and inspect archived execution output (NOOA renders
PythonOutput in user events). Check helper bytes against canonical refs and confirm
that tampered scratch never appears through evidence APIs. Inspect parity and
transport receipts. Write your own PASS/HOLD artifact binding manifest_sha256 and
task `operational-theory-T1R3`, role `independent-reviewer`. Do not reuse T1's review
as approval. No provider calls or T3 authorization are part of this handoff.

`test_evidence_integrity.py` adds real Actor host fault injections for pixel/value
changes, replacement, truncation, symlink and same-byte relinking, catalog file and
live catalog mutation, startup/read boundaries, and reconciliation/publication.
The reviewer fixture is preserved in evidence-review-fixture.json. Each hard-stop
case checks clean reaping, no completed RESULT and no invalid canonical version.
Read-then-drift checks include observation, transition and decision-packet APIs.
Untampered completion, request parity, Actor write denial, proposal repair, accepted
helper integrity, ID ownership, isolation, accounting and deadlines remain in the
full inherited suite. All evidence checks are identity-only, never semantic review.
