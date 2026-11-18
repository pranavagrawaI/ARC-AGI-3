# Operational Theory T1R1: local repair candidate, no launch

T2 held frozen T1 for helper authority, transient evidence mutation and ID
corruption. PREDECESSOR.json binds the untouched predecessor and review.
The scientific question, recorded CD82 prefix, caps and interpretation remain
those in EXPERIMENT-CARD.md. This identity requires independent T2R review and
separate exact-identity T3 user authorization before any provider call.

Repairs:
- `helper_authority.py`, `actor.py`, `supervisor.py`: host owns active treatment
  helpers. Read-only mount prevents direct writes; broker refuses replacement of
  a canonically bound name. New bytes need a new name and canonical revision.
  Checks run before broker operations, including decision preparation. Original
  helper AST rules remain. Registration must match active bytes.
- `evidence_api.py`, `worker.py`: both arms' unchanged getters and state-tail
  producer read host-owned `/evidence`. Host publishes only admitted observations.
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

Mechanical authority does not certify truth, helper semantics, or why the Actor
chose an action. Arbitrary Python can calculate arbitrary values; accepted theory
refs still resolve only to the host catalog. No gameplay or treatment efficacy is
established by local fixtures.

From repository root, reproduce local qualification without credentials:

```bash
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/operational-theory-t1r1-20260917/verify.py --output /tmp/t1r1-review-unique
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
trees are excluded. A reviewer adding T2R-REVIEW.json must not change manifest hash.

T2R handoff: independently verify identity.py, PREDECESSOR.json and qualification
CHECKS.json; reproduce attacks and inspect archived execution output (NOOA renders
PythonOutput in user events). Check helper bytes against canonical refs and confirm
that tampered scratch never appears through evidence APIs. Inspect parity and
transport receipts. Write your own PASS/HOLD artifact binding manifest_sha256 and
task `operational-theory-T1R1`, role `independent-reviewer`. Do not reuse T1's review
as approval. No provider calls or T3 authorization are part of this handoff.
