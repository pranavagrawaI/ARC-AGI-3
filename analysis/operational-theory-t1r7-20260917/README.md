# Operational Theory T1R7: C6 predecessor-chain repair, no launch

C6 (equivalent gate T6R2) held frozen T1R6 for a single blocker, C6-F1: its frozen
PREDECESSOR.json was not a complete, mechanically resolvable predecessor chain — the
top node claimed files that live at a different identity, orphaned T1R5's own
predecessor/review records, and skipped the T1R3 and T1R4 nodes. Every launch-critical
hash still matched, so this is a provenance-record defect, not a machinery defect: C6
independently verified all three T1R6 machinery repairs (stale globals, verifier,
`nemo`) and every verifier group. T1R7 preserves T1R6 byte-for-byte in all
Actor/runtime/workflow/theory/evidence/helper/transport code and fixes only the
predecessor chain and fresh-identity bindings. `PREDECESSOR.json` is now a fully nested
chain (t1r6->t1r5->t1r4->t1r3->t1r2->t1r1->t1); `chain.py` validates it deterministically
and `test_chain.py` proves the malformed T1R6 chain fails while T1R7 passes. No provider
or Kaggle calls occur here; independent C6R1 review is required before any replacement
release.

The Actor-facing cell receives an `ActorFacade` capability surface. The raw Agent
object and raw bound methods never enter model globals. Facade calls and returned
containers are sanitized into opaque wrappers, while explicit AST policy rejects
reflection, type/MRO/descriptors, callable-global, pickling, dynamic private-name,
runtime-module and future-import traversal before execution. Public solver tools
remain available through approved facade methods.

Model-defined functions are reconstructed after each cell against a single
session-owned globals mapping, and that same mapping is the namespace the cell
executes in. The mapping is never replaced by a filtered copy while a cell runs,
including after the blocked-module strip, so rebinding `_value` to 2 changes what a
retained `_read()` observes to 2. The mapping contains only model-owned values and
the facade, and is created per CodeAct session; arms do not share it.

T4 invalidated T3 because NOOA dropped `_json` between cells despite promising
persistent Python bindings. The exact six preserved tool-call code strings fail
under the unmodified Actor boundary and succeed at the missing schema read under
this candidate. That fixed replay still makes no action proposal. This is a
machinery qualification, not evidence of successful construction or efficacy.

`cell_state.py` brackets actual CodeAct execution with task-local binding ownership.
Python's symbol table identifies user assignments/imports/functions. The vendored
`actor_execute.py.txt` changes the pinned execution function at the capture boundary:
model-owned names are declared/captured regardless of prefix or callable type,
private host module globals are withheld, and a comprehension avoids a collision
with the old `_gvar` capture-loop temporary. Only these user bindings leave the
wrapper; no host namespace is serialized or inserted into session locals.
`runtime-boundary.json` pins the exact installed method/module. Installation compiles
the frozen vendored function with the original trusted runtime globals; it writes
no installed dependency. ContextVar reset separates nested executions. Host
prefill is recognized by allocator-owned IDs, never a provider-controlled prefix.

The same public contract and adapter run in both arms. Actual call parameters are
tracked separately from public convenience imports, so `import json` and local
module rebinding remain permitted without replacing host module objects. Reserved host inputs and
runtime names, private attributes/reflection and runtime imports receive local
CELL_STATE_REPAIR before execution. Missing free underscore names are rejected
before side effects. Other runtime NameErrors preserve already executed effects
and return explicit bounded feedback. No new model requests or retries are made;
all repair opportunities use the existing six-call allowance. Python AST policy
is a contract guard, not a proof of containment against arbitrary Python. The
unchanged OS namespaces and broker authorities remain the containment boundary.

Qualification exposed a pre-existing producer shutdown race: after done, the
broker stops reading while a background reader can still need an RPC. The
worker now cancels/joins registered jobs before done. A forced cancellation-time
evidence read fails before and exits cleanly after; deadline/reaping caps are
unchanged. Raw failed qualification and reproduction remain preserved.

`test_cell_state.py` exercises the real Actor/CodeAct path for exact T3 replay,
underscore and ordinary bindings, callable/module aliases, private-name attacks,
provider prefill impersonation, deletion, missing-name repair and session/arm
isolation. `test_release.py` checks fresh authority and one claim across output
paths. `run_pair.py` requires independent T6 and new user release, writes external
outputs only, and cannot reuse T3 authorization. `chain.py`/`test_chain.py` verify the
predecessor chain resolves end to end and that the malformed frozen T1R6 chain fails.
See EXPERIMENT-CARD.md for the
unchanged comparison, sealed prefix, numeric caps and exact path convention.

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

From the repository root, run offline qualification with no credentials:

```bash
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/operational-theory-t1r7-20260917/verify.py --output /tmp/t1r7-review-unique
```

The runner executes all 290 prior tests plus new regressions, Ruff and format,
standalone canonical replay, and source stability. Known baseline collection
exclusions remain test_assessment.py and test_packaging.py (missing
prepare_candidate). All raw fixtures are archived outside the source identity.
The actual Actor fixtures require unshare/bubblewrap and loopback only. They
carry no credentials, make no provider calls, and do not execute game actions.

Freeze follows successful local qualification. MANIFEST.json covers the explicit
source allowlist, vendored boundary, exact T3 code fixture, immutable evidence,
metadata, current chassis and installed NOOA donors. Qualification outputs,
review, release and paid artifacts are excluded. PREDECESSOR.json is a fully nested,
mechanically resolvable chain binding the T1R6 manifest/card and C6 review, the T1R5
manifest/card and T6R1 review, the T1R4 and T1R3 nodes (T1R3 carrying the T3 launch
manifest and invalid T4 audit at their real locations), and every earlier frozen
predecessor through terminal T1; chain.py and test_chain.py enforce it.
Independent C6R1 must review this identity; T5 does not grant paid launch authority.
