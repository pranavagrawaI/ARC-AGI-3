# Controller-owned release queries and runtime observation scope

Status: **SOURCE-BOUND LOCAL CPU INTEGRATION; REAL GPU RELEASE UNPROVED**.
Canonical proof: `integrated-qualified-r8/PROOF.json`:60tests PASS14.012s,
436 raw artifact references checked. This includes33 adapter/ownership/scope
checks and all27 unchanged accepted shutdown/resource/IPC regressions executed
against the narrow controller preparation seam. Original sources and all908
trial10 inputs remain byte-exact. No GPU, external network, cache/threshold
change, experiment freeze, slot/launch, staging or commit occurred.

## Failed premise and structural intervention

R1/r2 relied on short-lived probe-worker ownership plus PR_SET_PDEATHSIG to clean
query execs. First, ordinary same-PGID/session grandchildren survived direct-child
cleanup. R2 enforced fork prohibition, but a permitted nonleader thread exec of
the same interpreter cleared PDEATHSIG without changing PID/starttime/NNP/seccomp.
The reviewer killed the worker, observed its query survive, then cleaned it by
retained pidfd. These are preserved witnesses, not claims about nvidia-smi.

The structural correction removes query execution from the receipt worker.
`controller_shutdown.py` is the exact accepted core plus a preparation seam
before worker.start and tagged preparation-failure receipt preservation. Clean
query-phase expiry still enters the original owned fallback; uncertain query
cleanup remains fatal and its complete command receipt reaches the release result. Adapters execute the fixed GPU/apps query captures
synchronously in the long-lived release controller using the original probe end.
Only after both queries are verified cleaned can the receipt worker start; it
parses cached bounded raw references. It cannot reuse controller query authority.
No query is live during the actual receipt-worker lifecycle. Other probe ordering,
absolute release deadlines, native frontend signaling and resource gates remain.

`capture.py` binds controller identity and starts a trusted gated helper. The
helper installs the unchanged seccomp fork prohibition, publishes its attestation
and waits. The controller opens/retains the owned query pidfd, checks exact query
parent/session/PID/starttime and attestation, then authorizes exec through the gate.
A captured CPU startup witness showed an initially empty /proc cmdline before
helper readiness; every other identity field was unchanged. Only at that gated
initial phase may cmdline be empty. Ready identity must contain the exact expected
helper argv hash with every other identity field unchanged. Empty/wrong ready
argv and unexpected nonempty initial argv still fail. Native/frontend identity
semantics are not relaxed. gate-audit-2/115 preserves both snapshots; deterministic
initial-empty and negative-ready regressions cover the boundary.
The controller retains that pidfd across exec, cancellation, command errors and
timeout. PDEATHSIG is auxiliary; it is no longer cleanup authority. No numeric
PID/PGID signal fallback is used.

A fixed100ms cleanup reserve is taken from the existing absolute query end. A
query is not admitted with only the reserve left. Data work stops at work_end;
owned pidfd cleanup and reaping use the same end. Every receipt separates clocks
for pidfd binding, execution authorization, work end, cleanup and finalization.
Scheduling/kernel syscall completion is cooperative; no hard syscall bound is
claimed. Late completion or uncertain cleanup cannot pass. A live/unreaped query
retains its Popen/pidfd in the controller's RETAINED ledger and blocks dispatch.
No clean receipt is synthesized. The owning controller must remain alive; abrupt
controller/process/host loss is outside this local guarantee and needs platform
containment in any later runtime packet.

Tests reproduce actual allowed thread exec/PDEATHSIG0 and death-triggered
cancellation; retained pidfd cleanup succeeds with unchanged PID/starttime.
Actual receipt-worker SIGKILL occurs only after prepared queries are reaped.
Ordinary fork/process-clone denial, clone3 ENOSYS/thread fallback, normal thread
completion, pre-exec cancellation, owner/attestation identity errors, raw partial
bytes and no-false-clean failures are covered. A deliberately suppressed kill
retains the live handle and blocks new dispatch. Test-only rescue is recorded
separately and never changes the original failed deadline receipt. Fatal controller-query errors retain command
receipts in controller_query_failure, including unresolved-handle tokens; a
second query/receipt worker is not dispatched after uncertain cleanup.

## Required runtime boundaries, exercised locally

`runtime_scope.py` is release-specific, not a process framework. It binds while
the registered frontend is alive and requires it to share the intended observer's
cgroup and PID/mount/cgroup namespaces. The cgroup2 hierarchy and exact applicable
mount root derive memory.max/current/stat paths; meminfo is /proc/meminfo and disk
is /tmp. Saved fixture paths cannot stand in for those live inputs. Ambiguous,
outside-root, parent traversal, scope/mount/path/file identity drift fail closed.
The controller or its direct receipt worker validates this scope before/after
snapshots. Reads remain sequential, not an atomic kernel snapshot.

A new exclusive run directory owns an exclusive initially empty native.stderr,
a nonce/run/controller marker, and directory/log inode identities. The runner
opens that log before wrapper Popen and already retains Popen ownership before
fallible record_wrapper publication. Registration carries the prepared run_id.
Binding verifies the wrapper identity and the **live native frontend's actual
fd2 inode/path** against the fresh lease. Existing/copied logs, prefilled logs,
wrong run IDs, reused output directories, stale paths and inode swaps are rejected.
The CPU integration exercises real inherited stderr and a real /proc cgroup/mount
scope in the receipt worker. The scope object is a trusted runner-produced input,
not permission to fabricate lifecycle receipts or copy historical log contents.

Adapters require these bindings for the source-only future GPU-query policy.
Local qualification still forbids that policy's execution. Actual Kaggle mount/
namespace layout, shipping runner wiring, nvidia-smi compatibility under unchanged
seccomp, model preparation and native GPU release remain unproved. Unsupported
runtime layout or command behavior must fail; no enforcement bypass is authorized.
Native text does not directly expose watchdog stop or GPU context destruction.

Original <=2048MiB GPU, >=140GiB host and >=100GiB disk predicates remain exact.
There is no active_file credit/cache intervention. Host140GiB provenance and
actual eligibility remain independent and unresolved. The actual local /tmp is
below the original disk threshold, correctly preventing a release pass. CPU
fixture resource success does not qualify GPU release, inter-arm release or
capacity. Trial10 remains paired FEASIBILITY_ONLY with ratio null.

## Preservation and reproduction

`PRIOR-SNAPSHOT.json` preserves R1; `PRIOR-R2-SNAPSHOT.json` preserves R2 and the
reviewer's thread-exec leak receipt. `OWNERSHIP-SNAPSHOT.json` preserves the
separate21-test ownership proof before this scope integration. Prior copies and
failed evidence are unchanged. Local failed runs also remain:

- ownership-first.log: old test assumed all near-cutoff command artifacts were
  complete; partial evidence is now hashed and can never satisfy a success gate.
- scope-qualified-first: a test chose the last near-deadline sample rather than
  a completed positive resource witness; original eligibility gates unchanged.
- integrated-qualified-r7: safe helper pre-exec identity refusal, investigated
  separately with captured empty-initial-cmdline evidence; query remained gated
  and was reaped.
- scope-qualified-r3: unresolved query ledger correctly blocked later test
  scenarios. Cleanup reserve is now fixed instead of shrinking for late admission;
  separate scenarios explicitly rescue retained test queries without qualifying
  their failed original receipts. No production auto-clear of unresolved state.

Verify with a fresh output directory:

```bash
timeout 600s python3 verify.py --output independent-review
```

Verify binds current sources before/after, actual loaded local module paths,
original adapter/core/review/reference inputs, all908 trial10 files and exact
case inventories/raw references. The inherited audit guard permits only CPU
Python fixtures/IPv4 loopback binds, and denies network connects/DNS/GPU devices/
native runtime imports/external executable launches. It is a guard for trusted
fixtures, not a hostile-code sandbox. The unchanged seccomp filter hash and Python
interpreter hash are recorded. The accepted local-core import remains local;
this is not a remotely executable experiment packet. See LOCAL-HANDOFF.md.
