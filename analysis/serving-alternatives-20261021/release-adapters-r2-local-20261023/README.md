# Release adapter descendant-contract correction

Status: **LOCAL CPU CORRECTION; NOT RUNTIME ENABLED OR GPU QUALIFIED**.
This separate directory preserves the submitted adapter snapshot and its failed
review. `PRIOR-SNAPSHOT.json` binds the original sources, local proof, coordinator
review and reviewer leak receipt. Trials1–10 and both shutdown prototypes remain
unchanged. No GPU query, model load, external network, cache/threshold change,
experiment freeze, slot/launch, staging or commit is authorized here.

## Demonstrated defect and narrow correction

The old helper's PR_SET_PDEATHSIG applied to its direct query exec only. It did
**not cover any grandchildren**, ordinary same-PGID/session children included.
The review receipt records a surviving ordinary grandchild after query timeout;
`cleanup_error:null` described direct-child cleanup, not descendant containment.
That original receipt and all submitted bytes remain preserved.

This revision chooses an enforceable **no-new-process contract** rather than a
process-tree cleanup framework. `no_processes.py` installs a Linux x86-64 seccomp
filter before query exec, with no_new_privs. It denies fork, vfork and clone
without CLONE_THREAD. clone3 returns ENOSYS so libc can fall back to inspectable
clone for real threads. Unknown architecture/x32 calls fail closed. The filter
persists over exec and cannot be removed. Threads share the query's thread group;
query SIGKILL terminates them. PR_SET_PDEATHSIG still handles probe-worker death.
No engine bytes or measurement behavior change.

The helper writes a small exclusive attestation before exec. Capture checks the
contract, filter SHA, architecture, no_new_privs/seccomp state, PID and starttime
against its observed owned query before returning success. Raw stdout/stderr and
containment evidence are exact hash references. Partial command evidence remains
partial; a valid containment receipt is independent of command success. Cleanup
uses an owned pidfd, never a speculative process group or unrelated process.
The same command end is used; no new grace or deadline extension is added.
Kernel syscall/scheduling completion remains cooperative, not a hard time bound.

Actual CPU tests exercise ordinary subprocess fork attempts on timeout and killed
probe worker; both receive EPERM and create no grandchild. The direct query is
reaped/observed gone. Normal threaded completion and existing large-output/error
behavior pass. Direct fork/clone process syscalls are denied; clone3 reports
ENOSYS. Unknown/drifted attestation identity and contract fields are rejected.
The original prepared registration, unknown frontend identity, absolute release
core, occupied/free loopback plumbing and unchanged resource predicates remain.

Actual nvidia-smi compatibility with this filter is **unproved**. It may fail if
the executable/library needs a subprocess; that must stop a future reviewed run,
not disable enforcement. The trusted helper/interpreter/source/dependency and
execution environment must be pinned in runtime packaging. This is a constrained
query-command contract, not a sandbox for arbitrary hostile code.

## Runtime scope remains explicitly blocked

`Adapters` now refuses any non-LOCAL_CPU_ONLY policy. Current cgroup fixtures and
native fixture/historical logs carry explicit input-scope labels. They prove CPU
plumbing only. Packaging must separately bind the actual frontend's registered
cgroup and observer PID/cgroup/mount namespace to the derived memory.max/current/
stat paths (including the cgroup2 mount root), with scope/path drift rejected.
A fresh run-output identity must own the native log before frontend startup;
run marker, path and inode/creation binding must reject stale, copied or swapped
logs. A matching text marker alone cannot establish freshness. These runtime
bindings are not implemented or proved by saved fixtures, and cannot be bypassed
by choosing the source-only GPU-query policy label.

Original GPU <=2048MiB, host >=140GiB and disk >=100GiB predicates remain exact.
No active_file credit or cache intervention. The host140GiB rationale/eligibility
is independently unresolved; this correction cannot establish it. Local disk
can still fail the original100GiB gate. No real GPU release is inferred.
The earlier one-load/health200/release proposal remains only a proposal:
3900s platform /3600s cooperative /300s cleanup, no replay or candidate. No GPU
packet is constructed or frozen here.

## Source-bound local qualification

Canonical proof: `qualified-r3/PROOF.json`. Run with a new output path:

```bash
timeout 600s python3 verify.py --output independent-review
```

The inherited guard permits CPU Python fixtures and IPv4 loopback binds only;
connect/DNS, GPU device/runtime imports and external executables are refused.
Verify checks prior adapter/docs/reference preservation, accepted core and all908
trial10 bindings before/after, current source hashes before/after, loaded local
module paths, every case inventory byte and referenced raw artifact. The filter
program/interpreter hashes are included. The accepted local-core absolute import
remains unchanged; this directory is not a remote-executable GPU bundle.

`qualified-r1` preserves16PASS/1ERROR from a new test reading /proc/exe while its
query was exiting. No production identity rule was relaxed. The fixture now
retries incomplete reads without counting them as disappearance and holds an
already-open pidfd for failure cleanup. `qualified-r2` passed all17tests before
verification-only preservation/import checks were added. Sources remained exact
during the final qualification. Both failed and superseded evidence remain.
