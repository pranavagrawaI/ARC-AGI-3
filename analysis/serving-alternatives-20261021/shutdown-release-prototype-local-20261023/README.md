# Local frontend-first shutdown/release prototype

Status: **LOCAL CPU/SOURCE-PATH PROTOTYPE ONLY**. No successor experiment freeze,
slot request, launcher, GPU query, network request, model load, measurement change,
cache operation or commit. Trials1–10 and the accepted diagnosis are untouched.

## Concrete delta

`shutdown.py` is an unintegrated lifecycle helper. `bind` requires a directly owned
wrapper in its new session, an exact-command-line native frontend directly below
that wrapper, same executable identity, and an explicit nonempty source-hash set.
It records PID/starttime, PGID/SID, UID, executable metadata, command-line hash,
boot ID, PID namespace and cgroup. Every initially registered member must have an
ancestry chain to the wrapper. Unknown later group members fail closed.

`release` verifies those bindings and sends SIGTERM **only to the native frontend**
using a verified pidfd. It observes native/owned-process/resource evidence within
a graceful sub-budget, then uses identity-verified pidfds for fallback SIGKILL to
registered owned processes. It never uses group-wide numeric signaling. Handles
are bound and the target set is checked before signals; identity/source drift,
missing visibility or unknown members stop signaling. Primary and secondary
cleanup errors remain separate. PIDfds address numeric PID reuse; executable and
source validation are cooperative checks, not an atomic exec lock or visibility
into another PID namespace. Incomplete live identities remain failures.

One caller-provided absolute end governs graceful waiting, fallback and the bounded
release series. Graceful expiry uses the existing cleanup reserve; it does not
create a fresh timeout. Signals are refused when verification crosses the end.
Blocking probe callbacks run in a CPU child with a bounded collection budget and
owned child cleanup. Kernel syscall stalls are not a proven hard subsecond bound;
every qualification invocation additionally has an external600s cap and alarm.

Every series sample stores separate start/end clocks and receipts for processes,
GPU memory/identity, compute apps, cgroup headroom, port, disk and native evidence.
An early failed/malformed GPU result does not conceal later observed predicates.
Unobserved fields are explicitly missing/ineligible. `resources.py` is pure saved-
input classification; the GPU<=2048MiB, host>=140GiB and disk>=100GiB predicates are
unchanged. Device UUID matching is additionally tied to the supplied identity.
No active_file credit or cache clearing is added. A port provider is mandatory.

No live GPU or port adapter is included. Saved-resource classification clocks are
not new hardware-sampling clocks; receipts label that distinction. Runtime frontend
registration and actual clocked query/port adapters remain future integration work,
after this local proposal is reviewed. This directory cannot launch anything.

## Qualified CPU evidence

Canonical retained result: `qualified-r12/PROOF.json`: **18 tests PASS,6.295s**;
all908 trial10 inputs exact before/after. Its source map binds every local Python
file. Per-case native traces, lifecycle signals, resource series and cleanup results
are under `qualified-r12/cases/`; all fixture processes were observed gone or
zombie after test cleanup. A Python audit guard denies network calls, GPU devices,
real engine imports and non-Python subprocesses in the parent and fixture children.

`native_fixture.py` compiles the actual pinned `SubprocessWatchdog`, `SignalHandler`
and `sigterm_watchdog` ASTs without modifying their source. Engine/request/GPU
collaborators are CPU substitutes. Both old and proposed paths use a controlled
schedule: the first native poll is parked until TERM, then delayed50ms, leaving the
child watchdog time to observe child death. Later async waits are compressed100x.
This is a source-path/scheduling witness, not actual native GPU shutdown or a race
incidence estimate. An earlier unsynchronized old-path fixture sometimes took the
graceful path (`qualified-r6/tests.log`); that failure is preserved.

The old group-TERM witness records child exit-15, native watchdog SIGQUIT and the
crash path. The proposed path records watchdog-stop before scheduler shutdown,
with no crash path, and only the frontend receives initial TERM. Other tests cover
stubborn owned processes/fallback, persistent captured GPU **and** host failures,
blocked/malformed probes, transient saved/synthetic eligibility, unknown/reused-
identity receipts, missing/source drift, cleanup-delivery fault injection, and
deadline crossing during identity verification. The identity-drift test changes a
recorded starttime; it does not force actual kernel PID reuse. Cleanup failure is
an injected undelivered KILL, not a reproduced kernel D-state.

Reproduce from this directory with a fresh output name:

```bash
timeout 600s python3 verify.py --output independent-local-check
```

Historical red/failed evidence is retained: missing lifecycle behavior; late-signal
regression; graceful-end incorrectly skipping fallback; fixture teardown observing
exit transition before absence; guard blocking a benign uname subprocess; and a
local metric formatter error. Canonical qualification fixes these without edits
to prior trials. Risk metrics in PROOF are an explicitly defined local AST/trace
proxy, not a repository gate; fork-worker coverage is conservatively uncredited.
There is no new dependency or claim of a full implement/commit workflow.

## Independent host blocker

See `HOST-THRESHOLD-TRACE.md`. The original140GiB remaining-headroom requirement is
retained and unresolved. Neither graceful shutdown nor waiting is shown to satisfy
it. No paired capacity inference or candidate production fit follows from this
prototype. The original trial10 remains FEASIBILITY_ONLY, ratio null.
