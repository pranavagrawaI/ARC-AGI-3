# Local release adapter integration

Status: **CPU PLUMBING QUALIFIED LOCALLY; REAL GPU RELEASE UNPROVED**.
Trials1–10 and both accepted shutdown prototypes are unchanged. No external
network/Kaggle/GPU query/model load, cache intervention, staging or commit.

Canonical proof: `qualified-r4/PROOF.json`,11tests PASS4.103s,214 artifact-reference
checks, all908 trial10 inputs and accepted r2 sources exact. Local qualification
uses a600s alarm/external cap and an inherited Python audit guard: only pinned CPU
Python programs, loopback binds, no connects/DNS, GPU devices or native imports.

## What is implemented

- `registration.py`: actual prepared bytes, argv, executable shebang/interpreter,
  native entrypoint/pinned source hashes and observed wrapper/frontend process
  identities. Registration is published complete via exclusive file+hard link;
  no overwrite. Controller binding verifies current identities and exact prepared
  contract, then delegates unchanged identity/signaling semantics to accepted r2.
- `control_prepare_registered.py`: source-only shipping hook candidate. Exact
  original wrapper plus time/registration imports and one post-Popen registration
  call. Native engine, argv/env, prefetch and all measurement behavior unchanged.
  It is syntax/patch qualified, not executed with real preparation/model mounts.
- `capture.py`/`command_child.py`: real CPU subprocess stdout/stderr capture with
  per-command clocks, exclusive bounded raw files, exit/error/cleanup receipts,
  intended versus executed argv and query process keys. The exec helper binds
  PR_SET_PDEATHSIG plus parent check; a killed probe worker cannot leave its direct
  query exec alive in the tested path. This does not cover arbitrary detached
  grandchildren, which the allowed query/CPU fixtures do not create.
- `adapters.py`: GPU/apps command parsers, sequential cgroup file snapshots and
  scope, actual disk query, loopback port bind, native log snapshots/identity/exit
  markers. Original2048MiB/140GiB/100GiB gates remain exact. No active_file credit.
- `artifacts.py`: exclusive raw evidence capped8MiB, exact byte count/SHA and
  mutation/path checks. Large256KiB stdout is retained whole by hash reference,
  keeping the IPC receipt below the accepted64KiB frame bound.

The default executor is CPU-only. A source-only future GPU-query branch accepts
only the two exact argv entries, resolves their executable, and is explicitly
denied throughout local qualification. Its policy label is **not launch authority**;
enabling it requires a later exact reviewed runner/assignment. It was not executed.
INTENDED-COMMANDS.json records intended GPU/apps commands and historical shipping
argv/python from trial10 without launching them or inheriting its identity.

Native text eligibility requires stable log capture, the bound frontend absent,
a graceful-draining marker and no SIGQUIT/crash marker. It does not claim direct
watchdog-stop/GPU-context inspection. The CPU JSON fixture uses the previously
qualified pinned-method ordering witness; actual trial10 crash text is rejected.

## Evidence and limits

Tests exercise the actual prepared-registration path, live CPU process identity,
source/interpreter/contract drift rejection, subprocess exit7/raw stderr, timeout
with preserved partial bytes, parent-death cleanup, immutable artifact hashes and
occupied/free loopback ports. Registration alone initially preceded installed
signal handlers: `local-first.log` preserves that failure. Tests now wait the CPU
readiness barrier before controller binding, matching the proposed lifecycle.

The actual local /tmp has only about5GiB free, so its unchanged100GiB disk gate
correctly blocks release. The integrated test expects that result and separately
asserts the other CPU-observed predicates. It does not manufacture passing disk
evidence. Saved positive cgroup/GPU inputs are explicitly CPU fixtures, not new
hardware measurements. This host's disk result is not a claim about Kaggle.

Per-test RESULT/raw files under `qualified-r4/cases` are primary local evidence;
PROOF binds sources and every case byte. Timeout/parent-death cases may retain
unassociated raw prefixes/intents because the probe worker was killed. Those are
preserved gaps, not complete command receipts. No real GPU release is inferred.

The accepted-core loader currently binds the local r2 directory. Runtime-relative
packaging and actual mounted/native execution remain future reviewed work; this
is not a remotely executable GPU experiment bundle. Host140GiB rationale and
eligibility remain independently unresolved.

Reproduce with a fresh output directory:

```bash
timeout 600s python3 verify.py --output independent-local-check
```

Read RELEASE-FEASIBILITY-PROPOSAL.md for the separate one-control-load question,
fresh3900s platform/3600s cooperative cap, resource gates and what each outcome
can establish. It proposes no replay/candidate/rate comparison or launch authority.
