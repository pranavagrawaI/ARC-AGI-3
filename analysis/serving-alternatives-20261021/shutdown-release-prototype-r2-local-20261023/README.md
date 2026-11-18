# Local shutdown/release revision2: bounded receipt transfer

Status: **LOCAL CPU REVISION FOR REVIEW**, not integrated or GPU-qualified.
No GPU packet/freeze/slot/launch, native-engine or measurement change, cache or
threshold change, staging or commit. The original submitted directory, its
17PASS/1FAIL independent run and reviewer evidence remain untouched.

## Narrow changes

The malformed-GPU continuation test now uses the coordinator's diagnostic2s
allowance/grace1s/reserve0.5s. It asserts later receipt visibility and releasefalse;
it no longer requires whole-release scheduling within250ms. All fourteen other
original shutdown tests, including every original deadline/negative test, are
byte-exact at the function level. The identity/signaling/release functions are
also byte-exact. `verify.py` checks those facts and prior evidence hashes.

`observe` replaces multiprocessing pickle messages and blocking `recv` with
`probe_ipc.py`: a nonblocking pipe and a four-byte length-prefixed JSON frame.
Header/payload reads occur in chunks of at most8192 bytes with one absolute
transfer deadline. Decode/validation also check that end before returning.
There is no poll-then-blocking-recv or pickle deserialization path.

The explicit receipt contract is:

- At most65536 encoded payload bytes per frame, six ordered named probe frames,
  and clean EOF. An extra, duplicate or out-of-order probe invalidates the sample.
- JSON types only; finite numbers, integers at most128bits, depth<=12,
  at most2048 value nodes, string length<=16384 and key length<=256.
  Scalar/key encoding is budgeted during validation before building the full
  encoded frame. Duplicate JSON keys and nonfinite values are rejected.
- Provider fields cannot override name/start/end/transfer metadata. Parent
  validation checks typed eligibility, exact order, clocks and the transfer end.
- Oversized or unsupported provider values become a small explicit ineligible
  receipt; they are not truncated into a successful observation. Large raw
  artifacts will require separately bounded file/hash references in future
  adapters. This revision does not implement those adapters.
- Oversized announced length is rejected before body allocation. Partial EOF,
  stalled header/body, incomplete receipt set and worker/transfer failure all
  invalidate the sample. Already observed predicates remain visible; missing
  predicates stay explicitly unobserved/ineligible.

Process observation start/end and IPC deadline are separate fields. The transfer
budget does not include preceding identity/process inspection. Syscalls, Python
execution and scheduling remain cooperative; no hard kernel/subsecond bound is
claimed. All qualification invocations have the external600s cap and alarm.

## Canonical retained evidence

`qualified-r1/PROOF.json`: **27 tests PASS,9.812s**, Python audit guard active,
GPU/network0, all908 trial10 bindings exact before/after. It binds all local Python
sources, case receipts and prior-revision checks. Source-path native CPU fixtures,
the unchanged thresholds and limitations are inherited from revision1.

The isolated malformed test captured26 samples with later port receipts and
releasefalse. Direct partial-header and stalled-payload reads with50ms transfer
budgets returned TransferDeadline around50.2–50.3ms in this run. These are measured
local timings, not worst-case guarantees. The source ends and verdict stay
fail-closed if execution runs late.

The full observation tests first establish an empty owned-process set, then inject
one fault while other receipt predicates are true. A64MiB byte value is rejected
as a bounded error and later probes remain visible. A64MiB announced frame, partial
EOF, stalled frame and extra seventh frame all produce ineligible observations;
workers are gone. The extra-frame case retains six otherwise-eligible predicates,
so failure cannot be credited to an unrelated resource gate. These are CPU fault
injections, not actual GPU queries, port evidence or device-release claims.

Retained pre-implementation RED: `red-framing.log`. Preliminary IPC checks are in
`ipc-first.log` and `ipc-integrated-first.log`. The canonical complete proof is the
fresh guarded invocation above; it does not erase revision1's failing review.

Reproduce from this directory with a fresh output name:

```bash
timeout 600s python3 verify.py --output independent-local-check
```

`PRIOR-SNAPSHOT.json` binds the original submitted sources and review/failure
receipts. `LOCAL-DELTA.json` binds this revision, the prior review and canonical
qualification. These are local evidence indexes, not launch freezes/authority.

## Unchanged unresolved work

The140GiB host-headroom gate and its missing load-derived rationale remain
independent. See the preserved revision1 `HOST-THRESHOLD-TRACE.md`. No threshold
is fitted to trial10's failure. Runtime frontend registration, actual query/port
adapters, actual native GPU release and a same-session paired comparison remain
unqualified and require subsequent scoped integration/review.
