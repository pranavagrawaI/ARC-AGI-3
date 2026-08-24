# Context/Evidence Plane v1

## Judgment

Bulk evidence and old execution transcripts must not remain resident in every
provider request. Native and CodeAct now have a shared candidate substrate:

```text
identity    content hash
name        opaque handle
permission  invocation-scoped evidence grant
authority   explicit capability
```

`EvidenceAdmissionAuthority` alone acquires, stores, names, and grants evidence.
`EvidenceReadBroker` is the separate reference monitor and the only dereference
path; it has no admission or query method. A handle cannot enumerate or acquire
evidence, and a read returns a JSON copy of an immutable, hash-verified cache
entry. Grants bind session, invocation, role, frontier, handle, content hash,
operations, and expiry. Ending the invocation revokes access without deleting
the cache entry.

## Provider-visible memory

An evidence query returns `EvidenceReceipt`, containing only structural facts:

- handle and content/query hashes;
- canonical evidence references;
- record count and field names;
- encoding, frontier, and exact stored byte count.

The exact page stays in T1. Both Native and CodeAct computation cross the same
`evidence.read` broker operation. Native Python cannot use `evidence.query`; it
can only load values already acquired and granted by the Native evidence tool.

`EvidenceViewRequest` selects explicit records, fields, and rows from a granted
entry under an independent serialized-byte limit. It cannot advance the
frontier or discover a record outside the entry.

## Context containment

The shared computational worker caps stdout and stderr and replaces overflow
with a bounded marker containing the full byte count and SHA-256 digest.
Lossless T1 provenance remains unchanged.

`ContextProjector` keeps the two most recent Native exchanges or CodeAct chunks
exact. Older spans become deterministic structural summaries with counts,
hashes, and bounded receipt orientation. The summaries make no game-semantic
claims.

Exact-request preflight remains the final containment ring; this plane does not
replace it.

## Preserved-trace replay

Run:

```bash
cd solvers/tycho_nooa
env -u TMP -u TEMP TYCHO_SANDBOX_RUNTIME=host \
  uv run --frozen python ../../analysis/context-evidence-plane-v1/replay.py
```

The committed report replays two preserved failures:

| Trace | Before | After |
|---|---:|---:|
| Native call 40 | 74,870 serialized bytes | 6,672 bytes |
| CodeAct sixth call | 11,235 serialized bytes | 5,773 bytes |

Both projected requests fall below the existing conservative 16,384 input-token
bound. Native's three exact evidence pages remain hash-identical in the cache;
CodeAct's two recent execution chunks remain semantically identical after
removing provider-added cache-control wrappers.

The replay also derives and evaluates answers from brokered exact values. The
Native boundary answer recovers `H_LEFT` from `L0:T16` and `L1:T0`. A separate
preserved CodeAct golden-path trace is replayed from its recorded query and
payload: its source, cached, and brokered hashes agree, and decoding `L1:T1`
recovers the accepted `H_history` answer. These are offline attainability proofs,
not claims that either paid six-call trajectory reached the answer.

## Scope boundary

This commit adds and proves the candidate plane and matched controllers. It does
not alter the frozen V1–V3 experiment packages, authorize provider calls, or run
ARC. A new hash-bound pilot package is separate work.
