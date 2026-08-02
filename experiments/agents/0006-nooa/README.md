# 0006-NOOA

`0006-NOOA` is the competence treatment over frozen `0002-NOOA`. Its first
vertical slice makes one action prospectively testable without giving the host
semantic action authority.

The treatment code is intentionally small:

- `competence.py` — archive, neutral evidence operation, claim ledger,
  prediction/reconciliation state machine, and fixed context packet.
- `host/competence_runner.py` — the 0006 host adapter over the frozen 0002
  worker. It carries the commitment through the unchanged final-action channel
  and fails closed before action execution.
- `tests/` — pure protocol and actual offline frozen-worker coverage.

Read [CONTRACT.md](./CONTRACT.md) before adding a mechanism. The candidate and
deletion record for mechanisms held out of 0002 remains in
[NOOA_FEATURES.md](./NOOA_FEATURES.md); executed removal checks are in
[DELETION_LOG.md](./DELETION_LOG.md). Read
[THREAT_MODEL.md](./THREAT_MODEL.md) before filing a verifier finding — it
declares the attacker in scope and closes findings that assume a stronger one.

## Reproduce the current slice

Create the frozen worker's ignored local runtime once:

```bash
cd experiments/agents/0002-nooa/nooa-runtime
uv sync --locked --python 3.12
```

Then, from the repository root:

```bash
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache \
uv run --python 3.14 python -m unittest discover \
  -s experiments/agents/0006-nooa/tests -v
```

The end-to-end fixture makes no provider call. It demonstrates protocol and
boundary function only; it is not a score, capability, or treatment-effect
result.

## Bounded live smoke

`run_live_probe.py` is the single-arm treatment entry point. It is restricted to
LS20 seed 0, offline mode, OpenRouter's Gemma 4 31B wire model, at most three
actions, 12 model calls, 65,536 reported tokens, 16,384 output tokens per call,
four rounds per action, two decision repairs, and zero transport retries. The
reported-token value is a cumulative pre-call gate rather than a provider billing
ceiling; one admitted action costs roughly 16.6k of it, so the earlier 32,768
ceiling could not admit the two-action smoke it was meant to bound.
`--provider-route` selects one of two routes, and the approval records which.
`pinned` (the default) adds the exact `modelrun/fp4` provider slug, FP4 filter,
and disabled fallbacks, so provider identity is exact and the run can qualify a
gate. `open` carries no provider block and lets OpenRouter route freely, for
shaking the harness out without competing for one provider's shared capacity.
An open run cannot qualify: `route_can_qualify` fails in the operational verdict.
That guardrail exists because `0002-NOOA`'s own G9 gate passed a broken substrate
— unpinned load balancing put its later crossings on CoreWeave, which accepts an
unanswered tool call that ModelRun refuses. Both routes travel the same audited
launcher. On the pinned route the overlay adds the slug, filter and fallbacks; the frozen 0002 runtime remains
byte-for-byte unchanged. Parameter filtering is disabled because OpenRouter's
Responses envelope includes optional transport fields absent from ModelRun's
endpoint capability list; tools, tool choice, reasoning, and output limits remain
listed as supported. Only `execute_python` is exposed, and the one-environment-
action bound remains enforced by the host.

Execution has two phases. First use `--prepare-approval-request PATH` with an
explicit unused `--run-root`; this is a no-spend operation that binds the run
identity, complete transitive source hashes and file modes, frozen 0002 revision and lockfiles,
runtimes, provider request, and all budgets into one confirmation digest. Then,
after approving that exact file, invoke the same command with
`--approval-request PATH --confirm-live-run DIGEST` and an inherited
`ARC_LLM_API_KEY`. Any source, spec, or run-root drift invalidates approval, and
the runner atomically renames an accepted request with a `.consumed` suffix
before creating the approved run root, making the grant single-use. Participating
sources are then copied and hash/mode-verified against the consumed approval;
the repository root, source file, and every intervening source-path component
must be real rather than symlinked. The
parent process revalidates the handoff, consumed approval, mandatory snapshot
record, and all source bytes, modes, directory topology, and root metadata
immediately before re-executing the complete
runner under isolated Python import mode. The child then revalidates and
atomically consumes the handoff before invoking the harness, treatment, or
frozen 0002 source. Before creating the live game environment, the worker virtual
environment is content-addressed by the approval—including modes, external
symlink-target type, mode, bytes, root-directory mode, and interpreter identity—and copied
into the run rather than linked to mutable project state. A symlinked worker
environment root is rejected. The provider launcher and transport configuration
have approval-derived expected bytes and explicit modes before materialization;
the materialized overlay must match that external baseline before it can run.
Post-run checks reject source-snapshot additions (including empty directories,
bytecode caches, and symlinks), source byte or mode drift, worker-environment
drift, provider-runtime root mode/symlink drift, or mutation of the executed
provider launcher/config overlay. The approved source set includes the loaded
project package initializers as well as the runner's direct harness/treatment
imports.

The run directory contains the execution manifest, raw ARC recording, public
trace, frozen worker audit, `competence.jsonl`, provider-profile projection, and
the exact raw OpenRouter generation record plus a provider-identity report when
a generation completes. A terminal summary is preserved even when setup,
environment execution, observation, worker shutdown, or evidence finalization
fails. A one-action admission run succeeds only with exactly one admitted action,
one observed successor, complete trace and shutdown, an actual HTTP request
profile matching the approved profile, and returned generation metadata proving
ModelRun served every generation under the exclusively requested FP4 route. A
two-action continuity smoke requires one cross-linked chain with action IDs 1 and
2: both commitments, transitions, predictions, and successors plus the exact
hashed successor context before commitment 2. Observation and transition IDs
are recomputed from their content, and prediction checks are independently
re-evaluated against the linked successor. Each action must also follow the
recorded context → request → model crossing → validated decision → admission →
environment transition → successor-observed order, use an action legal in the linked observation,
and agree across worker, host, competence, and environment records. The raw
model `reason` is parsed by the production parser and replayed through a fresh
production competence controller; its complete replayed event stream must equal
the recorded stream. Process wrappers, public-observation schemas, metadata,
representations, and nested requests use exact producer keys; model-call audits
satisfy an exact treatment predicate that binds the canonical audit snapshot,
top-level messages and tools, successful crossing attempts, and the final
`return_result` payload to the recorded worker decision. Each frozen environment transition—including
pixel digests, changed-pixel count, public state, reset flag, available actions,
and observation paths—must equal the two linked competence observations. The
treatment admits only the approved ls20 `|i1` producer dtype, preserves each
projection's complete shape and digest, and reproduces the producer's full-array
`tobytes()` digest rather
than flattening only its final layer. Action 1's
selected claims must originate in its own ledger segment; action 2 must
preserve, revise, or retire each one in its causally ordered ledger segment.
Natural mismatch IDs are
derived from the failed assertion indexes and must agree across prediction
context, reconciliation event, and commitment 2. Counts without this linked
evidence chain fail.

**Option A — producer consistency:** the acceptance threat model is inside the approved,
immutable execution boundary. It detects malformed, independently altered, or
internally contradictory evidence; it does not claim resistance to a malicious
post-run editor who can rewrite the complete artifact and recompute every
unkeyed hash. That stronger claim requires an external signature, MAC, or
append-only transparency authority and is outside this smoke.

The model-visible competence packet publishes the complete existing field types
and shape limits, requires the final Python workspace cell to construct the
model-authored commitment as a dictionary, and serializes `reason` with
`json.dumps`. The model still authors every semantic value; the host neither
guesses nor repairs malformed or semantically invalid commitments.

This is a live interface-and-behavior smoke, not a matched 0002-vs-0006 result.
The request pins ModelRun FP4 with fallbacks disabled; this establishes requested
routing, not independently returned quantization. The post-run generation lookup
verifies the returned provider identity, model revision, finish reasons, tokens,
cost, and agreement with response usage before allowing an endpoint-specific
operational verdict. Exactly one unique generation ID is required per HTTP
crossing. The metadata lookup refuses redirects, and a
generation ID must pass a strict syntax check before it can name an artifact.
