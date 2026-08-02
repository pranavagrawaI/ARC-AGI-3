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
[DELETION_LOG.md](./DELETION_LOG.md).

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
actions, 12 model calls, 32,768 reported tokens, 16,384 output tokens per call,
four rounds per action, two decision repairs, and zero transport retries. The
reported-token value is a pre-call gate rather than a provider billing ceiling.
The treatment-side transport overlay adds the exact `modelrun/fp4` provider slug,
FP4 filter, and disabled fallbacks to the request; the frozen 0002 runtime remains
byte-for-byte unchanged. Parameter filtering is disabled because OpenRouter's
Responses envelope includes optional transport fields absent from ModelRun's
endpoint capability list; tools, tool choice, reasoning, and output limits remain
listed as supported. Only `execute_python` is exposed, and the one-environment-
action bound remains enforced by the host.

Execution has two phases. First use `--prepare-approval-request PATH` with an
explicit unused `--run-root`; this is a no-spend operation that binds the run
identity, complete transitive source hashes, frozen 0002 revision and lockfiles,
runtimes, provider request, and all budgets into one confirmation digest. Then,
after approving that exact file, invoke the same command with
`--approval-request PATH --confirm-live-run DIGEST` and an inherited
`ARC_LLM_API_KEY`. Any source, spec, or run-root drift invalidates approval, and
the runner atomically renames an accepted request with a `.consumed` suffix
before creating the approved run root, making the grant single-use. Participating
sources are then copied and hash-verified against the consumed approval; the
treatment overlay and frozen 0002 source execute from that run-local snapshot,
not from mutable working-tree source.

The run directory contains the execution manifest, raw ARC recording, public
trace, frozen worker audit, `competence.jsonl`, provider-profile projection, and
the exact raw OpenRouter generation record plus a provider-identity report when
a generation completes. A terminal summary is preserved even when setup,
environment execution, observation, worker shutdown, or evidence finalization
fails. A one-action admission run succeeds only with exactly one admitted action,
one observed successor, complete trace and shutdown, an actual HTTP request
profile matching the approved profile, and returned generation metadata proving
ModelRun served every generation under the exclusive FP4 route. Multi-step runs also
require at least two admitted actions and a predecessor prediction result exposed
before the next context. A malformed competence commitment remains a failed smoke
with no host-selected replacement action.

The model-visible competence packet publishes the complete existing field types
and shape limits, requires the final Python workspace cell to construct the
model-authored commitment as a dictionary, and serializes `reason` with
`json.dumps`. The model still authors every semantic value; the host neither
guesses nor repairs malformed or semantically invalid commitments.

This is a live interface-and-behavior smoke, not a matched 0002-vs-0006 result.
The request pins ModelRun FP4 with fallbacks disabled; the post-run generation
lookup verifies the returned provider identity, model revision, finish reasons,
tokens, cost, and agreement with response usage before allowing an endpoint-
specific operational verdict. The metadata lookup refuses redirects, and a
generation ID must pass a strict syntax check before it can name an artifact.
