# 0002-NOOA

Status: frozen minimal substrate. Offline G1–G8 and bounded current-byte live G9
pass; gameplay competence is deliberately not claimed.

`0002-NOOA` is the 0002 minimum Python REPL agent implemented using NOOA. It is a candidate base for the planned `0006-NOOA`, not a 0006 implementation and not the existing NVIDIA NOOA ARC agent under a new name.

The documents have separate authority:

- [`../NOOA_IDEOLOGY.md`](../NOOA_IDEOLOGY.md) — shared implementation ideology and direction checks;
- [`CONTRACT.md`](./CONTRACT.md) — stable semantic capability and host/model authority;
- [`QUALIFICATION.md`](./QUALIFICATION.md) — executable parity and rejection gates;
- [`IMPLEMENTATION.md`](./IMPLEMENTATION.md) — current copy-and-prune engineering plan;
- [`SOURCE_MANIFEST.json`](./SOURCE_MANIFEST.json) — pinned upstream provenance and planned copies.

## Family

```text
0002-R                   → planned 0006
0002-R-minimal ↔ 0002-NOOA  (usability qualification only)
0002-NOOA                → planned 0006-NOOA
```

Frozen `0002-R` remains an unchanged historical reference. The executable live
pair uses `0002-R-minimal`, an explicitly derived no-helper compatibility arm;
it tests interface usability but is not a causal comparison with frozen
`0002-R`. If `0002-NOOA` is retained, compare the later `0006-NOOA` treatment
against this unchanged NOOA base.

## Build strategy

Start from the working NOOA ARC agent integration at pinned commit `bd51d16e2594f0bc237b14fb31e9a880ec385b39`, then delete memory, retrieval, summarization, reflection, semantic skills, action batching, and other unmatched cognition until the contract passes.

For every mechanism:

```text
NOOA-like implementation AND 0002-R-like function
→ keep

otherwise
→ delete from 0002-NOOA
→ record in ../0006-nooa/NOOA_FEATURES.md
```

NOOA requires Python `<3.14`, while this repository requires Python `>=3.14`. The NOOA agent will therefore run in a nested Python 3.12 project and communicate with the shared ARC host through narrow file IPC. NOOA will not be added to the root `pyproject.toml`.

Features removed from this base but potentially useful later are recorded in [`../0006-nooa/NOOA_FEATURES.md`](../0006-nooa/NOOA_FEATURES.md). That inventory is not a treatment specification.

## Current executable slice

The host-side `NooaProcessAgent` starts one episode-scoped Python 3.12 worker,
projects the minimal no-helper Python/files surface, and crosses a
versioned append-only JSONL boundary. The worker exposes only generic persistent
Python/files and one typed final action. Offline fixture tests currently exercise:

- cumulative Python memory and model-authored files across environment actions;
- exact current text/image observation crossings and a normalized
  `0002-R-minimal` compatibility report;
- complete-turn eviction with a pointer-only omission notice;
- generated-code import, object, filesystem, network, and runtime restrictions;
- syntax/runtime/malformed-action repair, output bounds, timeout restart, budgets,
  process death, and process-tree cleanup;
- an immutable Responses request profile with NOOA and LiteLLM hidden retries
  disabled;
- explicit Chat-multimodal and NOOA successor-history projection onto the
  Responses wire schema;
- local-only LiteLLM import startup with implicit `.env` loading disabled;
- canonical and actual HTTP crossing evidence with credential/image redaction;
- retryable, fatal, and uncertain transport injection, including the temporal
  barrier that keeps fatal/uncertain failures out of NOOA's repair loop and
  end-to-end evidence that no failed transport path creates a host action;
- source-manifest verification and behavioral crossing mutations for every
  forbidden affordance;
- exhaustive process/environment/package/cache/IPC isolation probes,
  restarted-worker pruning, concurrency closure, accepted-action workspace
  commitments, explicit host execution-failure records, and deletion mutants;
- a per-crossing actual-fixture parity catalogue sourced from host-owned
  observation commitments and an offline-tested
  [`run_paired_probe.py`](./run_paired_probe.py) entry point for G9.

The G9 runner commits one neutral capability instruction to both arms: use
Python and reuse one file across the first successor. Its verdict requires that
evidence plus at least two actions, complete traces, clean shutdown, and exact
actual `/responses` profile projections. A blocker returns a nonzero status.

[`run_live_probe.py`](./run_live_probe.py) remains a bounded single-arm smoke
helper and is not G9. Offline G1–G8 and the same-spec
[`run_paired_probe.py`](./run_paired_probe.py) runner is green offline and live.
The accepted current-byte G9 used LS20, `openai/google/gemma-4-31b-it` routing,
`google/gemma-4-31b-it` on the wire, and OpenRouter's `/responses` endpoint.
It establishes bounded interface usability for the minimal surface, not
gameplay competence or cross-game generality. See
[`QUALIFICATION.md`](./QUALIFICATION.md).

The separate 44-action LS20 single-arm tracer is packaged as compact commitments
in [`artifacts/FREEZE_EVIDENCE.json`](./artifacts/FREEZE_EVIDENCE.json); its raw
provider evidence remains local and unstaged. It establishes long-horizon
crash-resistance for null-data actions, not paired parity or LS20 competence.
Coordinate action and multi-layer last-layer routing have deterministic end-to-end
closure tests. The bounded online CN04 click smoke completed two validated
coordinate clicks with successor transitions; the third decision hit its declared
reported-token budget, while the worker stopped cleanly. This is route evidence,
not gameplay competence. The final current-byte G9 separately qualifies the
no-helper minimal surface.

A separate CN04 tracer completed 50 validated coordinate clicks and 50 successors,
then stopped at its action budget with worker exit code 0. It is retained as
long-horizon crash-resistance/route evidence only; it neither solved CN04 nor
establishes general gameplay competence.

## Handoff to competence

The accepted current-byte G9 ends 0002 substrate work: the evidence shows that a
model can observe, use persistent Python and files,
take simple or coordinate actions, receive successors, exhaust a declared
budget, and stop without either process crashing. The next stage will not
enrich 0002. It will build the competent agent, using NOOA as the design
architecture and Tycho as the oracle from which mechanisms are modified or
deleted until the harness fits the target small model. Competence claims and
Tycho-derived mechanisms belong to that later agent, not this frozen control.

## Inspect a paired run

The read-only NOOA evidence viewer replays public frames and aligns each arm's
action point with its recorded provider text, tool or Python use, final action,
workspace evidence, and successor transition where present. For NOOA it also
shows the typed worker decision, host validation, and workspace commitment. It
serves only allowlisted recorded artifacts, binds only to loopback, and never
starts an agent or changes a run.

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPYCACHEPREFIX=/tmp/nooa-viewer-pycache \
  uv run python experiments/agents/0002-nooa/output_viewer.py
```

Open <http://127.0.0.1:8766/>. Select a paired G9 run and either `0002-NOOA`
or the control identity recorded by that run. The site labels the displayed reasoning as provider-returned text
and the runs as usability evidence, not gameplay-competence evidence.

To inspect single-arm provider-smoke runs such as the 44-action LS20 traces,
point the same viewer at their parent directory:

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPYCACHEPREFIX=/tmp/nooa-viewer-pycache \
  uv run python experiments/agents/0002-nooa/output_viewer.py \
  --runs-root experiments/agents/0002-nooa/artifacts/provider-smoke-probes/ls20
```

The picker identifies these as `Single-arm run`; there is no control selector
because no matched control artifact exists in that layout.
