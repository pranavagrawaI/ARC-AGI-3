# Implementing 0002-NOOA

Status: current engineering plan. It may change without changing the semantic identity in [`CONTRACT.md`](./CONTRACT.md), provided [`QUALIFICATION.md`](./QUALIFICATION.md) still passes.

Before changing direction, read the shared [`NOOA implementation compass`](../NOOA_IDEOLOGY.md).
The implementation should feel like native NOOA software while exposing only the function
authorized by the `0002-R`-like control boundary.

## Keep/delete decision rule

For every candidate mechanism, ask both questions:

1. **Does its implementation look NOOA-like?** It should use NOOA's native agent/runtime mechanisms rather than reconstructing another framework inside it.
2. **Does its function look 0002-R-like?** It should reproduce a capability or information surface already present in the repaired minimal Python/files base, without selecting evidence, maintaining beliefs, enforcing predictions, reconciling mismatches, or supplying richer context.

```text
NOOA-like implementation AND 0002-R-like function
→ keep in 0002-NOOA

otherwise
→ delete from 0002-NOOA
→ record in ../0006-nooa/NOOA_FEATURES.md
```

The deletion log includes rejected features as well as plausible `0006-NOOA` candidates. Deletion is not a promise to reintroduce.

## Runtime separation

Current repository constraints:

```text
arc-agi-3: Python >=3.14
NOOA:      Python >=3.12,<3.14
```

Initial implementation:

- keep the shared ARC host in the root Python 3.14 project;
- place the NOOA agent in `nooa-runtime/`, a nested Python 3.12 uv project;
- do not add NOOA to the root `pyproject.toml`;
- cross a narrow, versioned file-IPC boundary;
- keep the mutable checkout at `/home/hipra/projects/nooa` as source material, not a runtime dependency;
- pin the runtime dependency to the source commitment in `SOURCE_MANIFEST.json`.

This layout is contingent. A future compatible in-process runtime may replace it without changing the contract if qualification still proves the same model-visible boundary.

## Current tree

```text
experiments/agents/0002-nooa/
├── README.md
├── CONTRACT.md
├── QUALIFICATION.md
├── IMPLEMENTATION.md
├── SOURCE_MANIFEST.json
├── run_live_probe.py
├── nooa-runtime/
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── agent.py
│   ├── launcher.py
│   └── llm.py
├── host/
│   ├── qualification.py
│   └── runner.py
├── tests/
│   ├── test_host.py
│   └── test_qualification.py
└── third_party/nooa/
    ├── LICENSE
    └── THIRD_PARTY_NOTICES.md
```

The nested runtime also owns its Python 3.12 tests and lockfile.

## Copy-and-prune source

Pinned provenance and source hashes live in [`SOURCE_MANIFEST.json`](./SOURCE_MANIFEST.json).

Initial copy seeds:

| NOOA source | Local destination | Retained reason |
|---|---|---|
| `examples/arc_agi_3/solver_agent.py` | `nooa-runtime/agent.py` | Working `InteractiveAgent`/CodeAct, turn dispatch, state ingestion, and action-yield seams |
| `examples/arc_agi_3/launcher.py` | `nooa-runtime/launcher.py` | Working headless dispatcher, queue attachment, startup, and shutdown seams |
| `examples/arc_agi_3/arc_llm.py` | `nooa-runtime/llm.py` | Working NOOA completion-client and provider wiring |
| `LICENSE` | `third_party/nooa/LICENSE` | Apache-2.0 terms |
| `THIRD_PARTY_NOTICES.md` | `third_party/nooa/THIRD_PARTY_NOTICES.md` | Upstream notices |

Do not copy the NOOA framework, harness, recorder, viewer, experiment fleet, skills, memory package, analysis, or run artifacts. Import the pinned framework dependency and write a host bridge against this repository's environment seam.

## First-green order

The tracer bullet completed the source-copy, nested-runtime, cumulative-loop,
forbidden-surface, and first parity slices. The remaining first-green work is:

- [x] Replace the offline fixture client with the copied-and-pruned Responses
  adapter without changing the audited model-visible surface.
- [x] Freeze the credential-free provider request profile in `PairedRunSpec`.
- [x] Disable both NOOA retry budgets and LiteLLM retries; keep the bounded
  transport loop explicit and audited.
- [x] Record canonical and actual HTTP crossings with credential/image
  commitments instead of credential/image bodies.
- [x] Inject retryable, fatal, and uncertain transport failures; prove fatal
  and uncertain outcomes cannot enter NOOA's generation-repair loop, and prove
  fatal, uncertain, and exhausted-retryable outcomes produce no host-selected
  action with exact attempt evidence.
- [ ] Finish the G3/G4/G5/G8 negative and mutation matrices.
- [ ] Extend G7 from the exercised provider crossing to the complete parity
  catalogue and rerun G1-G8.
- [ ] Construct and run the paired 0002-R/0002-NOOA G9 probe only after G1-G8
  are green. `run_live_probe.py` is a single-arm provider smoke helper, not G9.

Do not reimplement provider normalization, dispatch, yield semantics, or lifecycle from memory before the copied path has run.

## Isolation design

NOOA's live-object strength is deliberately not exposed to generated Python in this base.

The agent process and generated-code process must have separate capability views. Generated code receives only:

- its isolated Python interpreter;
- the run-local model-writable workspace;
- explicitly projected public observation files;
- documented generic packages and builtins.

It receives no object references or readable paths into:

- NOOA agent/context/event/client objects;
- ARC host/controller objects;
- hidden event stores or context blocks;
- provider configuration or credentials;
- IPC transport;
- trace/run artifacts;
- agent source or prompt source;
- host environment and process arguments.

The host-side parity manifest derives effective visibility from an exercised sandbox probe, not from configuration claims alone.

## IPC authority

The IPC protocol should be append-only and bind each record to one run and action point.

Minimum host→agent record:

- protocol version;
- run ID and action-point ID;
- current public observation projection paths and content commitments;
- exact permitted public status fields;
- legal event IDs and coordinate schema;
- paired-run context/budget declaration.

Minimum agent→host record:

- protocol version;
- run ID and action-point ID;
- one final `{action,data,reason}` object;
- completion marker.

The host rejects stale, duplicate, future, malformed, or mismatched records. Action acceptance closes the agentic method before environment mutation. Audit records are stored outside generated-code visibility.

## Overdelete after first green

```text
working copied spine
→ remove one mechanism
→ run the independent qualification check
→ keep it deleted when no check fails
→ if a check fails, restore only the smallest function that repairs the failure
→ record every deleted NOOA feature in the maintained deletion log
```

The goal is not a polished adaptation of NVIDIA's ARC agent. It is the smallest native-NOOA implementation of the repaired 0002 capability boundary.

## Current implementation limitations

Do not yet claim:

- hostile-code security;
- endpoint compatibility;
- provider determinism;
- cross-platform Python/runtime compatibility;
- complete G1–G8 qualification;
- live usability or gameplay parity.

The Linux/WSL nested-project and IPC path has survived deterministic fixtures,
including timeout and process-death cleanup. That is evidence for this substrate,
not a cross-platform claim. These are engineering risks and qualification
targets, not semantic clauses in `CONTRACT.md`.
