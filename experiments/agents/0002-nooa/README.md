# 0002-NOOA

Status: provider-capable tracer implemented; offline qualification is partial.

`0002-NOOA` is the 0002 minimum Python REPL agent implemented using NOOA. It is a candidate base for the planned `0006-NOOA`, not a 0006 implementation and not the existing NVIDIA NOOA ARC agent under a new name.

The documents have separate authority:

- [`../NOOA_IDEOLOGY.md`](../NOOA_IDEOLOGY.md) — shared implementation ideology and direction checks;
- [`CONTRACT.md`](./CONTRACT.md) — stable semantic capability and host/model authority;
- [`QUALIFICATION.md`](./QUALIFICATION.md) — executable parity and rejection gates;
- [`IMPLEMENTATION.md`](./IMPLEMENTATION.md) — current copy-and-prune engineering plan;
- [`SOURCE_MANIFEST.json`](./SOURCE_MANIFEST.json) — pinned upstream provenance and planned copies.

## Family

```text
0002-R    → planned 0006
0002-NOOA → planned 0006-NOOA
```

First compare `0002-R` with `0002-NOOA` as bases. If `0002-NOOA` is retained, compare the later `0006-NOOA` treatment against this unchanged base.

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
projects the frozen `0002-R` observation/workspace surface, and crosses a
versioned append-only JSONL boundary. The worker exposes only generic persistent
Python/files and one typed final action. Offline fixture tests currently exercise:

- cumulative Python memory and model-authored files across environment actions;
- exact current text/image observation crossings and a normalized `0002-R` parity report;
- complete-turn eviction with a pointer-only omission notice;
- generated-code import, object, filesystem, network, and runtime restrictions;
- syntax/runtime/malformed-action repair, output bounds, timeout restart, budgets,
  process death, and process-tree cleanup;
- an immutable Responses request profile with NOOA and LiteLLM hidden retries
  disabled;
- canonical and actual HTTP crossing evidence with credential/image redaction;
- retryable, fatal, and uncertain transport injection, including the temporal
  barrier that keeps fatal/uncertain failures out of NOOA's repair loop and
  end-to-end evidence that no failed transport path creates a host action;
- source-manifest verification and mutations that reintroduce hidden history or
  runtime prompt terms.

[`run_live_probe.py`](./run_live_probe.py) is a bounded single-arm provider
smoke helper, not the paired G9 comparison. This does **not** yet license
running it. The remaining G3/G4/G5/G7/G8 matrices must close, G1-G8 must be
rerun, and the same-spec 0002-R/0002-NOOA paired runner must be constructed.
See [`QUALIFICATION.md`](./QUALIFICATION.md).
