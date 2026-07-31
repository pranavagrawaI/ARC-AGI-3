# 0002-NOOA

Status: offline tracer bullet implemented; qualification is partial.

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
- source-manifest verification and mutations that reintroduce hidden history or
  runtime prompt terms.

This does **not** yet license a live comparison. The provider adapter, transport
classification/retries, full evidence commitments, the remaining mutation
matrix, and G9 are still open. See [`QUALIFICATION.md`](./QUALIFICATION.md).
