# Atomic commitment decision — 2026-08-12

The v20 live run accepted its second claim, then Qwen spent all 12 commitment
iterations inspecting already-visible evidence and constructors. Its final cell
successfully made `primary_unchanged()` but never closed `submit_probe(...)`.
This is evidence of procedural closure failure, not a semantic contradiction.

The next candidate changes only the closure interface:

- Remove the two-step effect constructors and `submit_probe`.
- Expose atomic `commit_translate`, `commit_unchanged`, and `commit_replace`.
- Count successful exploratory cells in host-side CodeAct interception.
- After three explorations, execute only a single-expression direct atomic
  submission; reject all other cells before their requested code can run.

The helpers still require Qwen to choose the region, action, and predicted
effect. The host supplies only claim binding and schema construction. Evidence,
verifier, G0-G2 gates, autonomous reset policy, model, provider route, budgets,
and 44-action cap remain unchanged. Semantic handoff is deliberately deferred:
adding it now would prevent this run from isolating closure ergonomics.

Offline qualification passed 36 host tests, 15 runtime tests, both Ruff checks,
and the exact provenance audit. The regression test demonstrates that a
multi-statement fourth call is physically rejected and a later direct atomic
submission succeeds.
