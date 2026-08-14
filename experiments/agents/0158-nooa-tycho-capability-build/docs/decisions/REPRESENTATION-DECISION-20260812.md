# Grounding representation decision — 2026-08-12

## Decision

Keep `bbox` plus binary `#`/`.` rows for region membership. Allow zero to three alternatives; `[]` means the model has no genuine distinct alternative. If alternatives are supplied, every mask must differ from the primary and from every other alternative.

## Why

Numeric grid-value rows, produced in the first live attempt, echo appearance but do not unambiguously express membership for sparse or multicolored entities. The verifier needs membership, and the evidence store already owns cell values. After the mask encoding became visible, Gemma produced a valid binary mask.

The next run exposed a different mismatch: Gemma duplicated its primary mask as the mandatory alternative. Alternatives are recorded but do not currently participate in verification or promotion. Requiring one therefore manufactured content without strengthening the claim. Empty alternatives preserve honesty and reduce ceremony.

All mechanically checkable claim invariants now exist at both boundaries: model-visible descriptions and Pydantic validation for in-artifact repair, plus independent host validation. Host verdict semantics and action authority are unchanged.
