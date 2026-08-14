# Inline-result decision — 2026-08-12

Remove the direct nested `return_result` tool from the provider-visible surface. `execute_python` is the sole callable tool. Gemma constructs a Python candidate dictionary and completes inside that call with `return_result(candidate)`; NOOA's existing internal signal independently validates the object with Pydantic.

Expose `make_mask(row, col, rows)` in the ephemeral sandbox. It validates `#`/`.` membership rows and derives bbox height and width mechanically. This is representation assistance, not semantic selection: Gemma still chooses the region, mask membership, evidence, lineage, and later action/effect.

This removes the provider's brittle nested-JSON tool-argument boundary while preserving model authorship, host validation, one-action authority, fresh artifact isolation, and bounded repair.
