# Exact effect interface decision — 2026-08-12

The first Qwen live run crossed grounding but exposed split validation: the
sandbox accepted any `predicted_effect` dictionary, while the hidden host parser
required an exact discriminated union.

The repaired runtime uses exact strict models for translate, unchanged, and
bounded replace. Phase-specific sandbox constructors make syntax mechanical:

- `translate_primary(d_row, d_col)`
- `primary_unchanged()`
- `replace_primary(expected_after, assertions)`
- `cell_after(row, col, value)`
- `submit_probe(action, effect, data=None)`

Claim phase separately exposes `submit_grounding_claim(...)`. The runtime and
host parity test requires every runtime-accepted effect to round-trip unchanged
through the host parser and rejects Qwen's prior invented representation.

No constructor chooses a region, action, displacement, value, or prediction.
Before ARC contact, Qwen must produce both a typed claim and a typed commitment
from synthetic evidence through the exact worker.
