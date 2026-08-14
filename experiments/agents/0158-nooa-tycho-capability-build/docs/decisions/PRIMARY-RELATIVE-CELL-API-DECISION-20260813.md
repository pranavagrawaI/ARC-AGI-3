# Primary-relative cell effect API decision — 2026-08-13

## Failure boundary

The closed v23 run exposed an interface failure before any ARC action. Qwen wrote
`predict_replace(mask, cell_after(...))`; the old helper silently expected a list
of assertions. A one-factor sandbox replay accepted only the added list wrapper.
That established a schema discriminator, not a valid live prediction: Qwen's
`row=1, col=1` also expressed the center of its 3x3 primary region, while the old
helper interpreted those fields as absolute frame coordinates.

The old API therefore asked the model to reproduce three mechanical details at
once: an absolute coordinate transform, an `expected_after` mask, and a list
container. None was part of the semantic judgment being tested.

## Candidate change

The model-visible replacement form is now:

```python
selected = predict_cells(
    cell_after(d_row=1, d_col=1, value=3),
)
```

- `d_row` and `d_col` are offsets from the top-left of
  `accepted_claim.primary_region`.
- Every offset must name a `#` cell in that accepted mask.
- `predict_cells()` accepts 1–16 `cell_after(...)` arguments variadically; a
  list wrapper is rejected explicitly.
- Offsets must be distinct, values remain integers 0–15, and the derived
  assertion bounding box remains limited to 64 cells.
- The host sorts and expands the assertions to absolute frame coordinates and
  derives the exact `expected_after` mask. The persisted commitment remains the
  existing exact `Replace` protocol record.

`make_mask()` is now physically present only in the claim phase.
`predict_replace()` is absent. Translation, unchanged, contrast, verifier, gate,
evidence, action authority, model, and budgets are unchanged. No action meaning,
object identity, coordinate, or route is supplied.

## Qualification boundary

The offline tests exercise exact relative-to-absolute expansion, multiple
variadic assertions, deterministic ordering, explicit rejection of the old list
form, rejection of cells outside the selected primary mask, post-exploration
constructor submission, and physical phase-surface omission. Host failure
packets report the same reduced callable surface.

This qualifies only the API mechanism. It does not show that Qwen will learn the
directional map or complete LS20. A fresh live run requires a new card and
approval; the closed v23 evidence and approval are not reused.
