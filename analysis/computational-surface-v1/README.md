# Computational Surface v1

## Claim

Native provider tooling and sterile CodeAct use the same authority-free RLE
decoder. They differ only in invocation syntax:

```text
rle_decode_row(row=...)  -> decode_rle_row(row) <- rle.decode_row(row)
```

Neither surface accepts an evidence reference, retrieves a record, selects a
row, or accesses solver state. The fixed decoded-row limit is 4,096 cells.

## Zero-cost proof

`tests/test_computational_surface.py` checks:

- identical values for the V13 `11 -> 5` rows, short and alternating runs,
  full-width rows, and the boundary-sized row;
- equivalent rejection classes for empty, malformed, non-text, zero/negative,
  and oversized encodings;
- canonical JSON value parity across the real sterile process membrane;
- a Native adapter schema containing only `row`;
- absence of evidence, workspace, state, and self from the sterile namespace;
- replay of the `11 -> 5` calculation through the Native adapter; and
- live hashes matching `manifest.json`.

The manifest binds the decoder implementation, Native input schema, sterile
helper contract, sterile worker, and manifest version. Frozen V12/V13
controllers are unchanged.
