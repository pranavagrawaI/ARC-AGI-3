# Designer-player interface v2

Implemented the four measured reductions to serial CodeAct overhead while
preserving prediction and action authority.

## Changes

- Added `changed_region()` for bounded exact before/after crops around the latest
  measured transition, plus `grid_summary()` for measured shape, color counts,
  and color bounding boxes without object labels.
- Added `model_grid()` as the canonical detached NumPy integer input for helper
  encoders. `preflight_model()` now reports the actual input shape/dtype and
  JSON-latent contract; string grid rows are normalized consistently.
- Extended `render_grid()` with half-open `bbox` selection, a character cap,
  and explicit truncation metadata. It no longer silently loses output.
- Added a one-cell pre-action workflow template to the Actor guidance. Writes,
  loads, preflight, and experiment recording happen before `submit_actions()`;
  prediction checks, one-action exploration, and earned batching remain intact.

## Verification

- Full candidate suite: **63 passed**, 3 existing `set_dynamic` deprecation
  warnings.
- Notebook rebuilt and `prepare_candidate.py --check` passed.
- Mocked Kaggle package verification passed with zero network calls and the
  expected ARC-AGI-3 / RTX PRO 6000 binding.
- New tests cover region crops, bounded rendering, canonical model-grid input,
  and preflight contract metadata.

Current contract:
`3d6b8746e09de53313c22259b31b3044dccb5b1a5ca729ff1c1532398479e467`.

No OpenRouter, Kaggle upload, or paid launch was performed for this change.
