# Transition and helper interfaces repaired locally

Current contract:
`6db629026e9681f27cd91a56463a8f16bf2ef1dbf92d49b1d852a9ec8d07f3a6`.
No paid calls, Kaggle upload/launch, reasoning-budget or sampling changes.

## Changes

- `last_transition()` returns labelled measured evidence: step/action, before/after
  hashes, exact changed-cell count, bbox, complete color-transition counts, and
  up to64 coordinate/color changes with an explicit omitted count. Requesting
  grids returns `before_grid_rows` and `after_grid_rows`. Before any action it
  returns None. The compact decision packet now includes this measured view.
- Existing `trajectory()` keys/behavior remain compatible. Its documentation
  explicitly explains that one entry is a transition, not a frame, and gives
  the correct before/after comparison for the one-action case. Actor guidance
  no longer promises absent `diff_summary` or animation data.
- `preflight_model(actions)` calls the actual prediction preparation path,
  including helper selection, validation, computation budget, JSON normalization
  and latent limits. It returns ready/unavailable plus source identity and error;
  it does not commit predictions, submit actions, or earn batching. The direct
  tuple-based helper test can succeed while this preflight reports the real
  unhashable-list error, before spending an environment action.
- Actor guidance distinguishes measured facts from object/dynamics/goal hypotheses,
  asks for contradictions to be corrected or withdrawn, and recommends testing
  in the same cell once decisive evidence exists. A passing execution check is
  explicitly not a confirmed game rule. Exploratory single actions remain allowed.

The agent-instruction skill influenced where this guidance was placed: beside
the APIs and in the actual handle workflow, with executable names/examples,
rather than as another detached list of principles. This is guidance, not an
enforced guarantee that generated claims will be true.

## Verification

Full suite: **61 passed**, two occurrences of the existing `set_dynamic`
deprecation warning, 25.31 seconds.

- Captured live transition test exposes all52 changes directly and verifies the
  five0/1 cluster cells are unchanged. Exact before/after grids are available.
- Zero-change, shape-change and truncated-coordinate tests preserve honest
  counts and do not mutate input evidence.
- Preflight reproduces runtime JSON normalization failure, then succeeds with
  the corrected helper. Source identity changes are visible. No commit files
  appear from preflight; batch qualification remains unearned.
- Real Actor methods and legacy trajectory behavior are exercised.
- Actual subprocess/HTTP/CodeAct test calls the new APIs, observes unavailable
  then ready preflight, and submits one action. Only that submission writes a
  prediction record. The same test passes against notebook-extracted code.
- Existing lifecycle, cancellation, action-continuity, prediction, reasoning and
  package tests remain green. Independent scoped review found no blocker.
- Notebook rebuilt; `prepare_candidate.py --check` passes. Mocked Kaggle CLI
  verification passes with zero network calls, ARC competition binding and
  RTX PRO 6000 selection.

The previous live6b8a5c1d source was archived before editing, and every archived
source hash was verified against its original manifest. Failed-run evidence is
unchanged. This successor is locally qualified for a fresh bounded live test;
no gameplay improvement, speedup, or live prediction-feedback qualification is
claimed. `launch_ready=false` and `openrouter_qualified=false` remain unchanged.
