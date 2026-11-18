# Patch list: q2d skill-following corrections

## Target

- `analysis/local-probes/20260903T-skill-following-q2/perfect-approach.md`
- `analysis/local-probes/20260903T-skill-following-q2/flaws.md`
- `experiments/agents/0158-nooa-tycho-capability-build/third_party/source-snapshots/nooa/solver_agent.py`

## Patch items

1) Reorient solver docs from notes-first to model-first
- Update `ArcSolverBase`, `MdArcSolverAgent`, and `submit_actions` docstrings so the workflow starts from `grid_array` + helper prediction rather than note consultation.
- Reword rationale guidance from generic `"prediction..."` to explicit `predict:` protocol.
- Keep notes API text as “lab book” framing and avoid implying notes are the control source.

2) Enforce rationale contract at submission boundary
- In `submit_actions`, reject calls whose `rationale` is missing or does not start with `predict:` to prevent untyped action spray.

3) Make helper/world model discoverable in state line
- Extend the `MdArcSolverAgent._knowledge_status_line()` hint to report available helper API signatures (especially `encode` and `predict`) alongside note file list and last curation turn.

## Applied (current pass)

- Added execution guards in `solver_agent.py`:
  - Added `prediction_stable` + `prediction_stale_reason` tracking.
  - Added `mark_prediction_stale()` and `mark_prediction_stable()` methods.
  - Added level-change detection in `_sync_turn_guards()`; a level change now forces helper refresh before continuing.
  - Added batch throttle: multi-action submits are rejected when prediction stability is false.
  - `load_helpers()` now clears the level-change refresh gate when a module exposing `encode` + `predict` is successfully loaded.
  - `submit_actions()` now stores pending submit metadata for turn-to-turn guard updates.
  - Added trajectory-based retrodiction validation: multi-step batches must match helper `encode/predict` over the observed grid trajectory before keeping `batch_policy=multi-ok`.
- Added `_prediction_status_line()` and surfaced it in both `MdArcSolverAgent` and `MemArcSolverAgent` state hints as a machine-readable `prediction_status=*|reason=*` token.
- Hardened prediction payload handling to tolerate tuple return forms such as `(next_state, reward)` and dict-based outputs.
- Reduced retrodiction false negatives by coercing richer `predict()` return structures (dict/object/list metadata variants) and comparing latent states with tolerant array/scalar equality.
- Added explicit world model contract in `arc_world_model_contract` and `write_helper()` docstrings (accepted `predict` return formats + encode/predict deterministic expectations).
- Fixed initial prediction policy state to force deterministic single-action starts (`_prediction_stable=False`, explicit reason).
- Corrected pending submit accounting after batch truncation (`_pending_submit_count` now stores executed count, not requested count) to avoid false truncation failures.
- Introduced explicit active helper selection (`_active_world_model_name`) and used it in `_world_model()` to avoid ambiguous retrodiction model choice.
- Enabled automatic stability re-enable when retrodiction succeeds on the prior batch in `_sync_turn_guards()`.
- Constrained pass: allowed single-action retrodiction to actively re-enable batching by validating `_pending_submit_count >= 1` in `_sync_turn_guards()` and removing the early “<=1 returns valid” short-circuit in `_validate_last_submit_retrodiction()`.
