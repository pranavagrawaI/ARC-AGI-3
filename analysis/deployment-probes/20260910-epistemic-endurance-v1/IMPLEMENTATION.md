# Implemented contract

Runtime: deployment/kaggle/submission-runtime-20260910-v6.
Notebook: deployment/kaggle/submission-20260910-v9.

- Actor exploration no longer requires a fabricated goal or hypothesis pair.
  Instructions distinguish observations, scoped mechanics/objective claims, and
  conditional plans; notes preserve evidence steps and unresolved conditions.
- Typed experiment results carry measured_prediction_only authority. Matching
  a proposed consequence cannot certify the hypothesis or knowledge update.
- Optional bounded latent planner: declarative local targets, shared model
  selection and validation, detached finite JSON states, separate limits for
  generated successors/frontier/depth/actions, parent-enforced wall deadline,
  callback budgets, and fresh-instance replay before returning a path.
- Distinct path_found/already_satisfied/model_exhausted/limited/model_error/
  invalid_request/unavailable results; no inferred host outcome labels.
- Search references bind submitted prefixes to the current turn, step, level,
  grid digest and helper hash. Existing prediction/retrodiction batch gates
  remain authoritative. Source and diagnostics persist in search_results.jsonl.
- Planner advertised alongside helpers and exercised through literal CodeAct.

Validation: 99 runtime tests, 35 notebook/real-entrypoint tests, 4 contingent
worker tests, freeze verification and exact notebook embedded-byte checks.
The first full suite overlapped final source edits and failed the drift guard;
its evidence is retained in pre-freeze-checks/. The final frozen suite passed.
Original repaired v5 and historical v4 hashes were verified unchanged.

Limits: POSIX main-thread fork required; otherwise search fails closed. The
parent wall deadline contains callbacks even if the helper catches its alarm.
Semantic uncertainty management remains an Actor behavior to evaluate, not
something proven by a prompt, result schema, or deterministic fixture.

Remote: one same-format public endurance run, kernel version 7. Full operational
and behavioral interpretation awaits terminal artifacts. This is not an A/B
experiment; changes cannot be individually credited for an outcome.
