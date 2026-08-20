# tycho-nooa-symmetry-uptake-ar25-20260820T101248Z

Question: With `symmetry` seeded, preloaded, and named in both system prompts, does
either agent EXECUTE it — and if so, does a goal hypothesis become a test?
Posture: uptake of an affordance. Not a gameplay measurement.

Contrast: tycho-nooa-llm-trace-ar25-20260819T162040Z. Same game, same seed, same model,
same 20-action cap, same code path. One change: symmetry seeded + announced. That run
scores 0 on the primary measure (verified before this card was written).

Prior: this project has already measured that seeding a capability is not the same as
using it (durable-surface null, 0 lane writes across five runs). The honest expectation
here is 0. A 0 is a result, not a failure of the run.

Primary measure (PRESPECIFIED, probe written and calibrated before launch):
  tools/symmetry_uptake.py -> executed_symmetry_calls
  = run_python calls whose code matches `symmetry.<name>` AND whose result carries no
  NameError/ImportError/AttributeError. A reach that raises is counted separately as ERR
  and never pooled with a success; an all-ERR run is a WIRING failure, not an uptake null.

Secondary, in order:
  1. lanes: actor vs builder (which reader reaches first, and at which turn)
  2. durable: does `symmetry` reach world_model.py or notes/, or only scratch calls
  3. goal: does outcome() stop being the constant "ongoing"; does any `level_complete`
     predicate appear; PLANNER_STATUS lines in builder-tool-trace.jsonl
  4. levels_completed, actions, stop_reason, cost

Cap: ar25 seed 0; 20 actions; one live model (meta/muse-glimmer-30b, 131072); 60 min
wall-clock; no retries; one attempt.
Stop: action cap, terminal, wall-clock, transport failure.

Invalid if: crash before the first builder pass; ar25 unavailable; symmetry.py absent
from the run's workspace/ (that is a wiring failure and says nothing about uptake);
every symmetry reach is ERR.

Result map:
  executed > 0 and a goal predicate appears -> the vocabulary was the missing piece; scale
  executed > 0, no goal predicate          -> reached for, but not at hypothesis time;
                                              the gap is the loop, not the library
  executed = 0                             -> uptake null; announcing is not enough, and
                                              the next lever is compulsion, not exposition
  all reaches ERR                          -> wiring bug; run is void, fix and re-run

Note: 20 actions cannot answer whether ar25 is solvable, and levels_completed is a
secondary here on purpose. Pooling this with the 60-action positive control is forbidden.
Code: branch refactor/tycho-nooa-core, uncommitted symmetry work on top of 738c94f.
