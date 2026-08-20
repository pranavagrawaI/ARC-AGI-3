# tycho-nooa-goal-gate-ar25-20260819T120904Z
Question: In vivo, does gate 0 fire on the seed outcome(), and does the builder answer it by writing a candidate level_complete predicate?
Posture: verification of a mechanism, not of gameplay.
Contrast: 175557Z (same game, same seed, no gate; 6 searches, 0 predicates, 0 levels).
Prediction: the first builder pass prints PLANNER_STATUS: not_run_goal_unsatisfiable; zero no_plan_within_budget / planner_timeout while outcome() is the seed constant; a later pass writes "level_complete" into outcome() and the gate lifts to PLANNER_CONTRACT.
Measure: builder-tool-trace.jsonl counts of not_run_goal_unsatisfiable, PLANNER_CONTRACT, no_plan_within_budget, planner_timeout; grep "level_complete" in workspace/ar25/world_model.py and the turn it first appears.
Cap: ar25 seed 0; 20 actions; stop-after-levels 1; one live model; 45 min; no retries.
Stop: action cap, 1 level, terminal, wall-clock, transport failure.
Invalid if: crash before the first builder pass; ar25 unavailable; trigger_error traced; gate PROBE_ERROR traced (the gate failed open and the run says nothing about uptake).
Result map: gate fires + predicate written -> mechanism confirmed, scale actions; gate fires + no predicate -> uptake null, the affordance is not the bottleneck; gate does not fire -> diagnose the splice in vivo; PROBE_ERROR -> unanswered, fix the gate.
Note: 20 actions cannot answer whether ar25 is solvable. It is sized to the builder's first passes only. Code 738c94f + uncommitted gate.
