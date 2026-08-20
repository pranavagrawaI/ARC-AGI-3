# tycho-nooa-goal-gate-ar25-20260819T121859Z

Supersedes tycho-nooa-goal-gate-ar25-20260819T120904Z, which was INVALID by its
own "crash before the first builder pass" clause: 0 actions, stop_reason
exception. Two defects it exposed are fixed here — the gate's module-wide
literal check cleared the pinned seed model (subgoals() spends
"level_complete" while outcome() is the constant "ongoing"), and an exhausted
25-step actor budget ended the episode instead of demanding the action.

Question: In vivo, does gate 0 fire on the seed outcome(), and does the builder
answer it by writing a candidate level_complete predicate?
Posture: verification of a mechanism, not of gameplay.
Contrast: 175557Z (same game, same seed, no gate; 6 searches, 0 predicates, 0
levels).
Prediction: the first builder pass prints PLANNER_STATUS:
not_run_goal_unsatisfiable; zero no_plan_within_budget / planner_timeout while
outcome() is the seed constant; a later pass writes "level_complete" into
outcome() and the gate lifts to PLANNER_CONTRACT.
Measure: builder-tool-trace.jsonl counts of not_run_goal_unsatisfiable,
PLANNER_CONTRACT, no_plan_within_budget, planner_timeout; grep "level_complete"
in workspace/ar25/world_model.py's outcome() and the turn it first appears.
Secondary: how many turns end at the forced take_action demand rather than a
volunteered one — 25 steps may simply be too few for this model.
Cap: ar25 seed 0; 20 actions; stop-after-levels 1; one live model; 45 min; no
retries.
Stop: action cap, 1 level, terminal, wall-clock, transport failure.
Invalid if: crash before the first builder pass; ar25 unavailable;
trigger_error traced; gate PROBE_ERROR traced (the gate failed open and the run
says nothing about uptake).
Result map: gate fires + predicate written -> mechanism confirmed, scale
actions; gate fires + no predicate -> uptake null, the affordance is not the
bottleneck; gate does not fire -> diagnose the splice in vivo; PROBE_ERROR ->
unanswered, fix the gate.
Note: 20 actions cannot answer whether ar25 is solvable. It is sized to the
builder's first passes only. Code 738c94f + uncommitted gate and demand fixes.
