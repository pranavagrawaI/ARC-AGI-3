---
name: interactive-game-solver
description: Learn local rules through measured actions, preserve uncertainty, and execute conditional plans in an unknown interactive grid game.
---

# Interactive Game Solver

## Observe and act
You see a settled 64×64 color grid, with multiple levels to complete. Parse
`grid_rows` with `self.grid_array`; index `[row, column]`. CLICK x y uses column
then row, zero indexed. Use the current `available_actions`. End each turn with
`self.submit_actions(actions, rationale)`; write useful notes before submission.

One exploratory action requires no goal theory, helper, or hypothesis pair.
State the observation sought. Do not fabricate a prediction to justify acting.
All actions cost the game's action budget; take a bounded probe when it can
resolve a question that changes the next commitment. If live alternatives agree
on the next action, keep the explanation unresolved and act.

## Evidence and interpretation
Keep four things distinct in existing notes: observed transitions, hypotheses
about mechanics, hypotheses about the objective, and conditional plans. Each
important claim carries evidence step references, tested conditions, and open
conditions. Preserve those qualifications when summarizing. There is no global
"mechanics solved" state. Update claims dependent on revised evidence while
retaining unrelated supported rules.

An observed displacement does not establish its persistence after reversal.
A transient frame does not establish destruction. A prediction match supports
that measured consequence under tested conditions; it does not prove identity,
cause, a complete model, or objective progress. Mismatch challenges the reading,
measurement, and dynamics jointly. Test before assigning blame to one premise.
An unexpected outcome may introduce a new hypothesis outside your alternatives.

Use `set_experiment` only when real alternatives make distinct typed predictions.
Keep an otherwise outcome and record confounders in the question/updates. A
matched prediction label is not a certified hypothesis. Formal discrimination
is optional; an ordinary exploratory action remains available.

A host-reported level completion demonstrates that the executed sequence worked
from its observed starting conditions. It does not prove every action was
necessary or every explanation correct. Carry that scope to the next level.
A deadline can stop spending on a policy while its explanation remains open.

## Partial executable model and optional search
When useful, write `world_model.py`: `encode(grid)` receives a detached integer
numpy array; `predict(z, action)` receives a detached JSON object. Both return a
nonempty finite JSON dict of at most 16000 characters. Tuples become lists.
Record representation limits. The model assumes deterministic Markov dynamics;
matching past latents cannot establish that your encoder retains every relevant
feature. Read `prediction_feedback()` and test repairs with one action.

Use `plan_actions(target, actions=None, limits=None)` for bounded search through
YOUR model. Example target: `{"path":["x"],"op":"eq","value":3}`. Support is
`eq/lt/le/gt/ge`, or `{"all":[predicates]}` / `{"any":[predicates]}`. This can
be a probe configuration or intermediate target; no winning goal is required.
CLICK games need an explicit candidate action list. Do not hand-write unbounded
search loops. Limits may be lowered; see the method's defaults and ceilings.

`path_found` returns a replayed conditional plan and `predicted_latents`. Pass its
`search_id` to `submit_actions` with a nonempty prefix of `plan`. Author the host
outcome `predicted_path` separately; target satisfaction is not level completion.
`already_satisfied` means the target holds only in the initial encoded state.
`model_exhausted` concerns only that model and supplied actions; revisit premises,
not specifically the objective. `limited` leaves work unresolved. `model_error`
requires inspecting the failed stage. Correct an `invalid_request`; `unavailable`
means containment/model selection could not be provided. Search grants no action
permission and commits no prediction. It is optional for exploration.

## Submission and retention
Batching requires the existing frozen-model qualification, valid predicted_path,
and retrodiction checks. A search does not bypass them. Host execution stops on
boundaries, unchanged grids, or a frozen-prediction mismatch. A new observation
or helper invalidates a search reference; search again before continuing it.

Read existing notes before deciding. Record new evidence and changed claims,
without rewriting unchanged knowledge every turn. At a level boundary compare
`level_terminal` evidence and record the successful conditions, qualifications,
and open questions in `level_N_insights.md`. Before RESET preserve what was
observed and why the next attempt is worth its action cost.
