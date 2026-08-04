# Luna-supervised Gemma prototype

Question: can Gemma act usefully when Luna chooses the immediate problem but is
forbidden to choose the environment action? This is a human-designed dual-model
supervision condition, not an autonomous single-model contestant architecture.

## Information boundary

Both models receive the same `RULES.md`. Luna receives less context than Gemma:
the current public observation, the same annotated frame Gemma sees, available
actions, at most six recent transitions, its previous assignment, and a reduced
five-field state: the verbatim accepted `PLAYER_BINDING`, approximate position,
local objective, last action, and last outcome. The view records the controller's
procedural boundary: `PLAYER_BINDING` is opaque persistent text, while
`CURRENT_APPROX_POSITION` can change independently. The host neither interprets
nor normalizes either value. Luna does not receive Gemma's route, next action,
private model reasoning, arbitrary workspace files, hidden environment state,
source code, future observations, or a winning trajectory.

Luna returns exactly:

```text
LOCAL_GOAL
RELEVANT_EVIDENCE       # one to three items
SUCCESS_TEST            # observable one-action completion or progress
IF_TEST_FAILS
```

An action-bearing or extra field is refused. Luna is instructed not to smuggle
an action, action number, route, cardinal or relative movement direction,
adjacent target cell, or multi-step plan into the prose fields, including prose
that uniquely selects one legal action under `RULES.md`. The host cannot
determine that semantically without becoming another game interpreter.
Prose-level compliance therefore requires explicit human annotation after the
run; it is not falsely advertised as host-enforced.
`score_assignments.py` refuses to produce a rate until every successful Luna
generation has a boolean `withheld_action` annotation.
Gemma receives the accepted assignment in its current observation, so it
remains present throughout every deliberation round and every repair crossing
for that action point. `LOCAL_GOAL` and `SUCCESS_TEST` are binding for the
turn. Luna's evidence and failure advice remain defeasible: current admissible
evidence wins, but Gemma must name the contradiction rather than silently pick
a different objective. The same procedural continuity boundary is attached to
Gemma's handoff: it can update `CURRENT_APPROX_POSITION` without rewriting the
opaque `PLAYER_BINDING` merely to track ordinary movement.

If `LOCAL_GOAL` cannot finish in one action, Luna must make `SUCCESS_TEST`
measure one-step progress rather than completion.

Gemma binds every response to a short opaque token that the host issues once per
assignment within the run. It returns exactly one plain-text line:

```text
FOLLOW <token>: <immediate observable prediction>
CONTRADICT <token>: <precise admissible evidence>
```

The process log maps each unique run-local token to the full assignment hash;
the full hash is not placed in Gemma's handoff.
`CONTRADICT` releases no environment action; Luna may revise once, using the
existing single repair opportunity. A second contradiction ends the action
point without acting.

The host checks only structure and continuity. It does not decide whether
Luna's goal or evidence is correct, whether the success test occurred, or which
action Gemma should take.

## Status

The local vertical slice is implemented under
`host/supervision.py`, wired as `--knowledge-condition rules_supervised`, and
covered by `tests/test_supervision.py`. The Luna request uses the Responses API,
pins the OpenAI provider, attaches the same annotated PNG, requests strict JSON,
and records credential-free request/frame commitments plus the generation ID.

Each live supervised run passes through the existing digest-bound approval
flow and requires its own explicit confirmation.
