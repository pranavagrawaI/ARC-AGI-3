# Luna-supervised Gemma prototype

Question: can Gemma act usefully when Luna chooses the immediate problem but is
forbidden to choose the environment action?

## Information boundary

Both models receive the same `RULES.md`. Luna receives less context than Gemma:
the current public observation, the same annotated frame Gemma sees, available
actions, at most six recent transitions, the accepted eight-field state block,
and its previous assignment. It does not receive Gemma's private model
reasoning, arbitrary workspace files, hidden environment state, source code,
future observations, or a winning trajectory.

Luna returns exactly:

```text
LOCAL_GOAL
RELEVANT_EVIDENCE       # one to three items
SUCCESS_TEST            # observable after one action
IF_TEST_FAILS
```

An action-bearing or extra field is refused. Luna is instructed not to smuggle
an action, action number, route, or multi-step plan into the prose fields, but
the host cannot determine that semantically without becoming another game
interpreter. Prose-level compliance therefore requires explicit human
annotation after the run; it is not falsely advertised as host-enforced.
`score_assignments.py` refuses to produce a rate until every successful Luna
generation has a boolean `withheld_action` annotation.
Gemma receives the accepted assignment in its current observation, so it
remains present throughout every deliberation round and every repair crossing
for that action point. `LOCAL_GOAL` and `SUCCESS_TEST` are binding for the
turn. Luna's evidence and failure advice remain defeasible: current admissible
evidence wins, but Gemma must name the contradiction rather than silently pick
a different objective.

Gemma binds every response to the assignment hash and marks it `FOLLOW` or
`CONTRADICT`. `FOLLOW` includes an immediate prediction. `CONTRADICT` includes
the conflicting admissible evidence and releases no environment action; Luna
may revise once, using the existing single repair opportunity. A second
contradiction ends the action point without acting.

The host checks only structure and continuity. It does not decide whether
Luna's goal or evidence is correct, whether the success test occurred, or which
action Gemma should take.

## Status

The local vertical slice is implemented under
`host/supervision.py`, wired as `--knowledge-condition rules_supervised`, and
covered by `tests/test_supervision.py`. The Luna request uses the Responses API,
pins the OpenAI provider, attaches the same annotated PNG, requests strict JSON,
and records credential-free request/frame commitments plus the generation ID.

No live supervised run has been executed. A live request still passes through
the existing digest-bound approval flow and requires its own explicit
confirmation.
