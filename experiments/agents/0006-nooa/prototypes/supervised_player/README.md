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

## Verifier reconciliation — supervised-probe-rules_supervised-l1-3-v5

The first clean supervised run reached its 3-action budget with three
successful Luna calls and no agent errors, and its frozen `summary.json`
records four failed acceptance checks: `linked_process_records`,
`linked_schema_versions`, `successor_context_packet_linked`, and
`claim_continuity_linked_across_actions`.

Offline replay located three acceptor defects rather than malformed run
evidence. The verifier in the tree was byte-identical to the one in the run's
`source-snapshot`, and reproduced the recorded verdict exactly before the
repair.

1. `supervisor_assignment`, `supervision_response`, and
   `commitment_controller_state` were absent from the closed process-kind
   registry, so the spine kind sequence never matched. That zeroed the
   collected context packets, which is why one cause produced three failures.
2. The supervised model observation carries a `supervision` key that the closed
   observation key set forbade under set equality.
3. The controller condition replaces the published
   `reason_authoring_schema.working_state` with the eight-field block, while
   the verifier regenerated the base packet without the controller.

The repaired verifier accepts all twelve checks on the local immutable v5
artifact, and the unsupervised 44-action treatment artifact stays fully
accepted. As with the earlier reconciliation in `COMPARISON.md`, the original
`summary.json` is unchanged; this record states the post-run correction instead
of rewriting frozen output.

The repair strengthens rather than widens the boundary. The supervised records
are bound at fixed spine positions, so the assignment must precede the context
Gemma was given and the receipt must precede the validated decision; the
handoff must reconstruct to the recorded assignment; the controller block is
replayed from the model's own commitments and the recorded state and metrics
are checked against that replay. `tests/test_live_probe.py::SupervisedChainTests`
pins each rejection.

One known gap: a `CONTRADICT` receipt on an admitted action point is rejected
rather than modelled. `CONTRADICT` releases no action, so it cannot appear on
an admitted spine, but the revision path it belongs to has never been observed
live. It needs a real recorded shape before the verifier can admit one.
