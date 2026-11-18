# Solver epistemic contract — draft for review

Status: design only. Baseline: repaired submission-runtime-20260910-v5,
packaged by submission-20260910-v8. No runtime change or experiment is authorized
by this document. This contract covers ordinary exploration and any future
latent planner. Search is optional.

## 1. Authority and scope

The host records observations and evaluates declared measurements. The Actor
proposes interpretations, goals, and plans. Search derives conditional results
from an executable conjecture. No layer promotes another layer's result into
stronger evidence automatically.

| Record | Authoritative content | Does not establish |
| --- | --- | --- |
| Observation | Action, settled before/after grids, game state, level count, exact differences where comparable | Object identity, causation, usefulness |
| Experiment result | Which typed predictions match the measured outcome; none may match | Truth of a hypothesis label or proposed knowledge update |
| Model check | Agreement under one frozen encoder/predictor version on identified transitions | Complete state representation, correct objective, validity on untested configurations |
| Search result | A replayable path or a qualified failure within an identified executable model, start state, predicate, and action set | Reachability or impossibility in the real game; action permission |
| Objective result | Host-reported level transition or terminal state | Correctness of every explanation used to reach it |

An observed level transition supports that the executed sequence achieved a
level in its observed starting conditions. It does not prove that every step
was necessary or that the same sequence generalizes.

## 2. Commitment is local

The Actor may use any of three modes. These are descriptions of decisions,
not a requirement to add a mode-selection tool or a second hypothesis store.

- **Measure/explore:** inspect or take one legitimate exploratory action without
  a goal hypothesis, executable model, or fabricated competing explanations.
  State what observation is sought. A new outcome may introduce a new hypothesis.
- **Discriminate:** use set_experiment only when live alternatives predict
  different observations for an action. Keep an otherwise outcome. Record
  unresolved conditions that could make the observation ambiguous.
- **Execute:** act on locally supported rules. Any sequence remains conditional
  on its assumed state, mechanics, and local target. Existing submission gates
  and host interruption rules apply.

A local target may be a probe configuration, alignment, or intermediate state.
It need not be the winning objective. Completion of that target is recorded as
completion of that target, without inventing external progress credit.

Resolve an uncertainty before an action when plausible alternatives change
whether that action is permitted or worth its cost. If those alternatives
agree on the action, keep the uncertainty open and act. When they disagree,
prefer a bounded discriminating probe where one is available. The host cannot
certify the Actor's estimates of usefulness or whether its alternatives cover
all possibilities.

No elapsed-time rule makes a hypothesis false. A budget may end a search or
withdraw permission to continue spending actions on a policy while leaving
the explanation unresolved.

## 3. Evidence updates

Carry forward each decision-relevant claim with its evidence reference, tested
scope, and unresolved conditions in the existing knowledge notes/decision
packet. Use the current set_experiment records for predictions and results.
Do not require restating every claim on every turn.

- A match supports only the measured prediction in the tested circumstances.
- A mismatch challenges the conjunction of state reading, measurement,
  mechanics, and prediction. Attribute it to a particular assumption only
  when evidence separates that explanation from alternatives.
- An unexpected outcome can revise the hypothesis set. It is not forced into
  the closest existing label.
- An apparent impossibility under the current model reopens relevant premises.
  It does not specifically disprove the objective.
- Repeating or summarizing a claim must retain its scope and qualification.
  A claim dependent on revised evidence is marked for rechecking. Unrelated
  supported claims remain available.
- No global mechanics-solved flag. A local rule may be sufficient for action
  while pushing, reversibility, collisions, and later levels remain unknown.

Example: after one bar displaces another, retain 'growing bar 9 displaced bar 7
up three rows in transition T; persistence after shrinking is untested.' Do
not promote this to 'pushes are permanent.' If a plan requires permanence,
probe reversal before spending actions that depend on it.

These are reasoning and retention requirements. Structural validation can
preserve references and field distinctions; it cannot establish semantic
correctness merely by accepting a schema.

## 4. Optional planner interface

Proposed surface:

    plan_actions(target, actions=None, limits=None) -> SearchResult

The planner submits no actions and grants no batch eligibility. It may write
host-owned search diagnostics, but does not edit helpers, notes, experiments,
or model qualification records.

### Input

- `target`: a serializable predicate over JSON latent fields. Start with typed
  equality/order comparisons and bounded all/any composition. A missing field
  is an evaluation error, not false. Call it a target, not a winning goal.
- `actions`: a finite list of legal, canonical action strings. Default to the
  current non-CLICK actions. For CLICK, require explicit candidate coordinates.
  An empty action set returns invalid_request and explains how to supply one;
  it must not imply impossibility in the game.
- `limits`: host-capped wall time, callback time, generated states, frontier,
  depth, and action count. Defaults are deployment configuration, not model
  promises. Reject invalid values and requests above host ceilings.

Example target: {"path":["x"], "op":"eq", "value":3}. This specifies an
exact condition within the current encoder. The Actor remains responsible for
why reaching it is useful.

Use a declarative target instead of an arbitrary closure so the complete
search request is serializable, replayable, and executable in a disposable
process. Complex derived features belong in the versioned encoder. A callback
API is not part of this first contract.

### Snapshot and execution

Capture current turn, step, level, grid digest, exact helper source/hash,
canonical initial latent, target, actions, and effective limits. A shared model
selector and validator must serve prediction preparation and search. Keep
loading/top-level execution, encode, predict, normalization, target evaluation,
and result replay inside the overall bounded operation.

Use the same finite JSON representation and latent-size limits as prediction
preparation. Detach each callback input and output. Search assumes deterministic
Markov dynamics over this latent. Report this assumption; finite checks cannot
prove its completeness or determinism.

Hard containment is required. If it cannot be established, return unavailable
without running unbounded callbacks. The existing Actor-wide 60s watchdog is
not the planner's normal deadline implementation. The parent must retire and
reap the search process on timeout before returning its result. Bound resource
allocation independently of merely counting dequeued nodes.

### Results

Every result includes a status, elapsed time, expanded/generated counts,
bounded diagnostics with total error counts, and the snapshot fields captured
before it stopped. Unavailable fields are explicitly null; failure before
encoding must not fabricate an initial latent. Effective limits are included
once validated. Only model_exhausted certifies complete traversal under the
stated model assumptions. Result variants determine additional fields:

| Status | Exact meaning | Additional payload |
| --- | --- | --- |
| already_satisfied | Target holds in the encoded starting state | Empty action/state path |
| path_found | A returned path replays from the frozen start and satisfies the target under that model | Actions and detached predicted latents, replay status |
| model_exhausted | All reachable states under the supplied actions were explored with no callback failures or unexpanded cutoff frontier | Complete traversal diagnostics |
| limited | Search stopped with unresolved work | Reasons: wall time, callback time, depth, generated states, frontier |
| model_error | Loading, encoding, prediction, normalization, or replay failed | Stage and error; no impossibility conclusion |
| invalid_request | Invalid predicate/action/limits, or target references unavailable fields | Correctable input error |
| unavailable | Required execution/containment capability is absent | Capability reason |

For the first version, fail fast on model callback error. Skipping faulty edges
adds a partial-graph contract we do not need. A found path requires replay in a
fresh model instance from the frozen start. Disagreement is model_error.
Replay agreement remains evidence about the executable conjecture only.

Do not promise shortestness in version one. A returned path is sufficient;
shortestness adds a claim that depends on complete edge evaluation and model
purity. Budget expiry during replay returns limited without an actionable path.

Depth cutoff is limited if states at that boundary remain unexpanded. Even
if those states would prove to be leaves, the planner has not established it.
Exhaustion and paths apply only to the supplied action set and representation.

Do not synthesize predicted_path.expect from latent equality or target
satisfaction. A changed encoder value does not establish a particular host
outcome. Return model-predicted latents under a distinct field.

## 5. Submission remains a separate authority

Ordinary single-action exploration remains available through submit_actions.
The planner result does not become a required credential for ordinary actions.

Proposed join: add optional keyword-only search_id=None to the existing
submit_actions interface. A path_found result includes an opaque search_id
referencing its host-owned frozen result. When supplied, submission requires:

- the stored result has status path_found;
- current turn, step, level, grid digest, and helper hash match its snapshot;
- submitted actions are a nonempty prefix of the stored action sequence;
- all ordinary submission gates also pass.

Reject stale or mismatched references before committing anything. Do not
silently rebind or rebase a result. A shorter prefix supports cautious execution;
continuing after a new observation requires a fresh search reference. The
planner's discovery text tells the Actor to pass search_id when using its plan.

Calls without search_id retain the ordinary submission contract. The host
cannot infer where manually supplied actions came from; it must not claim
provenance validation for these calls. Host-owned search records are execution
diagnostics, not an additional hypothesis or knowledge store.

Preserve existing model-version qualification, predicted_path validation,
retrodiction policy, legality checks, and stop-on-boundary/no-impact/mismatch.
Do not describe these existing checks as proving objective correctness. A
successful search never bypasses them. Search does not need a separate rule
for choosing the world-model file or a duplicate batch-eligibility rule.

## 6. Qualification before implementation is accepted

Deterministic contract checks:

1. One exploratory action is possible with no objective or helper model.
2. A local-target path can be found without a level-completion claim.
3. Depth/resource cutoff never returns model_exhausted.
4. Missing dynamics or malformed latents never return model_exhausted.
5. Mutating predict cannot contaminate siblings or saved results; paths replay.
6. Every callback/load stage is contained and timed-out children are reaped.
7. Narrow node limits also bound frontier allocation and branching work.
8. A found path cannot bypass batch gates; an independently qualified path can
   pass when ordinary host requirements are satisfied.
9. Changed state/model identity rejects use of a stale search result.
10. An observed prediction match does not label the objective or mechanics true.

Behavioral evaluation, separately from these structural checks: the Actor
preserves uncertainty, proposes executable next actions, and revises the right
premise in the saved pushing, reversal, transient-frame, and false-impossibility
cases. Schema compliance alone is not a behavioral pass.

## 7. Recommended order

First audit and revise the existing prompts/experiment requirements around
sections 1–3: exploration must not require a manufactured goal or a forced
hypothesis pair. Test those behaviors on bounded cases. Sections 4–5 specify
what an optional planner must obey if added; they do not make building one a
prerequisite for fixing epistemic management.

## Implementation binding

Implemented in submission-runtime-20260910-v6 / submission-20260910-v9.
Hard containment uses a disposable POSIX main-thread fork and fails closed
elsewhere. Host ceilings: 5s wall, .25s callback, 5000 generated successors,
1000 queued states, depth 12, 64 action candidates. Limits may only be lowered.
A parent-killed child has unknown traversal counts (null), not zero work.
Frozen source is retained in search_results.jsonl; the Actor receives its hash
and result without redundant source text. At most 16 recent plan references
are retained in host memory; expired references are rejected. The shared
selector is select_model; compilation/validation remains PredictionFeedback._module.
This preserves a single owner for each rule while keeping executable code
outside the parent planner process.

The reasoning/retention rules are Actor instructions and typed evidence
authority labels. No schema validator can certify that the Actor has preserved
the correct uncertainty. Live behavior is a separate evaluation obligation.
