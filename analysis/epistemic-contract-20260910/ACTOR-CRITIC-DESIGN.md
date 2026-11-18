# Actor + Epistemic Critic: interface and decision tests

Status: proposed contract, 2026-09-11. Design only; no live solver mutation,
new model calls or experiment launch. The rejected narrow-v2 prompt is evidence
for the requirements below, not an implementation satisfying them.

## 1. Responsibility and the unresolved question

The Actor constructs interpretations, chooses when to build or search a model,
and commits environment actions. The critic seeks counterevidence and unsupported
inferences. The host supplies observations, controls resources, and enforces the
existing action rules. Reality supplies outcomes; neither model's verdict is truth.

The critic's governing question is: which error in this case could change its
action, action horizon, or retained basis for future action? It may also report a
direct factual correction that leaves the immediate action appropriate. Correcting
a claim and blocking an action are separate outcomes.

The distinction between a falsified claim and an unsupported claim is mandatory.
A counterexample under established conditions can refute a deterministic claim.
Insufficient testing cannot. Plausible alternative explanations are conjectures,
not counterexamples merely because the critic generated them.

Our first critic excused S5I5's story because clicking seemed reasonable. The
narrow critic caught its universal quantifier but missed an explicit count
contradiction, and introduced an unsupported absolute position in LS20. No existing
critic has qualified for an authoritative veto. See the preserved
[first screen](../local-probes/20260911-critic-trace-screen-v1/RESULT.md) and
[narrow screen](../local-probes/20260911-critic-narrow-v2-prepared/RESULT.md).

## 2. One review operation

Conceptual interface (JSON records, not new runtime code):

```python
review(case: DecisionCase, evidence: EvidenceView, budget: ReviewBudget) -> ReviewResult
```

`DecisionCase` carries Actor-authored claims. `EvidenceView` is host-owned and
frozen at the same snapshot. `ReviewBudget` is host-owned and shares the game's
existing admission deadline. The critic cannot create its own budget or snapshot.

Two concrete consumers should use this exact interface: saved-prefix evaluation
and the future pre-submission runtime adapter. The evaluation adapter cannot read
events beyond the selected prefix, even if the source files contain later events.

### DecisionCase

```text
identity: host-assigned case_id, revision, snapshot_id
proposal:
  actions: existing submit_actions action representation
  intent: explore | discriminate | execute
  rationale: short reason for acting now
  relies_on: claim IDs used to justify the action or action horizon
  predicted_path: existing representation, optional for one-action exploration
  search_id: existing reference, optional
  review_after: declared action horizon or observable event, when testing a policy
claims: zero or more Claim records
```

```text
Claim:
  id: stable reference in current notes/case
  statement: exact Actor wording
  kind: observation_interpretation | mechanism | objective | plan_assumption
  stance: conjecture | asserted
  scope: tested actions, targets, state conditions, time/level applicability
  support_refs, against_refs: host evidence references, possibly empty
  open_conditions: unresolved prerequisites or ambiguities
  formal_assertion: optional supported measurement predicate
```

Only claims relevant to the decision or changed since the previous review need
inclusion. Reference existing notes and helpers; do not create another world-model
store. The host also attaches the exact current submission rationale, so the critic
can identify an unstated dependency by quotation. `relies_on` is not authoritative
proof of which assumptions the Actor actually uses.

Do not require separate populated objective and mechanism hypothesis lists, two
competing hypotheses, an executable model, or complete evidence before exploration.
An empty claims list plus one legal action and an observation sought is valid.
Missing evidence is unknown, not evidence against a claim. The Actor may directly
observe a special case without having established a general law.

### EvidenceView

The host provides:

- Snapshot identity: session/level epoch, turn, step, grid hash, and relevant helper
  hashes. A claim scoped to one helper version cannot silently use another.
- An initial compact ledger: transition ID, action, before/after shape, changed-cell
  count, changed rows, level-count change, settled-state status and raw frame refs.
- Full raw frames and earlier in-prefix transitions available through read-only,
  bounded queries. Full availability does not mean flooding every prompt with grids.
- Frozen predictions made before actions, measured comparison outcomes, and existing
  typed experiment/search records with their original authority labels.
- Coverage: included time range, omitted frames or regions, truncation, unavailable
  measurements, and exact next queries that can resolve omissions.

Every fact has an origin: host measurement, Actor interpretation, or model-derived
simulation. A count of changed cells is host-measurable; a claimed player identity
or a counter's meaning is not automatically so. Cropped encodings cannot establish
whole-grid absence. Absolute positions cannot be recovered from change lists alone
without the necessary initial state and complete intervening changes.

Host code computes mechanical measurements. The critic may request additional
read-only measurement or run bounded computation in a disposable process. It cannot
write the Actor's helpers, memories or action queue. Offline model tests retain the
model hash and are labeled model-conditional, not world counterexamples. Process
deadlines must actually kill/reap hung computations; an in-process async timeout
does not meet the contract.

Optional formal assertions use a small measurement vocabulary (for example
`changed_cells == 1` on specified transitions). The host can check these directly.
It cannot certify that arbitrary natural-language claims were faithfully translated
into predicates. Keep the original claim alongside any proposed formalization.

### ReviewResult and Critique

Transport completion and epistemic disposition are separate:

```text
ReviewResult:
  status: completed | insufficient_evidence | timed_out | invalid | skipped_budget
  case_id, revision, snapshot_id
  critique: present only when completed
  missing_refs: when insufficient_evidence
  usage: calls, tokens, seconds, offline computation

Critique:
  disposition: ACCEPT | REVISE | PROBE
  findings: at most two Finding records
  probe: zero or one Probe record; required for PROBE

Finding:
  claim_ref_or_exact_quote
  type: contradicted | unsupported | unresolved_attribution | nondiscriminating
  axis: epistemic | instrumental
  evidence_refs
  reason: precise faulty inference
  correction: smallest warranted claim
  consequence: retain_action | shorten_horizon | change_action | reopen_choice
  dependency: why this claim affects that consequence; or why only notes need repair
```

`ACCEPT` means no decision-changing objection established from inspected evidence.
It may include a nonblocking factual correction with `retain_action`; it never
means the whole model is verified. `REVISE` requires a specific correction and its
effect, but need not prescribe a replacement action. `PROBE` requires a live
decision-changing ambiguity and a concrete distinguishing observation. It cannot
be returned just because the goal is unknown.

`insufficient_evidence` is not ACCEPT or PROBE. If a decisive contradiction is
already visible, emit it and describe remaining omissions; do not evade it by
requesting the whole game history.

```text
Probe:
  alternatives: at least two explicit, non-exhaustive conjectures
  observation_or_actions: offline query or legal proposed action sequence
  predictions: alternative -> measurable outcome predicate
  decision_map: outcome -> how the next choice/horizon changes
  otherwise: what remains unresolved on an unexpected/ambiguous outcome
  max_actions, observable_stop, cost_basis, unknown_risks
```

Use existing `set_experiment` measurement expressions and include an otherwise
outcome. The critic proposes; only the Actor may register or execute the probe.
Distinct predictions are necessary for a discriminatory probe, not proof that it
is worth its opportunity cost. Do not invent exact probabilities or numerical
information gain. Known preconditions must be available; a test requiring a known
player position is not executable when that position is unknown.

Findings should fit approximately two short sentences each, with reference IDs
outside the prose. Do not sacrifice evidence to a sentence limit: record full
measurement output in the host log, while the Actor receives a concise summary.

## 3. Decision tests, in order

| Test | Question | Permitted consequence |
| --- | --- | --- |
| Identity and coverage | Is this the right state/model, and does the evidence contain what the claim needs? | Reject stale review; fetch missing evidence or return insufficient_evidence. |
| Literal consistency | Do counts, coordinates, units, before/after values or quantified claims contradict a supplied measurement? | Correct the exact claim. Block or shorten only if a dependent commitment requires it. |
| Scope and negative evidence | Did a local observation become universal? Would the method detect the effect whose absence is asserted? | Narrow scope; keep untested conditions open. |
| Attribution and independence | Does this outcome isolate the proposed cause, or merely match a conjecture or reused data? | Preserve alternative premises; do not automatically rewrite dynamics. |
| Objective connection | Does measured change establish the proposed objective/progress relation? | Keep the proxy provisional; review a continuing policy when its horizon relies on that relation. |
| Decision relevance | If the questioned claim were false, what changes about the next action, horizon or retained policy? | If nothing changes now, allow the action and retain the correction without demanding a new experiment. |
| Discrimination | Do alternatives predict distinguishable observations, and would those observations change a subsequent decision? | Suggest PROBE only with predictions and an outcome-to-decision map. |
| Instrumental value | Is there evidence that a shorter/offline observation answers the same decision question, or that all proposed outcomes lead to the same choice? | Label an instrumental finding with its basis. Do not claim optimality or fabricate risk estimates. |
| Self-audit | Did the critic introduce an unsupported fact, demand certainty, overclaim falsification, or ignore an otherwise branch? | Withdraw or narrow that finding before returning it. |

The first qualified phase exercises epistemic findings. Instrumental findings are
specified for the interface but remain separately scored and advisory until tested.
The narrow critic's failure does not justify immediately expanding its task to
optimize experiments. Parsimony is a tie-breaking heuristic among empirically
adequate conjectures, not a substitute for these tests.

## 4. Invocation and authority

Interception point: an Actor's proposed `submit_actions` batch, before commitment
to the host action queue. A review never races with execution of that same batch.
The critic gets a fresh context using the same Qwen checkpoint and precision.
It receives no prior critic conversation; previously retained findings are evidence
records with dispositions, not unquestionable critic instructions.

Use a deterministic host trigger policy recorded in each experiment card. Available
triggers are a batch longer than one, a new level, frozen-prediction mismatch, a
declared objective commitment, or a fixed number of actions without level progress.
Thresholds and admission priority must be frozen before running. No-progress is a
review trigger, not proof the policy is false. Objective commitment and alleged
irreversibility may depend on Actor declarations; do not claim reliable detection
of silent certainty or unknown game hazards.

Review at most once per decision snapshot, with no recursive critic-of-critic or
automatic re-review of the Actor's response. A new revision at the same state does
not reset this allowance. Host-owned total call/token/wall-clock and query/CPU caps
are mandatory experiment parameters and share the original game's deadline.
Critic cost must appear in total time and action-efficiency reports.

On ACCEPT, the frozen original proposal may proceed through existing submit gates.
On REVISE or PROBE, the Actor receives one concise finding packet and responds:

```text
ActorResponse:
  case_id, critique_id
  disposition: incorporate | contest
  claim_updates: scoped changes to existing notes, if any
  revised_proposal: existing action proposal format
  evidence_refs_and_reason: required to contest
```

The host requires an explicit response record before resubmission, but cannot
certify that a prose response really resolved the criticism. This is acknowledgment
enforcement, not epistemic enforcement. The Actor can contest a mistaken critique.
Only existing host-verifiable failures (invalid actions, stale snapshots, batch
eligibility, etc.) have hard veto authority in the initial design.

A timed-out, invalid, incomplete or budget-skipped review is logged as such and
returns control to the Actor under the ordinary action gates. It never becomes
ACCEPT. Do not retry the critic or silently truncate a plan. On a stale state or
helper hash, discard the review's applicability and rebuild the case through the
normal Actor flow; the unchanged game budget still applies.

The Actor is the sole writer of environment actions and adopted knowledge. Host
records preserve the original claim, critique, response and actual subsequent
outcome. Rejected critiques remain in the audit, not as adopted truths. Evidence
from future levels or actions is never exposed retroactively to a saved-prefix test.

## 5. Acceptance fixtures through the same interface

| Fixture | Expected observable behavior |
| --- | --- |
| S5I5 counts 1,2,1 vs asserted exactly one | Renders a host count ledger; identifies transition 2 as contradiction; correction makes no claim about the objective. |
| S5I5 one further exploratory click | May retain the action while correcting a universal claim; never treats allowing an action as validating that claim. |
| S5I5 continuing counter-threshold policy | Flags unsupported objective dependence; bounded extra observation is possible, but zero completion is not falsification. |
| LS20 sparse differences only | No invented absolute position; request a raw frame if position is necessary. |
| LS20 current raw frame supplied | Any coordinate objection must match that frame; distinguish a bar boundary from its length and displacement. |
| Untested bounce used in a long sequence | Identify the local assumption on which continuation depends; do not claim future failure as already observed. |
| One UP then proposed LEFT exploration | No demand for a complete objective or player model. |
| Same alternatives, same next action | ACCEPT with optional scoped correction; no compulsory probe just to eliminate uncertainty. |
| Different alternatives, identical probe outcomes | Do not return PROBE as a discriminator. |
| Probe requires an unknown player location | Fetch necessary evidence or mark it unavailable; do not describe the probe as executable. |
| No visible change with inadequate observation coverage | Report no observed change in the covered region, not no effect in the game. |
| Depth-limited search finds no path | Do not infer exhaustion, world impossibility, or a false objective. |
| Unsupported critic objection contested with raw evidence | Actor can retain its proposal; critic does not gain truth authority from role. |
| Stale review, malformed JSON, model hang, exhausted budget | No action commit by critic, no synthetic ACCEPT, bounded cleanup and return to Actor. |
| Prefix source contains later winning/failing actions | Queries cannot retrieve later records; future-outcome leakage makes the test invalid. |
| Critic attempts helper/note/queue writes | Read-only boundary prevents the write; no Actor state mutation. |

Mechanical fixtures qualify identity, transport, authority, measurement, lifecycle
and resource boundaries. Model fixtures qualify whether objections and corrections
are actually grounded. Passing the first category cannot substitute for the second.

Use previous six packets as development regressions, with the updated distinction
between claim correctness and immediate action choice. Freeze grading before new
responses. Require the known direct contradiction and no unsupported corrections on
the sparse-position and cautious controls before integration. Then test held-out
claims and real bounded Actor continuations. Compare the critic arm to ordinary
continuation and an equal-inference-budget Actor self-review. Preserve original
Actor prompts, tools and state in that comparison; count the critic's added cost.
Measure actual adopted corrections, false interventions, action horizon changes,
progress and total wall time. Do not grade architecture by critique eloquence.

## 6. Fit to the inspected runtime and source limits

Use the deployed-family `submission-runtime-20260910-v6/chassis/solver_agent.py`
surface: `decision_packet`, `measured_evidence`, `prediction_feedback`, optional
`plan_actions`, `set_experiment` and `submit_actions`. Reuse their evidence and
measurement contracts. Existing frozen-model gates remain the owner of batching
eligibility. New review metadata does not replace `predicted_path` or `search_id`.

Model-building remains an Actor capability. The inspected path already allows
Actor-authored helpers and optional planning; creating a permanent Builder peer is
not required. A separate critic necessarily changes orchestration and the Actor's
inputs, even if its weights, tools and core instructions remain unchanged.

[Tycho's primary report](https://arxiv.org/abs/2607.28287) reports higher transition
accuracy with automatic repair but lower observed game performance than its best
Actor-requested builder policy, in one matched public-set run per policy. This
supports separating model fit from usefulness, not a causal claim that this critic
will improve play. [Rodionov's initial report](https://arxiv.org/abs/2605.05138)
describes verifier-driven executable models; it does not by itself establish the
effectiveness of our proposed intervention. The particular premature-commitment
attribution and remedy quoted in the proposal would need its exact passage before
being treated as independently verified here.
