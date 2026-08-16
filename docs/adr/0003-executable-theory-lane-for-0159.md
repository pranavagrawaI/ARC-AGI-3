# Executable Theory Lane for 0159

Accepted for implementation in a cumulative experiment derived from the frozen
0159 baseline at `f3d8b3027508167a41a3de4ba5b125fec130575e`.

## Context

0159 already provides one persistent NOOA investigator for an entire game, a
human-facing action vocabulary, immutable host-recorded interaction history,
bounded CodeAct analysis, model-authored durable prose notes, explicit context
compaction, and exactly one host-validated environment action per turn.

Its generated Python is nevertheless used primarily as a decision-local
scratchpad. The CodeAct sandbox persists variables within one decision but is
destroyed at the end of that decision. Durable notes preserve prose, not an
executable commitment that can be replayed, contradicted, or used to predict an
unseen successor.

Tycho demonstrates a related capability: a model-authored executable theory is
stored outside conversational context, evaluated against recorded interaction,
and optionally used for prospective prediction or planning. Importing Tycho's
actor or harness would duplicate responsibilities that 0159 already assigns
correctly. The missing capability is narrower.

Repository terminology calls the durable runnable artifact an **External
Executable Reconstruction** containing provisional **Executable Conjectures**.
When sufficiently useful, it may serve as the current **Working Theory**. This
ADR uses **Executable Theory Lane** for the complete path by which 0159 authors,
stores, evaluates, versions, and measures that artifact. The lane is not a
claim that the artifact is complete, correct, or a recovered source program.

## Decision

Keep the persistent NOOA investigator and add one brokered Executable Theory
Lane beneath it.

The governing rule is:

> Evaluation may contradict Muse's executable commitment, but only Muse may
> replace it. Measurement may observe a decision, but it may not participate in
> that decision.

The system has four planes.

### 1. Evidence plane

The Python 3.14 ARC host remains the only authority for public observations,
actions, terminal state, reset state, animation frames, legality, and execution
order. It records a typed, append-only evidence ledger and produces immutable
evidence snapshots for evaluation.

The evaluator may derive a deterministic replay layout from that ledger, but
the layout is a checked projection rather than a second evidence authority.

### 2. Belief plane

Muse owns two independent durable belief forms:

- prose notes preserve competing hypotheses, uncertainty, rejected
  explanations, and discriminating probes;
- executable source preserves the current runnable commitment worth testing.

One deterministic executable commitment does not replace the broader set of
live prose hypotheses. A contradiction to the executable commitment does not
delete alternatives from the notes.

### 3. Evaluation plane

The worker broker mediates source mutation and version storage. It assembles an
immutable evaluation bundle containing candidate source, pinned machinery, and
a classified evidence snapshot. A fresh isolated evaluator executes the bundle
without access to the persistent worker, provider state, IPC state, canonical
workspace, network, or mutable host evidence.

Evaluation returns structured mechanical results to Muse. It never edits
source, selects a replacement theory, promotes a hypothesis, or chooses an
action.

Operational validity controls whether a candidate becomes the current
commitment. Empirical score does not. An executable but contradicted candidate
becomes current and receives its contradiction; a syntax-invalid, prohibited,
unbounded, or contract-incomplete candidate is preserved as rejected provenance
but does not replace the previous current version. No second promotion ceremony
is required.

### 4. Measurement plane

After the actor has returned a legal decision, but before the host invokes
`environment.step()`, the lane automatically freezes the current commitment and
durably records its prediction for that exact action. The prediction is hidden
from the already completed decision.

When the successor enters the evidence ledger, the lane scores the pending
prediction exactly once. Prediction failure or measurement failure never
suppresses, replaces, or duplicates the accepted environment action.

Authoritative append order, not wall-clock time, proves that the prediction
preceded the environment step.

## Whole-system estimand

The initial treatment is intentionally a bundle. It asks:

> Can Muse adopt an Executable Theory Lane and produce valid prospective
> predictions without breaking the existing persistent actor loop?

The bundle includes focused evidence machinery, a seeded executable contract,
durable source, brokered editing, isolated replay, bounded mechanical feedback,
causal versioning, and automatic shadow prediction. It does not estimate the
effect of durable code independently from the seed, inspection machinery, or
feedback.

If first contact establishes containment, uptake, and prospective validity, a
later matched control/treatment experiment may estimate whether assignment to
the lane changes behavior or outcomes. Historical development runs are not a
control.

## Candidate activation

Every edit produces exactly one of these statuses:

- `rejected_operationally`: preserved, but not current;
- `active_contradicted`: current and contradicted by applicable evidence;
- `active_inconclusive`: current, with insufficient applicable evidence for a
  supported or contradicted judgment;
- `active_supported`: current and supported by every applicable frozen check so
  far.

There is no `correct`, `verified_true`, or equivalent status.

An empirically contradicted candidate remains current because Muse authored it.
Only a subsequent Muse-authored operationally valid edit may replace it.

## Attempt-boundary semantics

Replay segmentation is derived only from public host-observable events:

- the initial environment reset allocates the first replay attempt;
- an explicit `RESET` action allocates a successor attempt;
- API `full_reset` allocates a successor attempt;
- `GAME_OVER` closes an attempt but allocates no successor until a host reset
  actually occurs;
- a visually reset-like ordinary `NOT_FINISHED` transition remains in the same
  attempt;
- level completion closes the current replay attempt; a later public
  observation allocates the next one if play continues.

A boundary record carries `attempt_before` and `attempt_after`; the latter is
absent when no successor exists. These identifiers are evaluator-local replay
segments, not new model-visible observation coordinates or semantic evidence.

## Model-visible surface for WM0

WM0 exposes only brokered reading and editing of the executable commitment:

```python
self.read_world_model()
self.write_world_model(source)
self.edit_world_model_function(name, source)
```

Every successful semantic edit atomically stores a version, evaluates it, and
returns bounded mechanical feedback. There is no separate model-visible
verification step.

WM0 does not add a planner, builder, shell, arbitrary workspace administration,
action batching, host-authored game semantics, forced model uptake, or a second
actor.

## Verifier posture

Verifier output may report syntax and contract failures, initial-render scores,
transition scores, coverage, first divergence, observed and predicted deltas,
and outcome-channel evidence. It must not recommend a repair, name an inferred
object role, choose a representation, propose an action, or mutate a belief.

Pinned Tycho reuse therefore targets structured measurement behavior, not the
full instructional prose of Tycho's feedback probe. Verbatim reused kernels
receive source-hash parity tests. Adapted action and evidence seams receive
behavioral conformance tests and explicit provenance.

## Claim ceiling

A treatment-only first-contact run may establish only:

- Reconstruction Activation or its absence;
- executable artifact creation and revision;
- retrospective replay behavior;
- frozen prospective prediction behavior;
- operational costs and failures;
- gameplay outcome.

Tool traces may support the descriptive claim that the artifact was consulted.
They do not establish that its content changed the decision. Causal decision use
requires a later matched intervention on the artifact's availability or
content. Gameplay benefit requires an appropriate external outcome contrast.

Creating, fitting, or executing source without prospective use remains Model
Theater rather than Reconstruction Activation.

Transition prediction, objective prediction, and gameplay outcome remain
separate result channels. Perfect transition replay with no terminal evidence
and zero completed levels is not world-model or gameplay success.

## Source and provenance

The cumulative experiment starts from the frozen 0159 baseline. Tycho machinery
is reused from pin `f68912a764372ead0a610db2e1c011d41ce5197e` with:

- source commitments for verbatim files and functions;
- behavioral parity fixtures for retained structured semantics;
- explicit adaptations for the human action vocabulary, `x`/`y` click
  coordinates, 0159 evidence records, containment, and feedback rendering;
- a `PROVENANCE.md` before live execution.

## Rejected alternatives

### Replace the NOOA actor with Tycho's actor

Rejected because it duplicates the persistent investigator and makes NOOA a
transport mechanism rather than the agent substrate.

### Let candidate code execute inside the persistent worker

Rejected because model-authored code would inherit provider, IPC, conversation,
and worker authority.

### Let the verifier select the current theory

Rejected because empirical evaluation would silently become belief-selection
authority. Operational invalidity may block execution; contradiction may not
replace Muse's commitment.

### Use prose notes as the only durable belief form

Rejected because prose cannot independently produce a frozen prediction or be
replayed as an executable conjecture.

### Replace prose notes with one deterministic model

Rejected because unresolved competing explanations should survive until public
interaction discriminates among them.

### Bundle planner or builder into first contact

Rejected because either would confound executable-theory uptake with search or
delegation and would expose capabilities not yet qualified in 0159.

### Claim behavioral change from a treatment-only run

Rejected because uptake and prospective validity do not identify a treatment
effect. A frozen prospective control/treatment contrast is required.

## Consequences

- A typed event adapter and containment kernel are prerequisites, not later
  hardening.
- The existing 0159 branch and commit remain an untouched control candidate.
- Candidate source, evaluation, shadow prediction, and prediction score become
  independently committed artifacts.
- The lane adds latency and failure surfaces that must be measured separately
  from model and gameplay failures.
- Absence of artifact uptake is an admissible and informative result.
- Planner and builder remain future cumulative treatments whose qualification
  depends on evidence from WM0.

The normative schemas, lifecycle, atomicity, containment probes, and experiment
requirements are defined in
`experiments/agents/0159-raw-muse-one-turn/EXECUTABLE-THEORY-LANE-SPEC.md`.
