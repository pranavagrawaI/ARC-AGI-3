# WM2 Practice Guidance and Theory Opportunity Surface

Status: implemented and acceptance-tested; first live contact remains pending.

This document is the durable implementation plan and requirements contract for
WM2. It extends the implemented E0/WM1 lane specified in
[`EXECUTABLE-THEORY-LANE-SPEC.md`](EXECUTABLE-THEORY-LANE-SPEC.md). It does not
replace that specification or authorize P0.

Normative terms **MUST**, **MUST NOT**, **SHOULD**, and **MAY** carry their usual
requirements meanings.

## 1. Governing decision

WM2 adds two independently controllable modules:

1. **WM2-A, Theory Practice Guide**: game-general examples and possibilities
   that teach the model how executable conjectures can enter its work.
2. **WM2-B, Theory Opportunity Surface**: a deterministic view of current
   evidence and artifact conditions from which the model may recognize a useful
   modeling opportunity.

The modules MUST remain separately source-committed, testable, configurable,
and ablatable. The first feasibility run MAY enable both, but its estimand is
then the complete taught interface rather than executable memory alone.

The host does not determine that an executable theory would be useful. It may
only expose reproducible mechanical conditions. The model retains authority to
interpret those conditions, create or defer a commitment, choose its semantics,
and revise or retain it.

```text
host evidence + lane state -> mechanical workbench facts
practice guide             -> possible forms of use
facts + possibilities      -> model judgment
model judgment             -> optional model-authored edit
```

## 2. Evidence motivating WM2

The valid GPT-5.6 Luna capability probe preserved at
`artifacts/wm1-capability-probe-luna-v3/` established that the model could name:

- all three executable-theory methods;
- the partial-predictor contract;
- the isolated evaluator;
- the durable notes and history surfaces;
- the persistent actor and one-action host loop.

It still made no lane call and wrote: “I will avoid inventing a theory before an
observed transition.” The result weakens pure discoverability failure. WM1
documents an interface but does not yet teach a practice or make the artifact's
current job sufficiently legible.

The probe also showed that an explicitly situated use of another capability
worked immediately: Luna wrote the requested durable audit note on its first
attempt. WM2 therefore targets **Interface Debt between specification and
integration**, not a presumed inability to call tools.

## 3. Scope and non-goals

WM2 MUST:

- teach possibilities of use without supplying game semantics;
- teach deferral as a legitimate response;
- make current evidence and theory state continuously legible;
- explain every surfaced condition in ordinary language;
- bind derived facts to recoverable evidence and a frozen schema;
- preserve exactly one host-validated environment action per turn;
- preserve model-only semantic authorship and replacement authority;
- leave prose hypotheses first-class.

WM2 MUST NOT:

- infer objects, roles, mechanics, objectives, or useful actions;
- say that two transitions implement the same rule;
- decide that the model should create, expand, revise, or promote a theory;
- inspect private reasoning or notes to infer an opportunity;
- edit, synthesize, promote, or select executable source;
- maximize prediction coverage or describe greater coverage as better;
- stitch reset, death, terminal, or level boundaries into ordinary dynamics;
- add planning, building, generic shell access, or P0 shadow prediction;
- turn model-facing internal status codes into unexplained vocabulary;
- support a causal or gameplay-improvement claim from a treatment-only run.

## 4. Canonical terms

### Theory Practice Guide

A frozen, game-general model-facing guide that supplies possible uses,
minimal examples, non-use cases, and recovery patterns for executable
conjectures. It is a Contestant Prior, not a game explanation.

### Theory Workbench View

A bounded deterministic projection of authoritative E0 evidence and current
lane state supplied to the model at a decision boundary. It reports artifact
and evidence conditions without deciding their semantic relevance.

### Opportunity Condition

An exactly defined mechanical predicate over the Workbench View, such as
“ordinary evidence exists but no executable commitment is active.” It is not a
claim that using the lane would improve the decision.

### Interpretation Card

The plain-language rendering of one Opportunity Condition. It states what
happened, its mechanical consequence, what it does not imply, possible model
responses including deferral, and the supporting evidence references.

## 5. Authority and modular architecture

| Concern | Authority |
|---|---|
| Public observations and action results | E0 host evidence |
| Ordinary/boundary classification | E0 adapter |
| Active source, version, and evaluation state | Executable Theory Lane |
| Workbench derivation | Deterministic WM2-B machinery |
| Practice examples and wording | Frozen WM2-A artifact |
| Semantic interpretation | Model |
| Source authorship and replacement | Model through broker |
| Environment action validation and execution | Host |

WM2-A MUST have no run evidence or lane-state dependency. WM2-B MUST have no
access to the model conversation, notes content, provider reasoning, or the
WM2-A guide. A context-assembly adapter MAY compose their immutable outputs for
the decision prompt.

Disabling both modules MUST reproduce the WM1 model-facing interface. Disabling
only one MUST not alter the other's output.

## 6. WM2-A: Theory Practice Guide requirements

### 6.1 Required possibilities

The guide MUST explain that an executable conjecture may be used to:

1. preserve a repeatable rule instead of reconstructing it repeatedly;
2. make one narrow hypothesis falsifiable using partial cell claims;
3. compare an expected consequence with later public evidence;
4. localize a contradiction using replay and first divergence;
5. expose when apparent dynamics require additional state;
6. consolidate into a Moore machine only when stateful history is justified;
7. abstain from an action or unclaimed cells when evidence is insufficient;
8. defer executable commitment and retain competing hypotheses in prose.

The guide MUST distinguish “can be executed” from “is supported” and
“is useful for the next decision.”

### 6.2 Required examples

The guide MUST contain bounded synthetic examples for:

- a partial predictor that claims only a small selected region;
- whole-action abstention with `None` and cell abstention with `-1`;
- an operationally valid but empirically contradicted candidate remaining
  active until the model replaces it;
- revision after first-divergence feedback;
- justified promotion from a partial predictor to the Moore contract;
- justified non-use because evidence is insufficient or the rule is not
  decision-relevant.

Examples MUST use synthetic grids and actions unrelated to the current LS20
frame. They MUST demonstrate interface lifecycle rather than imply object roles,
goals, movement rules, coordinates, or palette meanings. Every executable
example MUST pass the same source and containment checks as a model-authored
candidate against a frozen synthetic fixture.

### 6.3 Presentation requirements

The always-visible guide summary MUST be no more than 3,072 UTF-8 bytes. It
MUST use the sequence:

```text
situation -> smallest useful commitment -> feedback meaning -> possible next move
```

The complete tested examples MAY be returned by `read_world_model()`, but the
essential possibilities and at least one complete minimal example MUST be
visible without first calling the lane.

The guide MUST repeatedly preserve optionality using language equivalent to:

> You may defer. An executable commitment is useful only when its prospective
> value justifies its construction and maintenance cost.

It MUST NOT use compliance language such as “you should now create a model.”

## 7. WM2-B: Theory Opportunity Surface requirements

### 7.1 Interface

The system-facing seam is one pure operation:

```python
class TheoryWorkbenchViewBuilder:
    def build(
        self,
        evidence: EvidenceSnapshot,
        theory: TheoryArtifactSnapshot,
    ) -> TheoryWorkbenchView:
        """Return a deterministic, bounded, read-only model-facing projection."""
```

`TheoryArtifactSnapshot` is an immutable projection containing only current
version, contract kind, source/evaluation frontiers, latest operational and
empirical results, coverage, abstention, and first-divergence references. The
builder MUST NOT receive a mutable lane object.

The builder MUST have no writable effect, execute no candidate code, call no
provider, and issue no environment action.

### 7.2 Required ambient facts

Every decision receives a Workbench View, including turn zero. It MUST state:

- whether an executable commitment is active;
- active version and contract kind when present;
- installed evidence frontier and evaluation frontier;
- number of ordinary transitions eligible for replay;
- counts of ordinary transitions by human-facing action;
- number of ordinary transitions not covered by the current evaluation;
- latest operational or empirical result, when one exists;
- prediction coverage and abstention in literal counts, when available;
- whether the latest public event is a reset/terminal/level boundary that must
  not be treated as ordinary dynamics.

The model-visible view MUST NOT expose internal attempt identifiers already
kept model-invisible by E0. Internal segmentation may enforce non-stitching.
For this document, the “current evidence segment” is the internally enforced
ordinary-transition suffix after the latest E0-classified initial reset,
explicit reset, API full reset, death, terminal, or level boundary. A segment
identifier is not exposed to the model.

### 7.3 Recent transition facts

The view MUST include at most the three most recent ordinary transitions from
the current E0 segment. Each fact may contain only:

- stable transition event ID;
- human-facing action;
- changed-cell count;
- half-open changed-cell bounding box, or explicit no-change;
- boundary class;
- whether the active evaluation claimed, partially claimed, or abstained from
  that transition, when known;
- evaluation or divergence reference, when known.

These fields are exact coordinate-aligned comparisons. They MUST NOT contain
object identity, correspondence, displacement, cause, inferred rule, or action
recommendation. Derived facts MUST bind the source grid commitments and
workbench schema version so they are independently reproducible.

### 7.4 Initial Opportunity Condition registry

The first implementation supports only the following frozen predicates:

| Internal code | Exact predicate |
|---|---|
| `no_transition_evidence` | No ordinary transition exists in the current evidence segment |
| `evidence_without_commitment` | At least one ordinary transition exists and no commitment is active |
| `repeated_action_evidence` | One action occurs in at least two ordinary transitions in the current segment |
| `new_evidence_since_evaluation` | The installed evidence snapshot is a strict monotonic descendant of the active version's evaluation snapshot |
| `whole_action_abstention` | The evaluator recorded `predict(...) -> None` for an applicable ordinary transition |
| `partial_cell_claim` | The evaluator recorded coverage strictly between zero and one for an applicable transition |
| `empirical_contradiction` | A claimed prediction differs from public evidence |
| `operational_rejection` | The most recent model edit failed an operational check and did not activate |
| `nonordinary_boundary` | The latest host-classified event is not an ordinary transition |

No adaptive threshold, similarity score, learned classifier, or host-selected
semantic condition is permitted in WM2. Adding a predicate changes the
treatment and requires a specification revision before live contact.

### 7.5 Interpretation Cards

Internal codes MUST remain audit-visible but MUST never appear alone in the
model-facing text. Every active condition renders an Interpretation Card with:

```python
@dataclass(frozen=True)
class InterpretationCard:
    what_happened: str
    mechanical_meaning: str
    does_not_imply: str
    possible_responses: tuple[str, ...]
    evidence_snapshot_id: str
    theory_snapshot_id: str
    evidence_event_ids: tuple[str, ...]
```

For example, whole-action abstention MUST be explained equivalently to:

```text
On transition <id>, predict(...) returned None for UP.
No cells were predicted, so this observation did not support or contradict the
theory. This does not mean UP is unpredictable or that the theory is wrong.
You may retain the gap, extend the theory, or keep the case unresolved in notes.
Nothing here requires an edit.
```

Every card MUST:

- use ordinary language before technical vocabulary;
- distinguish direct fact from inference;
- include a non-implication;
- include deferral among the possible responses;
- cite all evidence necessary to reproduce the condition;
- avoid saying “fix,” “correct,” “best,” or naming an environment action to take.

### 7.6 Bounded rendering

The complete Workbench View MUST be no more than 6,144 UTF-8 bytes. It may show
at most three Interpretation Cards. Fixed priority is:

1. operational rejection or empirical contradiction;
2. nonordinary boundary;
3. abstention or partial coverage;
4. new evidence since evaluation;
5. evidence without commitment or repeated action evidence;
6. no transition evidence.

When more conditions exist, the view MUST report the omitted count and retain
their internal audit records. It MUST NOT silently truncate a sentence, event
ID, or card.

## 8. Failure model

WM2 distinguishes:

1. **Detection error**: the surfaced mechanical condition is false or misses a
   condition inside the frozen registry.
2. **Relevance error**: the condition is true but not useful, or a useful
   modeling situation lies outside the registry.
3. **Interpretation error**: the model misunderstands an accurate condition.
4. **Authority leakage**: wording supplies game semantics or pressures a
   particular edit or action.
5. **Salience failure**: the view is ignored because it is too long, repetitive,
   or unstable.
6. **Premature collapse**: examples or cards induce commitment before competing
   hypotheses deserve resolution.
7. **Coverage gaming**: the model expands claims to improve a displayed metric.
8. **Staleness**: evidence, evaluation, and active version identities do not
   describe one coherent frontier.
9. **Context tax**: guidance and status consume enough context or calls to harm
   the existing actor loop.
10. **Tool-use performativity**: the model edits merely because the interface
    makes editing appear expected.

Mitigations required by this specification include literal counts, source
references, non-implications, explicit deferral, stable layout, bounded size,
frontier identities, no coverage praise, and independent audit of exposure and
uptake.

## 9. Atomicity, monotonicity, and noninterference

- A Workbench View MUST bind one complete E0 snapshot and one complete immutable
  Theory Artifact Snapshot.
- Its identity MUST commit schema version, both input commitments, and complete
  canonical output.
- Rebuilding from the same inputs MUST produce identical logical output.
- A view MUST never become authoritative evidence or mutate either input.
- A rendering failure MUST fail the model turn before `environment.step()`; it
  MUST NOT silently omit the WM2 treatment.
- A lane edit or evaluation completed during a decision MAY affect only a later
  regenerated view, never the already supplied view.
- Neither module may alter the legal-action vocabulary, commitment schema,
  provider-call cap, compaction policy, or one-action host invariant.

## 10. Audit requirements

Every decision audit records:

- guide version and source digest;
- workbench schema and view digest;
- E0 snapshot and theory snapshot commitments;
- rendered byte count;
- active Opportunity Condition codes;
- displayed and omitted card codes;
- evidence IDs cited by each card;
- whether the model read, edited, or evaluated the lane afterward.

The audit MUST NOT infer that exposure caused an edit or that an edit affected
the action.

## 11. Acceptance gates before live contact

### WM2-A gate

- every example passes parser, contract, containment, and frozen-fixture tests;
- examples contain no LS20 frame, coordinate, palette-role, goal, or mechanic;
- the guide includes use, revision, promotion, abstention, and non-use;
- always-visible text remains within 3,072 bytes;
- disabling WM2-A restores byte-identical WM1 static context blocks.

### WM2-B gate

Fixtures MUST cover:

- turn zero with no transitions;
- one ordinary transition and no model;
- repeated same-action evidence without asserting a shared rule;
- an active current model;
- stale evaluation frontier;
- whole-action abstention;
- partial coverage;
- empirical contradiction;
- operational rejection preserving the previous version;
- reset, death, terminal, and level boundaries without stitching;
- more than three active conditions and deterministic priority;
- maximum-size identifiers and render-bound enforcement.

Each fixture MUST verify canonical determinism, evidence references,
plain-language rendering, non-implication, deferral, and absence of forbidden
semantic fields.

### Integration gate

- WM2-A and WM2-B can be enabled independently;
- both disabled reproduce WM1 behavior and source-visible interface;
- scripted model traces can read the guide, understand a card, defer, author a
  partial theory, recover from rejection, and still return one legal action;
- no treatment configuration executes candidate source in the worker;
- existing E0, WM1, containment, notes, history, compaction, and commitment
  tests remain green;
- source commitments name both modules and their frozen content.

## 12. Development plan

1. Freeze this specification and add the new domain terms.
2. Build WM2-A as a standalone immutable guide artifact and renderer.
3. Test every synthetic example through the real evaluator.
4. Define immutable Theory Artifact Snapshot and Workbench View schemas.
5. Build WM2-B as a pure derivation module over frozen fixtures.
6. Build Interpretation Card renderers for the closed condition registry.
7. Add an independent context-assembly adapter with separate feature flags.
8. Add audit commitments and noninterference tests.
9. Run scripted adoption, deferral, contradiction, and boundary traces.
10. Freeze a new experiment card before one live WM2 feasibility run.
11. Inspect uptake, latency, context cost, unnecessary edits, and gameplay
    descriptively before deciding whether P0 or a matched ablation is warranted.

## 13. First live WM2 claim ceiling

The first treatment-only run asks:

> Can a model use a taught executable-theory practice and transparent workbench
> state to make a voluntary, non-vacuous executable commitment after admissible
> transition evidence exists, without breaking the persistent actor loop?

It may establish:

- guide and workbench exposure;
- model comprehension visible in the trace;
- voluntary lane consultation or deferral;
- artifact authorship and retrospective evaluation;
- latency, context, failure, and gameplay observations.

It cannot establish:

- a treatment effect;
- that any cue was relevant rather than merely salient;
- prospective validity without P0;
- causal influence on action selection;
- gameplay benefit.

The live card MUST allocate enough actions for at least two ordinary transitions
unless a terminal or invalid boundary stops the run. Automatic WM edits remain
forbidden. A third model run on unchanged WM1 is not the next experiment.

## 14. Required implementation artifacts

Before live WM2 contact, the worktree contains:

- the frozen Theory Practice Guide and tested example fixtures;
- Workbench View, Theory Artifact Snapshot, Opportunity Condition, and
  Interpretation Card schemas;
- a closed condition registry;
- deterministic workbench fixtures and expected renderings;
- separate WM2-A and WM2-B configuration controls;
- context-assembly and audit integration;
- scripted end-to-end traces for use, deferral, rejection, contradiction, and
  reset boundaries;
- updated provenance and source commitments;
- a <=1 KiB experiment card naming the complete enabled treatment.

Implementation begins from this document. Any change to the condition registry,
example semantics, rendering priority, authority limits, feature modularity, or
live claim ceiling requires a documentation revision before another evidentiary
run.
