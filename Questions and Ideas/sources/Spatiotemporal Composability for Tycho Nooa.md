# Spatiotemporal Composability for Tycho Nooa

**Status:** Research synthesis and design-question ledger  
**Snapshot:** 2026-08-21  
**Source paper:** Yifan Shi, Wei Zhang, and Tianyi Cui, *A Programming
Paradigm for Spatiotemporal Composability* (2026-08-13), 88 pages.  
**Source path:**
`/mnt/c/Users/hipra/Downloads/a programming paradigm for spatiotemporal composability.pdf`  
**Target:** `/home/hipra/projects/arc-agi-3-tycho-nooa-refactor`  

This document captures the complete paper-to-Tycho-Nooa investigation:
observations, interpretations, proposals, latent failure hypotheses, objections,
open questions, candidate mechanisms, examples, and possible experiments. It is
not an implementation plan and does not imply that every idea should be built.

---

## 1. Governing judgment

The paper's greatest potential contribution is not a direct repair for the
current goal-acquisition or builder failures. It supplies concepts for a deeper
architectural problem:

> Tycho Nooa has persistent memory, but it does not yet have a rigorous model of
> contextual validity across changing evidence, goals, models, tools, and
> policies.

As the system acquires more capabilities, its principal latent risk is a
plausible decision composed from individually reasonable but mutually
incompatible artifacts:

```text
goal inferred under evidence E12
+ world model revised after E15
+ planner key designed for world-model revision WM4
+ generated tool installed under policy P2
+ actor reasoning from compacted memory M7
= action that no single coherent agent state would have produced
```

The paper suggests treating validity as a runtime relationship rather than a
convention. A consumer should know the exact providers and revisions against
which it became active. When that view changes, it must remain active, reload,
unload, or have its result quarantined according to an explicit lifecycle.

This is an architectural inference from the paper and current Tycho Nooa, not a
demonstrated ARC performance result.

---

## 2. Epistemic status of the claims

### 2.1 Observations from the paper

1. A component is described by required dependencies, provided capabilities,
   and effects.
2. Reactive coeffects expose which contextual dependencies a component reads.
3. A component's committed view records the exact providers against which it is
   active.
4. The runtime computes a target view and reconciles component lifecycles toward
   it.
5. The lifecycle distinguishes inactive, reloading, active, and unloading
   states, including changes that occur while asynchronous work is in flight.
6. Dependents withdraw before the provider on which they depend is removed.
7. Isolation realms allow the same dependency key to resolve differently in
   different contexts.
8. Interception can apply contextual behavior at dependency access.
9. Revertible effects return inverses that are composed in reverse order.
10. Transactional hot reload preserves the prior state if replacement fails.
11. The formal confluence result relies on conditions including independence,
    acyclic dependencies, total provisions, and an idealized failure boundary.
12. The paper presents self-evolving agent harnesses as an application area; it
    does not demonstrate improved agent performance in a controlled ARC study.
13. External emissions are not made reversible merely by calling them effects.
    Reversion applies only where the relevant state is internal, exclusive, and
    reifiable, or where an honest compensating operation exists.

### 2.2 Current repository observations most relevant to this synthesis

1. Tycho Nooa has a persistent actor and durable workspace, while builders are
   created for trigger or orchestrator requests.
2. Actor and builder use shared service and workspace surfaces.
3. Candidate goals can be recorded durably, but merely recording a candidate
   does not automatically create an executable discriminating obligation.
4. Builder output currently has several logically separate capabilities:
   decoding, action enumeration, transition prediction, rendering, outcome
   interpretation, and planner-state representation.
5. A generated world model may support some of these capabilities while lacking
   others. One feasibility run produced useful dynamics while omitting the
   `actions()` surface needed downstream.
6. Planner validity depends not only on having a state key but on that key being
   congruent with planner-visible actions, transitions, and outcome semantics.
7. Existing shadows, checkpoints, branch replay, and evidence ledgers already
   implement narrow forms of isolation and provenance. A new generic framework
   must earn its complexity over these mechanisms.

### 2.3 Inferences

1. Increasing persistence, asynchronous work, and self-generated machinery will
   make implicit compatibility progressively more dangerous.
2. The first likely failures will be silent stale-context failures, not obvious
   crashes.
3. A model revision can invalidate not just predictions but also plans, cached
   values, generated macros, explanations, and summaries derived from it.
4. Multiple executable hypotheses will be difficult to sustain if each must
   overwrite a shared canonical workspace.
5. Without declared dependencies, invalidation will tend toward one of two bad
   extremes: restart everything or unknowingly retain stale consumers.
6. A declarative target view could make restart, recovery, compaction, and
   replacement converge on one coherent state rather than reproduce historical
   accident.

### 2.4 Assumptions requiring tests

1. Tycho Nooa will acquire enough concurrent or dynamically replaceable
   components for lifecycle machinery to repay its complexity.
2. Cross-revision contamination occurs often enough to affect decisions.
3. Exact provider identity is more useful than content-addressed artifacts plus
   explicit provenance alone.
4. Executable rival hypotheses improve discrimination rather than multiplying
   expensive speculative work.
5. The agent can use structured obligations and capability status without
   converting them into ritualized prose.

### 2.5 Preferences, not findings

1. Prefer the smallest local mechanism that tests a paper principle; do not port
   Cordis wholesale.
2. Preserve host-authoritative evidence separately from model-authored belief.
3. Preserve rejected and stale work as inspectable evidence when cheap, but do
   not allow it to remain canonically active.
4. Prefer explicit incompatibility over permissive fallback.
5. Prefer deletion when existing shadows, checkpoints, or ordinary transactions
   already solve the demonstrated need.

---

## 3. The central concept: an epistemic context

An **epistemic context** is the exact set of facts and capabilities under which
a result is valid. A candidate representation is:

```text
EpistemicEpoch 17
├── evidence head: E42
├── active goal: G6
├── accepted world model: WM9
├── action vocabulary: A3
├── outcome interpretation: O4
├── planner equivalence relation: K2
├── policy: P5
├── compacted memory: M7
└── installed generated tools: T11, T14
```

The important property is not the class name. It is the answer to:

> Against exactly what evidence, interpretations, providers, and policies was
> this artifact computed?

A plan should not merely say `generated_by=planner`. It should say, directly or
through a content-addressed lineage:

```text
Plan P9 committed against:
    evidence E42
    goal G6
    transition model WM9
    action set A3
    outcome predicate O4
    planner key K2
```

If `WM9` is replaced by `WM10`, the system need not destroy `P9`. It must stop
treating `P9` as active evidence for action unless compatibility is re-established.

---

## 4. Ranked latent defects and candidate remedies

The ranking below deliberately looks beyond the current immediate defects. It
orders mechanisms by possible long-run leverage, discounted by evidence and
implementation cost.

### L1. No first-class contextual validity

**Promise:** Highest  
**Status:** Strong architectural concern; behavioral incidence unmeasured.

#### Failure mode

Artifacts are individually versioned or preserved, but the system lacks one
enforceable statement that a goal, model, planner, memory, and tool set belong
together.

#### Paper mechanism

Committed views record exact provider identity. Target views describe what the
consumer should now use. A difference forces a lifecycle transition rather than
silent continuation.

#### Example

```text
P8 begins search against G6 + WM9 + K2.
Observation E43 invalidates WM9.
WM10 is published.

Unsafe:
    P8 reads some later result from WM10 and publishes a hybrid plan.

Safer:
    P8 retains committed view {G6, WM9, K2}.
    Its target view becomes {G6, WM10, K3}.
    P8's result may land as stale evidence but cannot become the active plan.
    P9 starts against the new coherent view.
```

#### Smallest candidate mechanism

Add lineage to consequential artifacts before adding a general runtime:

```python
ArtifactRevision(
    key="validated_plan",
    revision="P9",
    producer="planner-2",
    evidence_commitment="E42",
    dependency_revisions={
        "goal": "G6",
        "world_model": "WM9",
        "planner_key": "K2",
    },
    status="candidate",
    payload_ref="...",
)
```

#### Discriminator

Reconstruct several complete decisions. For every consequential artifact, test
whether its exact validity context can be recovered and whether any action used
incompatible revisions. If reconstruction is already complete and reliable,
do not introduce an epoch abstraction yet.

---

### L2. No general protocol for context changes during in-flight cognition

**Promise:** Very high once overlapping work is real  
**Status:** Structural exposure; one builder deadline path already motivates
transactional isolation, but full concurrency failure is not established.

#### Failure mode

The actor reasons, builders revise models, verification runs, and planners search
over different timescales. The context may change between start and completion.

#### Paper mechanism

Lifecycle states plus diverted landings:

```text
Inactive -> Reloading -> Active -> Unloading -> Inactive
```

Work already in flight is allowed to land where abrupt cancellation would be
unsafe, but a result whose target view turned is not installed as canonical.

#### Example

```text
Search S12 starts under goal G4.
The actor obtains evidence rejecting G4.
S12 finishes with a legal route.

The route remains useful forensic evidence about S12 and the old theory,
but it is not an actionable plan for the current goal G5.
```

#### Smallest candidate mechanism

Attach a context commitment to every async request and check it at publication:

```text
start(request, committed_revision_set)
finish(result)
if committed_revision_set != current_required_revision_set:
    store_as_stale(result)
else:
    publish(result)
```

This publication gate is much smaller than a complete component lifecycle.

#### Questions

- Which operations can be safely cancelled?
- Which must land before cleanup?
- Which stale results remain evidentially useful?
- What is the safe boundary for an actor action already accepted by the host?

---

### L3. Rival hypotheses cannot coexist as genuinely separate executable worlds

**Promise:** High for scientific reasoning  
**Status:** Conceptually strong; must beat existing shadows and workspace copies.

#### Failure mode

Alternative hypotheses may coexist in prose but not in executable artifacts.
Building one model overwrites or contaminates another, causing premature
canonicalization.

#### Paper mechanism

Isolation realms allow one logical key to resolve to different providers in
different contexts.

#### Example

```text
Shared immutable evidence:
    E1 red moved right
    E2 contact with blue changed both objects
    E3 score did not change

Realm H1: blue is the terminal target
    goal -> G-blue
    model -> WM-blue-target
    planner -> P-blue-target

Realm H2: blue is a teleporter
    goal -> G-teleport
    model -> WM-teleporter
    planner -> P-teleporter

Realm H3: shape, not color, determines the rule
    goal -> G-shape
    model -> WM-shape
    planner -> P-shape
```

The realms share observations but cannot overwrite each other's derived models.
A discriminator chooses an action for which their predictions differ.

#### Smallest candidate mechanism

Use immutable evidence plus a realm identifier on derived artifacts:

```text
realm_id: canonical | h-blue-target | h-teleporter | h-shape
```

Only the canonical realm may publish an action. Promotion must cite the
discriminating evidence.

#### Danger

Realm creation can become speculative branching without selection pressure.
Require every realm to name:

1. a distinct prediction;
2. a legal discriminator;
3. a survival or retirement rule;
4. a resource cap.

---

### L4. Capability names are too coarse to establish substitution

**Promise:** High  
**Status:** Directly motivated by partial generated models.

#### Failure mode

`world_model.py` appears to mean one complete capability, but actually bundles:

```text
state decoding
legal action enumeration
transition prediction
rendering
terminal recognition
goal evaluation
planner-state equivalence
```

Two artifacts with the same filename need not be valid substitutes.

#### Paper mechanism

Components declare fine-grained requirements and provisions.

#### Candidate manifest

```text
WM9 provides:
    StateDecoder@2               verified
    TransitionModel@4            verified on recorded path
    Renderer@1                   verified
    ActionEnumerator@2           missing
    OutcomePredicate@3           unverified
    PlannerKey@1                 contradicted by UNDO
```

Derived readiness becomes explicit:

```text
simulation_ready = true
goal_experiment_ready = false
planning_ready = false
```

#### Important distinction

`provides` and `verified` are different predicates. Generated code can claim a
surface without having earned trust in it.

#### Smallest candidate mechanism

Create a manifest emitted by verification rather than trusting model-authored
declarations. Consumers fail closed with a named missing or contradicted
capability.

---

### L5. Dependency changes are not treated as semantic events

**Promise:** Medium-high once dependency count grows  
**Status:** Latent.

#### Failure mode

The important event may not be a new file or message. It may be:

```text
goal G4 became unsupported
action set A2 became incomplete
planner key K3 was contradicted
memory M8 omitted evidence required by G4
model assumption U7 was invalidated
```

Unless consumers declare these dependencies, the system cannot know what should
be reconsidered.

#### Paper mechanism

Reactive coeffects expose contextual reads. Context changes classify a consumer
as still active, newly activatable, deactivating, or requiring reload.

#### Example

```text
Planner requires:
    active_goal_revision
    transition_model_revision
    action_set_revision
    outcome_revision
    planner_key_revision

Renderer R3 -> R4:
    planner remains active unless rendering is part of its observation.

Goal G4 -> G5:
    planner target changes; its plan must reload.
```

#### Design warning

Do not turn every variable read into a dependency edge. Record only contextual
facts whose change can invalidate the consumer's result or authority.

---

### L6. No declarative target architecture

**Promise:** Medium-high for recovery and continuous evolution  
**Status:** Latent; current event flow may remain manageable.

#### Failure mode

Imperative event handlers can make the final system depend on event order:

```text
evidence -> builder -> goal change
```

may leave a different active configuration from:

```text
goal change -> evidence -> builder
```

even when the final evidence and accepted goal are equal.

#### Paper mechanism

Compute the desired target view from current context, compare it with committed
views, and reconcile until the system is quiescent.

#### Tycho Nooa target example

```text
Given:
    evidence head E42
    canonical goal G6
    accepted model WM9

Desired:
    discriminator compatible with G6
    simulator compatible with WM9
    planner compatible with G6 + WM9 + K2
    actor allowed to publish only from the canonical realm
```

#### Desired property

Two histories that reach the same authoritative context should converge to the
same active architecture, apart from explicitly preserved evidence and caches.

#### Counter-pressure

History can be semantically meaningful in an agent. The system must not erase
learning just to obtain configuration confluence. Separate:

- authoritative accumulated evidence, where history matters;
- active derived configuration, which should be reproducible from that evidence.

---

### L7. Installed internal capabilities do not consistently advertise cleanup

**Promise:** Medium, rising with dynamic tool generation  
**Status:** Mostly future-facing.

#### Failure mode

Generated capabilities may modify registries, prompt fragments, adapters,
callbacks, caches, or search paths. Installation without ownership and cleanup
leaves persistent residue after rejection or replacement.

#### Paper mechanism

Every reversible internal mutation returns its inverse. Inverses are applied in
reverse order.

```text
install tool -> undo_tool
add prompt fragment -> undo_prompt
add planner adapter -> undo_adapter

withdraw:
    undo_adapter()
    undo_prompt()
    undo_tool()
```

#### Boundary

This applies to internal controlled state. It does not erase an ARC action,
external message, provider call, cost, or observation. Those are irreversible
events or require explicit compensation.

#### Smallest candidate mechanism

Any future generated-tool installation API should return a `dispose` handle and
register it under an owning revision. Do not retrofit static code merely to
imitate the paper.

---

### L8. Provider withdrawal is not modeled separately from dependent withdrawal

**Promise:** Medium  
**Status:** Latent dependency-integrity concern.

#### Failure mode

Withdrawing a transition model can orphan planners, discriminators, cached plan
evaluations, or generated macro-actions that still appear active.

#### Paper mechanism

Stop new consumers, drain or divert existing dependents, unload them, then
withdraw the provider.

```text
mark WM9 withdrawing
prevent new consumers of WM9
invalidate P7, S4, D8
land or quarantine in-flight results
unload dependents
withdraw WM9
activate WM10
reload compatible consumers
```

#### Question

Can ordinary immutable revisions and publication checks already guarantee this,
making explicit drain states unnecessary?

---

### L9. Policy and authority may become coupled to service implementations

**Promise:** Medium-low now; potentially high with richer realms and tools  
**Status:** Partly covered by existing tool profiles, mode policy, deadlines,
and IPC boundaries.

#### Failure mode

The actor, builder, planner, and shadow evaluator may need the same conceptual
service with different authority:

```text
actor: may publish one real ARC action
builder: may inspect evidence and stage files
planner: may simulate many actions but publish no real action
shadow realm: may compare outputs but change no canonical state
```

#### Paper mechanism

Interception applies contextual policy at dependency access.

```text
caller=actor, realm=canonical, operation=real_action, budget=1
caller=planner, realm=H2, operation=simulated_action, budget=10000
```

#### Warning

Interception is not a security sandbox. Enforcement must remain at an authority
boundary the caller cannot bypass.

---

## 5. More immediate applications retained from the first analysis

These mechanisms address nearer defects. They remain useful, but they are not
the deepest contribution of the paper.

### I1. Reactive goal-to-discriminator obligation

When a candidate objective is recorded, create a structured unmet requirement:

```text
candidate goal
    -> needs discriminator
    -> discriminator ready
    -> awaiting consequence
    -> supported | contradicted | inconclusive
```

Required fields:

```text
hypothesis
predicted observation
credible rival
legal discriminating action
promotion/demotion rule
action deadline or expiration
```

The host should enforce structural completeness, not pretend to determine which
hypothesis is true.

### I2. Transactional builder revisions

Builder work should resemble a database transaction or temporary Git branch:

```text
Committed WM7
    -> staged candidate WM8
    -> interface and evidence verification
    -> current-context check
    -> atomic publication

on timeout, error, or stale context:
    preserve diagnostic artifacts
    discard candidate publication
    keep WM7 active
```

This prevents partial builder mutations from becoming the live model. It does
not undo actions taken in the ARC environment.

### I3. Verified capability provisions

Replace the coarse `world_model.py exists` readiness rule with a host-verified
capability manifest. This is both an immediate completeness repair and a
foundation for later contextual composition.

### I4. Planner identity as observational equivalence

Two states may share a planner key only if every planner-visible operation is
unable to distinguish them in a way that changes reachable goals.

For equal-key states, check at least:

```text
same legal action set
same successor keys under each legal action
same outcome interpretation
same required planner-visible rendering or observations
```

Example: equal visible positions with different histories are not equivalent if
`UNDO` produces different futures.

This must follow, not precede, an executable goal and adequate operators.

### I5. Isolated realms for rival hypotheses

Use only when two live executable hypotheses need distinct derived artifacts.
Existing frozen checkpoints and shadows may already provide the required
isolation for many experiments.

### I6. Revertible generated-tool installation

Require cleanup handles when dynamic installation becomes real. Avoid a generic
effect algebra while surfaces remain static.

### I7. Contextual interception

Extend current policy boundaries only when a demonstrated case requires the
same service to behave differently across roles or realms.

---

## 6. Candidate minimal domain model

This is a thinking model, not a committed implementation.

```python
ArtifactRevision(
    key,
    revision,
    producer,
    realm,
    evidence_commitment,
    dependency_revisions,
    verification_status,
    lifecycle_status,
    payload_ref,
)

Requirement(
    consumer,
    required_keys,
    committed_providers,
    target_providers,
    activation_status,
    blocking_reasons,
)
```

Candidate keys:

```text
candidate_goal
named_discriminator
verified_state_decoder
verified_transition_model
verified_action_set
executable_goal
verified_outcome_predicate
congruent_planner_key
validated_plan
action_authority
```

Candidate consumers:

```text
DiscriminatorObligation
ModelPublicationGate
PlannerReadiness
ActionPublicationGate
CompactionValidityCheck
```

The smallest useful main path may be:

```text
observation E13
    -> candidate goal G4
    -> discriminator obligation D4
    -> legal probe
    -> observation E14
    -> staged model WM8
    -> verified capability manifest
    -> atomic publication
    -> planner-key congruence check
    -> plan P7 with exact commitments
    -> current-context publication check
    -> irreversible ARC action
```

---

## 7. Hypotheses

### STC-H1. Hybrid-revision reasoning will become a dominant latent failure

**Prediction:** As concurrent builders, planners, compaction, and generated tools
increase, some decisions will depend on mutually incompatible artifact
revisions even though each artifact is locally valid.

**Rival:** Immutable files, existing provenance, and serialized decision flow
already prevent consequential mixing.

**Discriminator:** Reconstruct complete dependency lineages for a stratified set
of decisions and search for incompatible commitments.

**Decision:** Introduce explicit epochs only if reconstruction is incomplete or
actual incompatible use is found.

### STC-H2. Publication-time context checks capture most lifecycle value cheaply

**Prediction:** Attaching committed revisions to async work and validating them
at publication will prevent the important stale-result failures without a full
component runtime.

**Rival:** Stale consumers can cause harmful effects before publication, so full
activation and withdrawal control is required.

**Discriminator:** Enumerate effects available during each async operation. Test
whether a staged operation can mutate canonical or external state before its
publication check.

### STC-H3. Executable hypothesis isolation improves objective discovery

**Prediction:** Preserving two prediction-bearing realms will produce more
discriminating probes and fewer premature commitments than a single mutable
canonical model.

**Rival:** Additional realms consume budget, dilute attention, and produce more
rhetoric without changing actions.

**Discriminator:** At frozen failure checkpoints, compare one canonical
hypothesis against two capped isolated hypotheses. Measure distinct predictions,
legal discriminators, actual choice, and later hypothesis resolution.

### STC-H4. Fine-grained verified capabilities prevent invalid downstream work

**Prediction:** A capability manifest will block planning when actions, outcomes,
or key congruence are absent while still allowing valid simulation and
discrimination work.

**Rival:** Existing runtime errors and planner gates already expose missing
capabilities at lower complexity.

**Discriminator:** Replay known partial models through both readiness schemes and
compare false-ready, false-blocked, and diagnosis quality.

### STC-H5. Reactive invalidation is superior to rebuild-all only after scale

**Prediction:** With many consumers, declared contextual dependencies reduce
unnecessary recomputation while preventing stale artifacts.

**Rival:** The dependency graph remains small; explicit orchestration is easier
to inspect and more reliable.

**Discriminator:** Measure actual invalidation fan-out and rebuild cost before
designing a reactive runtime.

### STC-H6. Declarative reconciliation improves recovery fidelity

**Prediction:** Reconstructing the target view from authoritative evidence after
a restart produces the same active configuration as the uninterrupted run.

**Rival:** Some legitimate agent state is path-dependent and cannot be derived
from saved evidence without losing tacit or model-internal information.

**Discriminator:** At controlled checkpoints, compare uninterrupted continuation
with reconstructed continuation. Separate artifact parity, decision parity, and
gameplay consequence.

### STC-H7. Dynamic installation without cleanup causes epistemic residue

**Prediction:** Rejected or replaced generated tools will continue influencing
prompts, registries, caches, or planning unless installation has explicit
ownership and reversal.

**Rival:** Process isolation and fresh builders already bound all relevant
residue.

**Discriminator:** Inventory mutation lifetime and ownership before building any
cleanup mechanism.

### STC-H8. A mechanically activated discriminator obligation changes behavior

**Prediction:** Converting candidate goals into expiring structured obligations
will increase named discriminating actions and objective-relevant evidence.

**Rival:** It produces better-formatted reports without changing action choice.

**Discriminator:** Run a frozen matched checkpoint first. Promote to closed-loop
evaluation only if the treatment changes the predicted observation or legal
action in a relevant way.

### STC-H9. Transactional staging prevents partial-model contamination

**Prediction:** Builder timeout, exception, failed verification, or stale context
cannot alter the committed active model when all builder work occurs in a staged
workspace with atomic publication.

**Rival:** The existing builder writes are effectively atomic enough in practice,
and staging adds latency without affecting outcomes.

**Discriminator:** Inject failure after each mutation boundary and compare active
workspace hashes and downstream behavior.

---

## 8. Questions that could change the design

### Context and identity

1. What is the smallest set of revisions that defines an epistemic epoch?
2. Is evidence-head identity sufficient, or must the context name the exact
   selected subset and interpretation of evidence?
3. Does provider identity matter when two providers are extensionally equal on
   every verified operation?
4. Should compatibility be nominal, content-addressed, behaviorally verified,
   or some combination?
5. Which artifacts require exact lineage before they can authorize a real action?
6. Can an old plan be revalidated against a new model without recomputation?
7. When should stale work be preserved, and when is preserving it merely clutter?

### Lifecycle and concurrency

8. Which current operations genuinely overlap?
9. Which operations mutate canonical state before completion?
10. Which can be cancelled safely, which must land, and which have already
    crossed an irreversible boundary?
11. What is the quiescence condition for one Tycho Nooa decision?
12. Can the actor continue acting while a builder revision is staged?
13. What exactly drains when a world model is withdrawn?
14. How are model-call cost and elapsed time represented when a result is stale?
15. Is stale computation still admissible evidence about the provider or model?

### Hypothesis realms

16. What qualifies a rival hypothesis for its own realm?
17. Must realms predict different observations before they may consume budget?
18. Which data are immutable and shared across all realms?
19. Can derived features be shared safely, or must every derivation carry realm
    lineage?
20. Who has authority to promote a realm to canonical status?
21. What evidence is sufficient to merge two realms?
22. How do we prevent realm multiplication from becoming indecision?

### Capabilities and verification

23. What are the true capabilities currently hidden behind `world_model.py`?
24. Which are syntactic provisions and which require behavioral verification?
25. What evidence scope should accompany `verified`?
26. Can a capability be verified on the recorded path but contradicted on legal
    counterfactual branches?
27. How should partial validity be represented without encouraging unsafe use?
28. Is planner-key congruence itself a provided capability or a relation verified
    between a key, action set, transition model, and goal?

### Reconciliation and history

29. Which active state should be a pure function of authoritative evidence?
30. Which state is legitimately historical and cannot be reconstructed?
31. After restart, what parity is required: artifacts, model requests, immediate
    decisions, or full future behavior?
32. Can compaction be treated as a replaceable provider of memory context?
33. What dependencies must a summary declare so that later evidence can
    invalidate it?
34. Does convergence remain desirable when different histories formed different
    useful habits?

### Effects and authority

35. Which internal mutations are exclusive and genuinely reversible?
36. Which operations require compensation rather than reversal?
37. Who owns registrations made by generated tools?
38. Can cleanup failure leave the system in a known degraded state?
39. At what boundary is real ARC action authority enforced?
40. Can interception be bypassed by a model-authored tool or direct executor
    reference?

### Evidence and evaluation

41. What observed failure would justify an epistemic epoch implementation?
42. What existing mechanism would we delete if committed views are introduced?
43. How will we distinguish architectural correctness from model uptake?
44. How will we distinguish model uptake from decision influence?
45. How will we distinguish decision influence from gameplay benefit?
46. What positive control demonstrates that a treatment can affect the model at
    the chosen checkpoint?
47. What is the cheapest counterexample capable of breaking each mechanism?

---

## 9. Candidate research sequence

This sequence postpones infrastructure until a smaller test shows that the
distinction matters.

### Step 1: lineage audit, no behavior change

For several real decisions, draw:

```text
authoritative observations
    -> goal revisions
    -> model revisions
    -> capability verification
    -> planner request
    -> plan
    -> action
```

Record where exact revision identity cannot be recovered. Search for one actual
cross-revision use.

### Step 2: publication gate prototype

Attach revision commitments to one async product, preferably a builder result or
plan. Force its target to change before completion. Verify that it is preserved
as stale evidence but cannot alter the canonical artifact.

### Step 3: capability manifest prototype

Run existing complete and incomplete world models through a host-generated
manifest. Verify that simulation remains available where valid and planning is
blocked only for named absent or contradicted requirements.

### Step 4: two-realm frozen comparison

At a goal-acquisition failure checkpoint, instantiate exactly two capped
hypotheses with distinct predictions. Compare the selected legal discriminator
against the canonical-only control.

### Step 5: one transactional builder failure test

Inject timeout or exception after a staged edit. Check that the committed model
and consumer view remain unchanged.

### Step 6: only then consider a reusable runtime abstraction

Promote the mechanism only if at least two independent real uses reveal the same
ownership, identity, lifecycle, and invalidation structure. Otherwise retain
local gates and ordinary immutable artifacts.

---

## 10. Failure tests for the proposals themselves

The paper-inspired architecture should be rejected or revised if it produces:

1. **Epoch theater:** revision identifiers are recorded but never checked at a
   consequential boundary.
2. **Manifest theater:** generated code declares capabilities and the host trusts
   those declarations without evidence.
3. **Realm theater:** multiple hypotheses restate one another and predict the
   same observations.
4. **Reactive thrashing:** small evidence changes trigger cascading rebuilds
   without changing decisions.
5. **False rollback:** the system claims to reverse external actions, spent cost,
   or observed emissions.
6. **Lifecycle deadlock:** consumers wait for providers while providers wait for
   consumers to drain.
7. **Historical erasure:** declarative convergence discards evidence or learned
   distinctions that should survive.
8. **Provenance overload:** lineage recording consumes more cognitive and
   engineering budget than the failures it prevents.
9. **Authority leakage:** a noncanonical realm can publish a real action.
10. **Abstraction without deletion:** the new runtime duplicates existing
    checkpoints, shadows, staging, or policy gates instead of simplifying them.

---

## 11. What not to import from the paper yet

1. Do not port the Cordis runtime wholesale.
2. Do not assume the paper's confluence theorem directly describes an agent whose
   evidence and external environment evolve irreversibly.
3. Do not force all cognition into reversible effects.
4. Do not introduce generic isolation realms before two real hypothesis branches
   need simultaneous executable state.
5. Do not replace explicit orchestration with a dependency runtime merely because
   the latter is more elegant.
6. Do not treat the Koishi implementation history as controlled evidence of
   benefit for Tycho Nooa.
7. Do not call a component independent merely because its files do not overlap;
   semantic effects may still fail to commute.
8. Do not optimize planner-state identity before goal satisfiability and operator
   fidelity have been established.

---

## 12. Consolidated ranking

| Rank | Idea | Expected leverage | Current evidence | Earliest honest move |
|---:|---|---|---|---|
| 1 | Epistemic epochs and committed views | Prevent hybrid reasoning across incompatible revisions | Structural inference | Audit complete decision lineages |
| 2 | Publication-safe lifecycle for in-flight cognition | Makes continuous revision coherent | Partial exposure | Stale-result publication gate |
| 3 | Isolated executable hypothesis realms | Makes disagreement predictive and testable | Existing shadows show local feasibility | Two-realm frozen comparison |
| 4 | Fine-grained verified capability provisions | Prevents false substitution and false readiness | Direct partial-model evidence | Host-generated manifest |
| 5 | Reactive contextual dependencies | Enables precise invalidation | Latent | Measure actual invalidation fan-out |
| 6 | Declarative target-view reconciliation | Reduces event-order dependence and improves recovery | Latent | Restart parity map |
| 7 | Reversible internal installation | Prevents residue from generated capabilities | Future-facing | Inventory dynamic mutations |
| 8 | Dependent-before-provider withdrawal | Preserves dependency integrity during replacement | Latent | Model consumer graph |
| 9 | Contextual interception | Separates shared service from role-specific authority | Partly implemented | Authority-path audit |

For the current solver, the highest-value near-term probes remain:

1. mechanically activate a candidate-goal-to-discriminator obligation at a
   frozen failure checkpoint;
2. stage builder changes transactionally and test injected failure;
3. generate a verified capability manifest for existing partial models;
4. check planner-key congruence only after an executable goal and adequate
   operators exist.

---

## 13. Final working thesis

Tycho Nooa should eventually be able to answer five questions for every
consequential result:

```text
1. What exact context made this result valid?
2. Which providers and revisions did it depend on?
3. What change would invalidate it?
4. What happens if that change occurs while work is in flight?
5. Which effects can be reversed, which can only be quarantined, and which are
   irreversibly part of the world?
```

If the current system can already answer these questions through simpler
immutable artifacts, checkpoints, and explicit orchestration, those mechanisms
should be retained. If it cannot, the paper supplies a disciplined vocabulary
for the missing structure:

```text
committed views
target views
reactive coeffects
isolation realms
dependent-safe withdrawal
revertible internal effects
transactional replacement
```

The first decisive question is therefore not whether to build a composability
runtime. It is:

> Can we reconstruct one complete Tycho Nooa action and prove that every artifact
> contributing to it was valid against the same evidence, goal, model, and
> policy context?

That audit determines whether contextual validity is presently preserved,
merely reconstructible, or already broken.
