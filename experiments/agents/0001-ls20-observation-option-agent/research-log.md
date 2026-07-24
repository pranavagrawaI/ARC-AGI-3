# 0001 LS20 Observation-Option Agent

## Milestone 1: Passive trace understanding

### Hypothesis

A game-general parser using value segmentation, connectivity, temporal identity,
and action correlation can recover stable visual entities and action-linked events
from an admissible human trace without receiving object roles.

### Given

- rendered observations;
- public action IDs and transition order;
- public terminal and level-completion signals;
- generic geometry, connectivity, persistence, and comparison mechanisms.

### Learned or inferred

- the background value in each observation;
- neutral connected-region proposals;
- persistent object IDs;
- translations and appearance changes;
- objects that move consistently after particular actions;
- co-moving groups that may later support composite identity.

The implementation contains no fixed names or roles for visual objects. Semantic
role induction is deliberately deferred until repeated event evidence exists.

### Cheapest checks

1. Synthetic translation preserves identity and produces the expected delta.
2. A same-location visual mutation preserves identity and records appearance change.
3. Trace loading discards private recording fields and aligns each action with its
   before/after public observations.
4. The successful human trace produces seven public completion events and a final
   `WIN` while emitting only neutral object/event records.

### Result

The first passive evaluation processed the successful 404-action human trace:

- seven public completion events and final state `WIN` were preserved;
- the parser emitted 10–40 neutral detections per frame (median 28);
- all 404 transitions changed pixels and 396 contained tracked motion;
- after an eight-move confidence gate, 65 individual identities and 21 co-motion
  groups remained action-dependent candidates;
- the strongest co-motion group was neutral IDs `[29, 31]`: 44 movements with
  perfect within-action consistency and four distinct learned mappings:
  `A1→(-5,0)`, `A2→(5,0)`, `A3→(0,-5)`, `A4→(0,5)`.

The two group members retained their own inferred values, shapes, areas, and
bounding boxes; no composite role was supplied. Manual inspection of the emitted
metadata confirms that they move together as a stable two-part visual entity.

The broad candidate pool is an honest warning: action correlation alone does not
yet distinguish direct control from other action-conditioned motion. Likewise,
stationary appearance changes occur in 314 transitions, so they cannot yet be
treated as contact effects. The next falsifier must compare candidate motion
against an action-independent baseline and require spatial/temporal contact
evidence before any affordance is learned.

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/human-trace-events.jsonl`
- `experiments/artifacts/0001-ls20-observation-option-agent/human-trace-summary.json`

### Online control and contact continuation

The action-coupling evidence now feeds an online `ControlledEntityBelief` rather
than remaining a post-hoc ranking. It discovers one controlled co-motion group in
every public level. Acquisition costs in trace actions were `[6, 2, 9, 4, 7, 6,
3]`: level 0 was learned from local intervention evidence; levels 1–6 reused only
the learned action-to-displacement model and grounded fresh level-local IDs.

A geometric interaction detector then projects the controlled pixels by the
learned displacement and asks which neutral regions occupy the intended
destination. Its first version fired on 364/365 controlled steps because a large
scene substrate spans almost every destination. Requiring the intended footprint
to cover at least half of a candidate region reduced this to 46 discrete-contact
candidates. The criterion is relative to each region's inferred area and contains
no fixed palette, coordinate, or role name.

There are 41 observed blocked interactions. However, unrelated visual changes
still accompany 358 interaction events, so simultaneous change is not yet causal
evidence. No affordance role has been assigned. The next experiment must estimate
an autonomous-change baseline from non-contact transitions, then retain only
changes whose conditional rate increases around repeated contact.

The first conditional-rate pass returned 96 candidates, dominated by tracker
`disappearance` events. This was rejected as causal evidence: visibility loss and
identity loss are observational ambiguity, not demonstrated effects. Appearance
and disappearance remain in the event log but are excluded from affordance
learning until bounded occlusion identity is implemented.

The remaining attribution was tightened through three additional falsifiers:

1. compare contacts only against non-contact transitions in the same public level;
2. require a persistent neutral target identity rather than pooling by pixel value;
3. admit only stationary value/form transformations with at least three repeated
   contacts and at least 0.5 excess conditional rate.

Simultaneously overlapped parts are treated as one composite contact signature.
This leaves six candidate edges. Two are classified rather than generic gaps:
the same two-part contact signature in level 1 was observed three times, and two
persistent remote identities (`38` and `39`) each underwent `rotation:1` on all
three contacts. Each target had 36 within-level non-contact opportunities and zero
baseline rotations. Their bounding boxes remained fixed, so the result is a
contact-conditioned remote quarter-turn, not translation or scene reset.

This is the first earned affordance relation in `0001`. It remains numeric and
level-local: no object role or human name has been attached. Four other candidates
remain `unclassified_form_change` and are retained as hypotheses, not conclusions.

### Persistent neutral affordance and state memory

The conditional evidence is now exposed as a runtime affordance table rather than
only a post-hoc JSON grouping. Each active relation records:

- its level-local composite contact signature;
- the persistent target identity or co-changing target group;
- the observed transform kind;
- contact and non-contact support;
- the current target state;
- the chronological target-state transitions;
- whether those transitions are deterministic, continuous, and cycle-closing.

The classified level-1 relation activates online at step 38. Its three contacts
produce one continuous four-state joint path for target IDs `[38, 39]`, with
`rotation:1` on both targets at every transition. The path is deterministic and
continuous, but it does not return to its initial state. Therefore the memory
reports `closed_cycle=false`; a four-state cycle would be plausible but is not
observed by this trace.

This pass also exposed non-monotonic hypotheses. Across the trace there are five
affordance activations and two retractions, leaving three active relations. The
retractions occur because later non-contact observations alter the autonomous
baseline and change which targets clear the conditional-rate gate. The event
artifact now records both activation and retraction instead of presenting every
temporary candidate as permanent knowledge.

Only one final relation has a classified transform. The other two are still
`unclassified_form_change`; their target transitions are not continuous across
contacts, so they cannot yet support an attribute-state controller. No observed
relation closes a state cycle.

### Bounded visual occlusion

The tracker now retains an unmatched neutral identity for one observation before
declaring disappearance. If a compatible detection returns within that bound, it
reuses the prior ID and emits a reappearance event. A match across the hidden gap
is deliberately excluded from motion and appearance deltas: its displacement may
span multiple actions and cannot be attributed to the latest action alone.

On the full trace, 56 transitions contain at least one bounded reappearance. The
classified relation survives this stricter identity model unchanged in substance:
it still activates at step 38, has three of three contact support against zero of
36 baseline transforms, and yields a continuous four-state `rotation:1` path for
two persistent targets. Numeric IDs changed because fewer temporary disappearances
allocate replacement identities.

This does not prove identity through arbitrary occlusion. One frame is a generic,
explicit bound, and nearest compatible matching can still swap indistinguishable
objects. It is enough to falsify immediate tracker loss, but not enough to admit
disappearance as a learned causal outcome.

### Scaffolding boundary

`0001` stops short of named LS20 mechanic detectors. The concrete parser, tracker,
event differencer, contact geometry, and conditional-rate memory are instruments:
they generate candidate structure and falsifiable evidence. They are not the
final ontology.

In particular, the next step is not a dedicated pickup, changer, reset, trap, or
goal detector. It is a generic learned conditional transition model:

```text
history + candidate entity/relation + action
    → distribution over future visual/state changes
```

The existing event artifact is training material for that model. Its geometric
event labels are pseudo-labels and measurements, not truth. Future perception must
also treat connected components as region proposals whose usefulness is learned,
not as an assertion that same-valued connectivity defines an object.

This boundary is recorded in `docs/adr/0002-hard-code-the-lab-not-the-theory.md`.

### Generic transition prediction tracer bullet

The first learned-theory seam is a prequential transition predictor over neutral
event records:

```text
history + action + optional contact signature
    → probability distribution over generic visual-change symbols
```

The model is deliberately small: empirical symbol frequencies pooled across a
hierarchy of contexts (`action_contact`, `action`, `global`) with additive
smoothing. It predicts each transition before observing it, then updates from the
true outcome. This makes the metric an online learning check, not a replay score.

Two targets are evaluated:

1. broad visual symbols from all transitions;
2. contact-effect symbols: generic changes to non-controlled objects while a
   neutral contact signature is present.

The first hard-backoff version failed the baseline: sparse contact contexts
overrode better global evidence. Replacing hard backoff with hierarchical pooling
fixed that failure and is the current result:

```text
broad symbols, 404 examples:
  global log-loss         2.877062
  action_contact log-loss 2.724576
  Δ log-loss             -0.152486

contact-effect symbols, 48 examples:
  global log-loss         1.662372
  action_contact log-loss 1.622441
  Δ log-loss             -0.039931
```

Mean probability assigned to observed symbols also improves:

```text
broad symbols:          +0.055338 over global
contact-effect symbols: +0.022897 over global
```

Top-1 accuracy does not improve. That is not surprising: common symbols dominate
the argmax, while the more useful signal is probability mass assigned to rarer
observed effects. The result is therefore modest but real: action/contact context
contains predictive information, and sparse context must be pooled rather than
trusted as an isolated rule.

Artifact:

- `experiments/artifacts/0001-ls20-observation-option-agent/transition-learning-summary.json`

### Relation-scoped effect hypotheses

The transition learner now exposes a relation table:

```text
neutral contact signature
    → generic effect distribution
```

This is still learned from admissible event records and remains name-free. It is
not a mechanic classifier. The table reports:

- support: how often the contact signature was observed;
- effect rate: how often an effect appeared under that contact;
- lift: effect rate divided by the effect's rate across all contact events.

The lift score was added after the first table was dominated by common
`effect_area_change` and `effect_transform:unclassified_form_change` symbols.
Those effects are real observations, but they are common enough that they are
weak option proposals. Lift exposes relation-specific surprise without assigning
semantic roles.

Current top relation hypotheses include the earlier step-38 relation as a purely
generic learned hypothesis:

```text
contact:
  (value=0, area=3, shape=dd2e8905879a)
  (value=1, area=1, shape=7d9d4bda4403)
  (value=1, area=1, shape=7d9d4bda4403)

support: 7
top lifted effect:
  effect_transform:rotation:1
  rate: 1.0
  lift: 6.8571
```

This is the first table-shaped bridge toward option proposal: an option proposer
can ask for contact signatures with high-support, high-lift predicted effects
instead of relying on named mechanics. The table remains noisy and co-occurring
effects are not yet causally separated; it proposes experiments, not truths.

### Probe proposals and replay assay

`0001` now has a small probe policy:

```text
relation hypotheses
    → high-support/high-lift probe proposals
```

Each proposal is still neutral:

```text
contact_signature C
predicted_effect E
support
effect_rate
lift
```

This is not a live controller yet. The current assay is replay-only: learn probe
proposals from the first two-thirds of contact records, then test them on the
final third as a proxy for intentional reproducibility and negative controls.

The assay asks:

```text
when the same contact appears later, does E occur?
when other contacts appear later, does E stay absent?
```

Current replay assay:

```text
train contact records:   32
holdout contact records: 16
probe proposals:          3
```

The strongest relation passes this replay filter:

```text
predicted effect: effect_transform:rotation:1
support at proposal time: 6
effect rate: 1.0
lift: 5.3333
holdout positive attempts: 1
holdout positive successes: 1
negative controls: 15
negative-control violations: 0
```

Two other high-lift proposals are weakened by negative controls. One has no
positive holdout attempts and seven negative-control violations; another
reproduces on three matching contacts but also appears under four nonmatching
contacts. This is useful: high lift is a proposal score, not a causal guarantee.

The next live step is a routing-and-contact controller that can intentionally
recreate a proposed contact signature and compare it with a nearby nonmatching
contact. Until that exists, this assay is an evidence-preserving proxy, not an
intervention result.

### Trace-assisted live probe controller

`0001` now includes a live contact-probe controller for one neutral proposal.
The controller:

1. optionally executes an explicit admissible bootstrap action prefix;
2. updates the same parser, tracker, control belief, event detector, and
   interaction detector online from public observations;
3. chooses the action whose expected controlled footprint would overlap the
   proposed contact signature;
4. observes whether the predicted generic effect occurs;
5. routes to a nearby nonmatching contact when no direct negative control is
   adjacent;
6. records positive reproduction and negative-control violations.

The first implementation failed usefully: after the bootstrap prefix it routed
instead of taking the direct contact. The cause was not the contact signature; the
frozen active control estimate did not yet contain the horizontal action learned
later through transferred sensorimotor memory. The controller now expands the
active estimate with current `expected_delta(action)` values before choosing
contact/routing actions.

Current live smoke:

```text
mode: trace-assisted valid live probe
game: ls20
bootstrap actions: 29
total actions: 34
levels completed: 1

proposal:
  predicted effect: effect_transform:rotation:1
  support at proposal time: 6
  effect rate: 1.0
  lift: 5.3333

positive attempts: 1
positive successes: 1
negative controls: 1
negative-control violations: 0
```

This is the first real environment action assay of a learned neutral relation.
It is not fully online: the bootstrap prefix comes from an admissible human trace
and gets the agent into a comparable situation. The probe itself is live: the
positive contact action and subsequent negative-control routing are selected from
current observations and learned action-motion expectations, not copied from the
trace.

Artifact:

- `experiments/artifacts/0001-ls20-observation-option-agent/live-probe-summary.json`

### Milestone 1 closure: passive trace understanding

The milestone-1 analyzer now consolidates the passive trace evidence into one
summary artifact. It is intentionally conservative: it reports completion
proximity, generic contact/effect evidence, disappearance-like observations, and
reset/regression evidence separately instead of collapsing them into named game
mechanics.

Current milestone-1 summary:

```text
transitions:        404
final state:        WIN
levels completed:   7
overall status:     complete_with_known_limits

controlled movement:          done
attribute changes:            done
contact/effect relations:     done
disappearance/resource-like:  partial
regression/reset:             unknown_absent_in_trace
completion detection:         done
completion cause:             partial
```

The completion analyzer found seven public level-completion transitions and
records the recent neutral contact/effect context before each one. That is enough
for milestone 1's passive understanding target: the agent can reconstruct
movement, attribute-like visual changes, likely interaction contexts, and public
completion events from observations and actions only.

Two limits are explicit and should stay visible:

1. disappearance-like events are observed, but the current trace does not prove a
   resource variable or causal collection relation;
2. no public reset events or large same-level visual regressions occur in this
   successful trace, so reset/regression understanding remains
   `unknown_absent_in_trace`, not solved.

This completes milestone 1 as an evidence-producing passive trace reader. It
does not complete milestone 2. The next cheapest useful check is level-1 replay
with recovery: reuse the learned control/contact substrate, intentionally inject
one small deviation, and require the agent to recover through current
observation-derived belief rather than by replaying the human sequence.

Artifact:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-1-summary.json`

## Milestone 2: First-segment trace alignment with recovery

### Hypothesis

If the passive substrate is a useful acting spine rather than a trace script, then
a trace-assisted controller should recover from a small deterministic
displacement by observing its controlled visual group and closing the loop back
to the trace reference.

### Intervention

The milestone-2 controller adds four pieces:

- a first-segment trace reference ending at the first public level increment;
- windowed trace alignment with match margin and entropy;
- a live control verifier that suppresses repeated action/delta conflicts;
- a closed-loop local recovery servo with explicit action-source labels.

The evaluator runs three comparable assays through the public environment
boundary:

```text
normal trace-assisted run
pure replay with deterministic D1 displacement
trace-assisted run with the same D1 displacement
```

D1 is recorded as:

```text
injected step:   8
original action: 4
injected action: 3
reason:          harmless_displacement
valid:           true
```

### Result

Current milestone-2 summary:

```text
status: complete_with_known_limits

normal trace-assisted:
  completed:              true
  actions:                13
  extra actions vs trace: 0
  live confirmations:     6
  live conflicts:         0

pure replay + D1:
  completed:              false
  actions:                80
  stop reason:            action_budget

trace-assisted + D1:
  completed:              true
  actions:                15
  extra actions vs trace: 2
  action sources:         TRACE_PRIOR=13, LOCAL_RECOVERY=2
  realigned at step:      11
  live confirmations:     8
  live conflicts:         0
```

The anti-script invariant passed: after the injected displacement, the
trace-assisted action sequence diverged from pure replay, used local recovery,
and then completed the first segment. Pure replay with the same injected action
did not complete within the 80-action budget.

This does not prove a general option policy. It proves the first active bridge:

```text
trace prior + live visual alignment + verified action deltas
    → closed-loop correction after surprise
```

The test remains intentionally narrow. It is only the first segment, only
trace-assisted, and only D1 is required. D2/D3 deviations are named in the
artifact taxonomy but not yet completion criteria.

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-2-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-2-events.jsonl`

## Milestone 3: Structured recovery with epistemic brakes

### Hypothesis

Milestone 2 proved that `0001` can get back onto a trace after local
displacement. Milestone 3 tests the next question: whether it can tell when
getting back onto the trace is still valid.

The contract is:

```text
resume trace execution only if:
  current observation is compatible with a nearby trace-corridor state
  controlled visual position can be aligned through verified plain movement
  no non-controlled divergence appears during recovery
  resume validation returns TRACE_VALID
```

Otherwise the correct result is classified refusal, not blind trace continuation.

### Intervention

The milestone-3 layer adds:

- trace corridors over nearby reference states;
- neutral non-controlled semantic signatures with volatile-label tolerance;
- drift classification with explicit precedence:
  `ACTION_CONFLICT`, `WORLD_DRIFT`, `ON_TRACE`, `LOCAL_DRIFT`,
  `TRACE_INVALID`, `UNKNOWN`;
- bounded actor-only BFS over verified plain-movement deltas;
- blocked directed-edge memory;
- a resume circuit breaker.

The acceptance surface is hybrid. D1 and D2 are live public-environment action
substitutions. D3a, D3b, and D4 use synthetic public `Observation`-style runtime
paths because mutating non-controlled world state inside the live environment
would cross the public API boundary.

### Result

Current milestone-3 summary:

```text
status: complete_with_known_limits
false resume rate: 0.0

D1 live local drift:
  drift type:        LOCAL_DRIFT
  completed:         true
  recovery actions:  2
  resume validation: TRACE_VALID

D2 live larger local drift:
  drift type:        LOCAL_DRIFT
  completed:         true
  recovery actions:  4
  resume validation: TRACE_VALID
  valid injections:  step 8 action 3, step 9 action 2

D3a blocked direct correction, alternate route:
  drift type:        LOCAL_DRIFT
  recovery plan:     [2, 1, 1, 3]
  non-greedy:        true
  resume validation: TRACE_VALID

D3b blocked direct correction, no route:
  drift type:        TRACE_INVALID
  failure reason:    TARGET_UNREACHABLE
  trace resumed:     false

D4 non-controlled semantic change:
  drift type:        WORLD_DRIFT
  failure reason:    SEMANTIC_INCOMPATIBILITY
  trace resumed:     false
```

This milestone does not prove a general learned world model. It proves a narrow
epistemic brake:

```text
trace prior + semantic corridor check + local planning
    → recover when compatible
    → refuse when compatibility fails
```

The most important result is D4. The controlled position is compatible, but the
non-controlled signature is not. The agent records semantic changes and refuses
to resume. That is the distinction Milestone 2 did not have.

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-3-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-3-events.jsonl`

### Milestone 3 freeze and M3.5 scout

The milestone-3 claim is now frozen as a separate artifact. The freeze records
the official claim, trace identifier, public observation source, corridor and
BFS parameters, perturbation protocols, test/audit counts, and proof-panel
excerpts for D3a and D4.

Artifact:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-3-freeze.md`

An optional M3.5 scout was added to look for live semantic drift caused only by
legal public actions. The first version was rejected as too permissive because a
nearby-object signature can confuse actor relocation with world change. The
scout now compares absolute non-controlled signatures against nearby trace
corridor states and filters large substrate-like proposals.

Current scout result:

```text
status: available
probes: 24

candidate:
  probe step:       6
  original action:  1
  probe action:     2
  drift type:       WORLD_DRIFT
  failure reason:   SEMANTIC_INCOMPATIBILITY
  state after:      NOT_FINISHED
  levels before:    0
  levels after:     0
  control available: true
```

This is not a completed M3.5 milestone. It is a candidate live public-action
semantic-drift assay. The next decision is whether to promote this scout result
into an actual M3.5 evaluator that requires `WORLD_DRIFT`, refuses trace resume,
and records a proof panel under the same resume-gate contract.

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-35-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-35-events.jsonl`

## Milestone 4: trace-to-option abstraction

### Hypothesis

A trace segment can be converted into an inspectable state-based option. The
option may be induced from public demonstration evidence, but execution should
pursue an observable termination predicate with the raw segment action list
masked.

This is the first trace-fading step from:

```text
follow this demonstrated action chunk
```

to:

```text
satisfy the transition predicate this chunk appears to demonstrate
```

### Intervention

Milestone 4 adds:

- `OptionSegmenter`, which induces option candidates from public trace state and
  event structure;
- `OptionCandidate`, an inspectable predicate object containing initiation,
  termination, expected actor delta, expected semantic delta, invalidation
  conditions, and evidence provenance;
- `OptionPredicateReferenceProvider`, the first non-frame reference provider;
- `OptionExecutor`, which receives only the option object, current observation
  summary, action-delta model, semantic validator, and local planner;
- an action-list masking invariant: option execution cannot query raw segment
  actions.

The current predicate abstraction level is deliberately low: exact observable
actor coordinates are allowed but flagged as `predicate_abstraction_level: 1`.
This is a weak predicate, not a final concept.

### Result

Current milestone-4 summary:

```text
status: complete_with_known_limits
option count: 5
chosen option: option_006_007
segment range: [6, 7]
trace dependency level: 3
raw action list masked: true
execution mode: predicate_pursuit
predicate abstraction level: 1

baseline exact-action replay under blocked-script perturbation:
  completed: false
  endpoint: [32, 21]

option execution:
  completed: true
  termination predicate satisfied: true
  plan: [3, 1, 4]
  plan differs from raw actions: true

semantic mismatch refusal:
  completed: false
  failure reason: INITIATION_CONDITION_FAILED
```

The selected option is induced from the real public trace. The execution assay
is public-observation-style synthetic: the raw segment action is blocked, exact
replay fails to satisfy the predicate, and the option executor reaches the same
observable predicate by a different route. Every option-execution action records
`used_raw_trace_action: false`.

This does not prove live autonomous LS20 option execution. It proves the
specific M4 mechanism:

```text
public trace evidence → option predicate → masked predicate pursuit
```

Known limits:

- options are induced from human traces, not self-discovered;
- the successful predicate is coordinate/landmark-level, not a general goal
  predicate;
- execution uses local action-delta planning, not a learned world model;
- the assay is first-segment bounded;
- there is no cross-option planner yet.

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-4-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-4-events.jsonl`
- `experiments/artifacts/0001-ls20-observation-option-agent/option-candidates.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-4-freeze.md`

## Milestone 4.5: predicate abstraction upgrade

### Hypothesis

An option predicate can become more abstract than an exact coordinate without
becoming vague. A higher-level predicate only counts if it remains observable,
evidence-backed, discriminative, validatable, and non-indexical.

### Intervention

Milestone 4.5 adds:

- `PredicateCandidate`, with explicit state/transition mode;
- hard gates before scoring;
- exact-position fallback preservation;
- semantic compatibility as a guard, not a termination predicate, unless a
  specific semantic delta is witnessed;
- predicate expression hashes for later identity checks.

The selection pipeline is:

```text
generate candidates
→ apply hard gates
→ score survivors
→ select best survivor
→ preserve exact fallback
→ report rejected candidates with reasons
```

### Result

Current milestone-4.5 summary:

```text
status: complete_with_known_limits
option count: 5
options with multiple predicates: 5
chosen option: option_006_007
selected predicate: option_006_007:pred_region
selected predicate hash: 9fdcb4b65dcaffc9
selected abstraction level: 2
fallback exact predicate available: true
selected predicate validates M4: true
selected predicate terminal selective: true
max candidate abstraction level: 4
```

The selected predicate is a terminal region:

```json
{
  "type": "region_membership",
  "center": [27, 21],
  "radius": 2,
  "metric": "manhattan",
  "excludes_initiation": true
}
```

The level-4 semantic compatibility candidate is rejected as a termination
predicate because it is classified as a guard. This is the intended behavior:
abstraction by overbroad compatibility is not accepted.

Known limits:

- predicates are still trace-induced;
- the selected level-2 predicate is spatial;
- this does not discover options without demonstration;
- this does not chain options or build a world model.

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-45-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-45-events.jsonl`
- `experiments/artifacts/0001-ls20-observation-option-agent/predicate-candidates.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-45-freeze.md`

## Milestone 5: option re-execution under route variation

### Hypothesis

If an option is real, the same option identity and selected predicate should
survive route variation. The executor should satisfy the same predicate from
valid route setups with raw trace actions masked, and refuse invalid starts or
semantic mismatch.

### Intervention

Milestone 5 adds:

- same-predicate identity checks across assays;
- selected predicate expression hashes;
- start validation records;
- route and path difference metrics;
- type-specific predicate progress records;
- invalid-initiation and mid-execution semantic mismatch refusal assays.

### Result

Current milestone-5 summary:

```text
status: complete_with_known_limits
option: option_006_007
selected predicate: option_006_007:pred_region
selected predicate hash: 9fdcb4b65dcaffc9
selected abstraction level: 2
same predicate across assays: true
raw action list masked: true
valid-start success count: 2
route differs from raw: true
route differs from M4: true
invalid initiation refused: true
semantic mismatch refused: true
```

Successful route variants:

```text
A_ORIGINAL_M4_BLOCKED_SCRIPT:
  plan: [3, 1, 4]
  path: [32,21] → [32,16] → [27,16] → [27,21]

C_ALTERNATE_BLOCKED_EDGE:
  plan: [4, 1, 3]
  path: [32,21] → [32,26] → [27,26] → [27,21]
```

Both satisfy the same selected predicate:

```text
region_membership(center=[27,21], radius=2)
```

The invalid initiation assay refuses before action. The semantic mismatch assay
stops after the first invalidating recovery action and records
`WORLD_DRIFT`/`SEMANTIC_INCOMPATIBILITY`.

Known limits:

- route assays are synthetic public-observation-style assays;
- the initiation condition remains exact-position;
- the option is trace-induced;
- no self-generated options or cross-option planning yet.

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-5-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-5-events.jsonl`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-5-freeze.md`

## Milestone 5.5 / 5.5b: event predicate evidence and retarget

### Hypothesis

The option system should not promote spatial options into event options unless
public trace/event evidence supports that abstraction. If clean event evidence
exists on a different option, the system should retarget narrowly instead of
laundering the original spatial option.

### Intervention

Milestone 5.5 adds an event-predicate evidence probe over public trace-event
records. It scans trace-induced options for level-3 contact/boundary predicates
and level-4 semantic-delta predicates, while classifying semantic compatibility
as a guard rather than a termination condition.

Milestone 5.5b retargets the clean event evidence found by the probe:

- `option_006_007` remains a level-2 spatial `region_membership` option;
- `option_009_010` becomes the event-option assay target;
- the selected predicate is transition-mode `semantic_delta_event`;
- the executor stops on event predicate satisfaction, not trace index;
- negative assays refuse when the event is absent or semantic guards fail.

### Result

M5.5 probe:

```text
status: available_event_predicate_evidence_elsewhere
M5 selected option: option_006_007
M5 selected option event predicates: none
event evidence elsewhere: option_009_010:created_other_object
```

M5.5b retarget:

```text
status: complete_with_known_limits
selected option: option_009_010
selected predicate: semantic_delta_event(created_other_object)
predicate mode: transition
abstraction level: 4
event type: non_controlled_object_created
raw trace actions masked: true
termination by event predicate: true
termination by trace index: false
negative event-absent assay refused: true
semantic guard refusal: true
noise promoted to event: false
```

Known limits:

- this is public-observation-style transition validation, not live event
  causation;
- the option is still trace-induced;
- object identity is based on public trace-event artifacts, not private state;
- no self-generated options yet.

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-55-evidence-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/event-predicate-candidates.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-55b-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-55b-events.jsonl`
- `experiments/artifacts/0001-ls20-observation-option-agent/event-option-candidates.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-55b-freeze.md`

## Milestone 6: self-generated option discovery

### Hypothesis

The option stack should be able to change option origin from human trace
segments to self-observed public interaction evidence. The executor must still
avoid scripts: it may not replay the human trace or the self-discovery action
sequence.

### Intervention

Milestone 6 adds a deterministic self-exploration pass from a public reset
observation. The resulting `DiscoveryRecord` stores the public before/after
observation hashes, exploratory actions, controlled actor delta, learned
movement model, semantic/contact deltas, and only `self_observed_*`
provenance. `SelfOptionInducer` converts that record into a
`source="self_exploration"` option with `trace_dependency_level=4`.

The selected option is then executed through the existing predicate executor
with only predicate, guards, invalidation conditions, action model, and current
observation. The discovery action sequence is masked during execution.

### Result

```text
status: complete_with_known_limits
selected option: self_option_000
source: self_exploration
trace dependency level: 4
selected predicate: region_membership
predicate abstraction level: 2
self discovery actions: [3, 3, 3, 1]
primary execution plan: [1, 3, 3, 3]
variation execution plan: [3, 1, 3, 3]
same predicate/hash across assays: true
raw trace actions used: false
discovery script used: false
invalid initiation refused: true
expectation failure refused: true
```

The first evaluator pass exposed a useful measurement bug: the raw controlled
component center drifted one pixel off the learned movement lattice. M6 now
uses the reachable endpoint implied by the self-observed action model while
preserving the raw observed endpoint in evidence. This keeps the option target
public and self-observed without making the planner chase parser jitter.

Known limits:

- this is a bounded public-observation self-option assay, not a general solver;
- the selected option is spatial, not a self-discovered event option;
- execution assays are public-observation-style predicate checks rather than a
  full online benchmark run;
- prior trace-trained machinery is reused, but selected M6 option evidence is
  self-observed.

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-6-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-6-events.jsonl`
- `experiments/artifacts/0001-ls20-observation-option-agent/self-option-candidates.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-6-freeze.md`

## Milestone 7: held-out option-effect world model

### Hypothesis

After M6, the next world-model rung is not a full simulator. It is an
option-effect model: from a self-discovered option and an applicability
signature, predict the option-level effect before execution, then compare the
observed effect against that prediction.

### Intervention

Milestone 7 splits option-effect evidence into three record types:

- `OptionEffectObservation`: post-execution observed effect evidence;
- `OptionEffectPrediction`: pre-execution model prediction;
- `OptionEffectVerification`: comparison between a prior prediction and a
  held-out observation.

The evaluator trains on the M6 primary self-option execution and validates on
the M6 variation execution. This prevents the model from merely restating the
same execution record it trained on. Negative assays test actor-effect mismatch
and semantic guard violation, while an unknown-key assay checks that the model
refuses to invent predictions.

### Result

```text
status: complete_with_known_limits
model scope: option_effect
modeled option: self_option_000
selected predicate: self_option_000:pred_region
selected predicate hash: dcd8141d642155b1
option origin trace dependency level: 4
prediction trace dependency level: 5
train record: effect_train_000
held-out record: effect_test_000
prediction generated before execution: true
positive verification: PREDICTION_VALID
actor mismatch verification: MODEL_PREDICTION_MISMATCH
guard violation verification: GUARD_VIOLATION
unknown key verification: UNKNOWN
support count: 1
empirical consistency: 1.0
```

Known limits:

- this is an option-effect world model, not a full frame transition model;
- the model covers one self-discovered option class;
- confidence is empirical consistency over observed records, not statistical
  certainty;
- no multi-option composition or full LS20 solving is claimed.

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-7-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-7-events.jsonl`
- `experiments/artifacts/0001-ls20-observation-option-agent/option-effect-observations.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/option-effect-model.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-7-freeze.md`

## Milestone 8: option-applicability chain prediction

### Hypothesis

The next rung after M7 is not executing a longer script. It is composition over
the option-effect model: predict that option A changes the option-applicability
state class so option B becomes applicable, verify A's effect, and release B
only if the observed next state satisfies the predicted applicability condition.

### Intervention

Milestone 8 adds an `OptionApplicabilityGraph` contract:

```text
node:
  option-applicability state class

edge:
  source state class -- option-effect prediction --> next state class
```

State classes are defined by applicable options, termination-predicate truth,
semantic guards, controlled-region class, and non-controlled semantic class. Raw
frame hashes are retained only as debug metadata; they are not the primary model
key.

The verifier enforces one invariant:

```text
No link i+1 may execute unless link i has verified its predicted effect
and the resulting observed state satisfies the predicted applicability condition
for option i+1.
```

### Contract status

```text
status: contract_superseded_by_behavior_assay
new graph object: OptionApplicabilityGraph
new node object: OptionApplicabilityStateClass
new edge object: OptionApplicabilityEdge
new chain verifier: OptionChainVerifier
positive contract result: CHAIN_VERIFIED
effect mismatch gate: LINK_EFFECT_MISMATCH
next applicability gate: NEXT_OPTION_NOT_APPLICABLE
unknown chain result: UNKNOWN_CHAIN
```

Known limits:

- this section records the M8 composition contract; the behavior assay below
  records the demonstrated M8 behavior;
- same-option-family chains remain weaker unless they use distinct state classes
  and verified recursive applicability;
- open-ended independent discovery of a second natural option remains future
  work.

Artifact:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-8-design-freeze.md`

### Behavior assay

The M8 behavior evaluator now constructs and verifies a two-link
option-applicability chain. `self_option_000` comes from the live M6 reset-window
self exploration. `self_option_001` is a derived continuation option whose start
class is the verified terminal class of `self_option_000`.

Result:

```text
status: behavior_demonstrated_with_known_limits
contract status: complete_with_known_limits
chain result: CHAIN_VERIFIED
prediction generated before execution: true
first link released second option: true
effect mismatch gate: LINK_EFFECT_MISMATCH
effect mismatch executed links: 1
next applicability gate: NEXT_OPTION_NOT_APPLICABLE
next applicability executed links: 1
distinct state classes: true
```

Interpretation:

```text
S0 --self_option_000 / verified effect--> S1
S1 enables self_option_001
S1 --self_option_001 / verified effect--> S2
```

The negative gates are the important evidence. If A's observed effect disagrees
with its prediction, the verifier stops at link 0 and does not release B. If A's
effect verifies but the observed S1 class does not list B as applicable, the
verifier also stops at link 0.

Official claim:

```text
Milestone 8 demonstrates that 0001 can compose learned option-effect
predictions into a short option chain, predict that one option's verified effect
will establish the applicability conditions for the next option, execute the
chain with per-link prediction checks, and refuse continuation when a prior
effect does not establish the next link.
```

Lab-book line:

```text
M8:
  I can see that one learned option can set up another.
```

Known limits:

- this is a behavior assay, not a full live LS20 solver;
- the demonstrated chain uses `self_option_000` from live M6 reset-window
  self-exploration and `self_option_001` as a derived self-observed
  continuation option;
- this proves the M8 composition contract and gates, but not open-ended
  independent discovery of a second natural option;
- the chain has two links only;
- no long-horizon option-graph search is claimed.

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-8-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-8-events.jsonl`
- `experiments/artifacts/0001-ls20-observation-option-agent/option-applicability-graph.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-8-freeze.md`

## Milestone 9: progress-guided option graph search

M9 adds the first bounded policy loop over the M8 option graph. The agent now
generates multiple option/chain candidates, scores them before execution with
explicit public progress/cost/risk features, selects by score, executes with M8
link verification, records the observed outcome, and applies attempt-memory
updates that affect a later same-state counterfactual selection.

Result:

```text
status: complete_with_known_limits
candidate count: 3
selected candidate: chain_000
selected score: 49.0
runner up: option_000
runner up score: 41.0
selection generated before execution: true
selected by explicit score: true
positive verification result: CHAIN_VERIFIED
negative verification result: NEXT_OPTION_NOT_APPLICABLE
model update types: FAILED_CANDIDATE_OBSERVED, FAILED_CANDIDATE_PENALTY
failed candidate score before update: 49.0
failed candidate score after update: 10.0
selected after update: option_000
same-state reselection check: true
state-change-only explanation ruled out: true
trace leakage findings: []
```

The decisive test is the same-state counterfactual reselection. Before the
failed-attempt update, `chain_000` is selected because it has the highest
predicted progress score. After a failed next-applicability assay, the same
candidate set and same start state class are rescored with attempt memory. The
failed chain receives a typed `FAILED_CANDIDATE_PENALTY`; its score drops below
`option_000`, and the selector changes choice without relying on a changed world
state.

Official claim:

```text
Milestone 9 demonstrates that 0001 can treat self-generated options and learned
option-effect chains as selectable candidates, score them using public progress
and uncertainty features, execute the selected candidate with prediction checks,
update its option graph or progress model from the observed outcome, and use
that update to change a subsequent selection or refusal.
```

Lab-book line:

```text
M9:
  I can choose what small objective to try next, and learn from the result.
```

Known limits:

- this is a bounded option-graph policy loop, not a full LS20 solver;
- progress features are engineered and deterministic;
- the graph is the M8 two-link graph;
- the same-state reselection is a counterfactual assay;
- learning/revising the progress predicate itself is deferred to M10.

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-9-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-9-events.jsonl`
- `experiments/artifacts/0001-ls20-observation-option-agent/option-chain-candidates.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/option-progress-model.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/option-selection-ledger.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-9-freeze.md`

## Milestone 10: progress predicate inference

M10 adds feature-level progress learning over the M9 option graph. M9 learned
candidate-level attempt memory: `chain_000` failed, so penalize `chain_000`.
M10 learns progress-feature usefulness: a public outcome updates trust in
features such as `new_option_enabled`, and that update rescales held-out
candidates that share the feature.

Result:

```text
status: complete_with_known_limits
progress hypotheses: 6
positive outcomes: 1
negative outcomes: 1
feature updates: 4
held-out candidate score before learning: 41.0
held-out candidate score after learning: 61.0
selection before learning: candidate_stable_prior
selection after learning: heldout_positive_feature
selection changed due to progress learning: true
candidate-memory-only explanation ruled out: true
noise feature status: SUPPRESSED
noise feature posterior weight: -4.0
forbidden source findings: []
```

The decisive test is held-out feature generalization. The positive chain outcome
updates progress hypotheses for features predicted by `chain_000`, including
`new_option_enabled` and `new_state_class`. A different candidate,
`heldout_positive_feature`, was not executed, but shares those learned features;
its score increases from `41.0` to `61.0`. In the same state and candidate set,
selection changes from `candidate_stable_prior` to the held-out candidate. M10
therefore does not rely on M9-style candidate-specific failure memory.

The negative assay creates a noise-like candidate with `volatile_visual_delta`.
Its outcome produces no stable downstream progress and records an
`action_conflict_or_next_not_applicable` penalty. The feature is marked
`SUPPRESSED`, with posterior weight `-4.0`.

Official claim:

```text
Milestone 10 demonstrates that 0001 can represent alternative public progress
hypotheses, update their estimated usefulness from option-chain outcomes, and
change future option selection based on learned progress evidence rather than
only fixed engineered weights or candidate-specific failure memory.
```

Lab-book line:

```text
M10:
  I can learn what kind of progress is worth pursuing.
```

Known limits:

- this is bounded progress-feature learning over the M9 graph;
- M9 engineered progress features are used as priors;
- the update rule is deterministic and small;
- this is not final goal inference or a full LS20 solver.

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-10-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-10-events.jsonl`
- `experiments/artifacts/0001-ls20-observation-option-agent/progress-hypotheses.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/progress-outcomes.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/progress-weight-updates.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/progress-selection-ledger.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-10-freeze.md`

## M10.5 diagnostic: live-unleash check

M10.5 is a diagnostic, not a capability milestone. It asks what happens if the
current M6-M10 stack is driven against the public LS20 environment without
adding a new live planner.

Result:

```text
status: diagnostic_complete_with_bottleneck
live executable selected candidate: chain_000
live executable plan: [3, 1, 3, 3, 3, 1]
live public actions attempted: 6
final state: NOT_FINISHED
levels completed: 0
M10 learned selected candidate: heldout_positive_feature
M10 learned selection live-executable: false
```

The run executed the current M9-selected executable chain plan through
`env.step` using the public LS20 interface. All six actions were legal, but the
public state remained `NOT_FINISHED` and `levels_completed` stayed `0`.

The diagnostic exposes the first integration bottleneck:

```text
NO_LIVE_OBSERVATION_TO_OPTION_GRAPH_STATE_CLASSIFIER
```

Current M10 progress learning can rescore diagnostic candidates, but those
learned-progress candidates are not yet wired back into a live option actuator.
After a selected chain plan is exhausted, the system cannot classify the fresh
public observation into an option-graph state, generate new live candidates from
that state, and replan.

Interpretation:

```text
The current stack can select and execute a short public action plan, but it does
not yet have the live state binding needed to continue M10 policy selection from
fresh LS20 observations.
```

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-105-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-105-events.jsonl`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-105-diagnostic-ledger.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-105-freeze.md`

## Milestone 10.6: live option-state binding and replanning

M10.6 closes the integration gap found by M10.5. The agent now executes a live
public option action plan, observes the result, binds the fresh public
observation back into the option-applicability graph, regenerates candidates
from that classified state, scores with the learned progress model, and acts
again.

Result:

```text
status: complete_with_known_limits
live actions executed: 10
replan cycles attempted: 3
replan cycles completed: 3
post-execution state classified: true
final state classification: MATCHED_KNOWN_STATE_CLASS
final state class: S0
controlled position source: live_public_event_tracker
learned progress used for live selection: true
candidate sent to actuator: true
counterfactual fixture selection only: false
live candidate generation from classified state: true
stop reason: MAX_REPLAN_CYCLES
levels completed: 0
forbidden source findings: []
```

The important change is that the agent no longer spends only one selected chain
and stops. It repeatedly performs this loop:

```text
live observation
→ option-state classification
→ live candidate generation
→ learned progress scoring
→ selected candidate actuation
→ fresh live observation
```

The run still does not solve LS20. It completes no public level. The claim is
narrower: the agent has a bounded live replanning loop over its existing small
option graph.

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-106-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-106-events.jsonl`
- `experiments/artifacts/0001-ls20-observation-option-agent/live-state-classifications.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/live-replanning-ledger.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/live-candidate-actuation.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-106-freeze.md`

## Milestone 11: frontier-guided live option discovery

M11 addresses the next bottleneck after M10.6: the live replanner can keep
steering, but the option graph is still small enough to cycle locally. The
agent now detects local saturation, generates scored frontier probes from the
current classified live state, executes a selected probe with public actions,
promotes a stable self-observed actor-region transition into a new option/state
edge, and shows the expanded graph in a later candidate-generation step.

Result:

```text
status: complete_with_known_limits
saturation detected: true
saturation reason: local_cycle_detected
frontier probe count: 4
selected probe: probe_003
probe executed live: true
live actions executed: 2
probe outcome: NEW_STATE_CLASS
new state classes created: 1
new options created: 1
new edges created: 1
noise promoted to option: false
expanded graph used in replanning: true
post-expansion replan cycles: 1
levels completed: 0
forbidden source findings: []
```

The induced option is `self_option_002`, with `source = frontier_probe`. Its
evidence is a live public probe transition, not a human trace segment. The
probe execution used public actions `[4, 1]`; the public tracker observed actor
movement from `[46, 36]` to `[41, 43]`. Volatile visual novelty was explicitly
recorded as not promoted.

The run still does not solve LS20 and does not infer a final objective. The
claim is narrower:

```text
M11:
  I can notice my current option graph is exhausted and expand it.
```

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-11-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-11-events.jsonl`
- `experiments/artifacts/0001-ls20-observation-option-agent/frontier-probe-candidates.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/frontier-probe-outcomes.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/graph-expansion-ledger.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/live-option-induction-ledger.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-11-freeze.md`

## M11.5: live graph-growth diagnostic

M11.5 is not a milestone claim. It embeds the M10.6 live replanner and M11
frontier-expansion mechanism in a longer public LS20 run to test whether the
agent can keep growing graph territory when unleashed with a larger budget.

Result:

```text
status: diagnostic_terminal_reached
live actions executed: 22
cycles attempted: 10
frontier probes: 7
promoted options: 7
state classes: 3 → 10
edges: 2 → 9
options: 2 → 9
post-expansion state rebindings: 7
expanded graph used count: 0
unique selected probes: 1
repeated probe selection count: 7
levels completed: 1
final state: NOT_FINISHED
stop reason: TERMINAL_STATE
forbidden source findings: []
```

The diagnostic found two signals:

1. Frontier probing is strong enough to make live public progress. Repeated
   probes expanded the graph and completed one public level.
2. Newly induced options are not yet reused as outgoing candidate choices. The
   agent rebinds into new state classes, but those states have no outgoing
   option candidates, so the policy falls back to another frontier probe. The
   same probe id was selected seven times due score ties and deterministic
   tie-breaking.

Interpretation:

```text
M11.5 shows live frontier expansion can produce progress, but the next
bottleneck is frontier-derived option reusability and probe diversity.
```

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-115-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-115-events.jsonl`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-115-graph-growth-ledger.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-115-probe-ledger.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-115-freeze.md`

## M11.6: level-transition context binding

M11.6 fixes the boundary-crossing probe problem exposed by M11.5. The agent
now detects a public level transition during live probe execution, slices the
probe at the transition action, prevents the cross-level delta from becoming a
same-level option edge, creates a new level-scoped context, carries reusable
machinery, invalidates stale spatial graph structures, binds the post-transition
observation, seeds new frontier probes, and continues acting inside the new
context.

Result:

```text
status: complete_with_known_limits
transition step: 21
trigger action: 1
level index: 0 → 1
original probe: [1, 4]
pre-transition actions: [1]
post-transition actions: [4]
cross-level delta promoted: false
new context: level_1_context
post-transition state: L1_S0
post-transition probe count: 4
post-transition actions executed: 1
levels completed: 1
forbidden source findings: []
```

Interpretation:

```text
M11.5:
  I can reach a new level.

M11.6:
  I know that a new level is a new context, not just a new state.
```

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-116-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-116-events.jsonl`
- `experiments/artifacts/0001-ls20-observation-option-agent/level-transition-events.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/level-context-ledger.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/transition-slicing-ledger.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/post-transition-seeding-ledger.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/terminal-evidence-ledger.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-116-freeze.md`

## M11.7: cross-level option-template transfer

M11.7 transfers option-making form across the level transition without
transferring stale spatial instances. It extracts `frontier_region_navigation`
templates from prior self-discovered frontier options, strips level-local facts,
binds a fresh level-1 option instance from the live post-transition observation,
rejects a deliberately stale prior-level spatial instance, validates a fresh
instance with public actions, promotes it into the level-1 graph, and shows the
promoted instance on the later candidate decision surface.

Result:

```text
status: complete_with_known_limits
template count: 6
template kind: frontier_region_navigation
template contains prior coordinates: false
template contains prior state ids: false
template contains prior edges: false
template contains prior observation hashes: false
source probe sequence used as policy: false
new level instance count: 1
instance bound from fresh observation: true
stale instance rejected: true
validation probe executed: true
validated template instances: 1
promoted new-level options: 1
promoted instance scored: true
template contribution detected: true
levels completed: 1
forbidden source findings: []
```

The stale negative control attempted to bind a prior-level target coordinate and
was rejected as `REJECTED_STALE_INSTANCE`. The fresh level-1 instance used
`live_public_event_tracker` evidence at `[36,43]`, validated with public action
`[4]`, reached `[35,61]`, and promoted `level_1:self_option_template_000`.

Interpretation:

```text
M11.6:
  I can survive a level transition as a modeling agent.

M11.7:
  I can reuse the kind of option I learned, not the old coordinates.
```

Known limit: this transfers the first spatial frontier-navigation template
family only. It is not cross-level strategy learning or full LS20 solving.

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-117-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-117-events.jsonl`
- `experiments/artifacts/0001-ls20-observation-option-agent/option-template-registry.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/template-instantiation-ledger.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/template-validation-ledger.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/cross-level-transfer-audit.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-117-freeze.md`

## M12: public goal-hypothesis formation

M12 adds the first explicit instrumental goal-hypothesis layer. It does not
claim the final LS20 objective. It turns public terminal, progress, and
cross-level template evidence into competing future-state predicates, selects a
bounded public test for one hypothesis, updates confidence from the public
outcome, and shows that the update changes later prioritization beyond M10
progress-only scoring.

Result:

```text
status: complete_with_known_limits
goal hypotheses: 4
selected hypothesis: goal_hyp_level_transition_000
target predicate: public_counter_transition(levels_completed_increases=true)
final goal claim made: false
goal test mode: TEST_HYPOTHESIS
goal test generated before execution: true
goal test public actions executed: true
hypothesis confidence updated: true
progress-only top candidate: candidate_progress_only_stable_semantic
goal-aware top candidate: candidate_goal_heldout_terminal_like
goal hypothesis changed selection: true
heldout candidate goal score changed: true
terminal action script promoted: false
noise promoted to goal: false
forbidden source findings: []
```

The selected hypothesis uses public level-transition evidence as support, but
does not promote the trigger action itself as a goal. The negative-control
hypothesis for volatile visual novelty remains `SUPPRESSED`. M12 also keeps
progress features and goal predicates separate: progress features support
hypotheses, while the hypotheses themselves are predicates over future public
state.

Interpretation:

```text
M10:
  I can learn what kind of progress is worth pursuing.

M12:
  I can form and test a hypothesis about what kinds of future states are worth pursuing.
```

Known limit: M12 forms instrumental public hypotheses only. It is not final
objective discovery, long-horizon pursuit, or full LS20 solving.

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-12-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-12-events.jsonl`
- `experiments/artifacts/0001-ls20-observation-option-agent/goal-evidence-ledger.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/goal-hypotheses.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/goal-test-candidates.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/goal-hypothesis-updates.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/goal-selection-ledger.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-12-freeze.md`

## M12.6.1: public motor-constraint learning

The replay viewer exposed that the M11.7 first-level transition was not clean
competence. The agent often chose actions that produced no controlled-entity
motion, then happened to hit the public level transition. M12.6.1 adds the
smallest corrective layer: learn action-effect constraints from public
controlled-position deltas and make selection avoid repeated no-op actions.

Result:

```text
status: complete_with_known_limits
public action-effect observations: 21
public no-op count: 6
public moved count: 15
public level-delta count: 1
constrained action count: 1
constrained action: action 3 in public_context
constrained action support: 4
constrained action no-op count: 2
constrained action no-op rate: 0.5
baseline selected candidate: z_candidate_repeated_no_op_risk
motor-aware selected candidate: a_candidate_observed_motion
selection changed due to motor constraints: true
repeated no-op candidate deprioritized: true
forbidden source findings: []
```

Interpretation:

```text
M12.6.1:
  I can tell when my own actions are failing locally, learn a public action-effect constraint model, and stop selecting repeated no-op actions as progress.
```

Known limit: this is a public action-effect regularity, not object identity,
terminal causality, or a general planner. It deliberately does not name the
cause of no-op outcomes.

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-1261-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-1261-events.jsonl`
- `experiments/artifacts/0001-ls20-observation-option-agent/motor-action-effect-audit.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/motor-aware-selection-ledger.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-1261-freeze.md`

## M12.R0: public factor discovery assay

The representation branch pauses planner growth and asks whether public
interaction traces can be compressed into reusable object/action factors. R0
does not name object categories. It extracts provisional factors from rendered
frames and public action effects, then compares factor-state prediction against
raw observation-hash prediction.

Result:

```text
status: complete_with_known_limits
transitions: 23
actor factor count: 15
stable factor transition count: 23
changed factor transition count: 21
raw state count: 23
factor state count: 8
compression gain: 0.6522
raw context reuse count: 0
factor context reuse count: 10
raw prediction accuracy: 0.0
factor prediction accuracy: 0.6
prediction gain: 0.6
forbidden source findings: []
```

The factor state key is deliberately coarse: public status plus a histogram of
component values and area buckets. This is a representation pressure, not a
semantic ontology. On the M11.7 replay it reuses contexts that raw frame hashes
cannot reuse, and gives a prequential local action-effect predictor nonzero
coverage and accuracy.

Interpretation:

```text
M12.R0:
  I can extract public object/action factors from interaction evidence well enough to predict local effects better than raw frame identity.
```

Known limit: this is not full object discovery. The changed-factor count is a
proposal surface from tracker/component matches, not causal attribution. R0 does
not infer terminal causality or solve LS20.

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-12r0-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-12r0-events.jsonl`
- `experiments/artifacts/0001-ls20-observation-option-agent/factor-candidates.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/factor-transition-predictions.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/factor-compression-audit.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-12r0-freeze.md`

## M12.R1: public factor counterexample assay

R1 asks whether reused factor contexts are actually reliable enough to promote.
It does not add new object names or game-specific roles. It groups public
factor-state/action contexts from R0, counts observed outcomes, promotes only
contexts with repeated stable outcomes, and records contradictory contexts as
counterexamples against the current factor vocabulary.

Result:

```text
status: complete_with_known_limits
transitions: 23
contexts: 13
reused contexts: 5
stable contexts: 1
contradicted contexts: 4
unsupported contexts: 8
promoted contexts: 1
blocked promotions: 4
counterexamples: 4
no-op contradictions: 2
forbidden source findings: []
```

Interpretation:

```text
M12.R1:
  I can use public interaction counterexamples to decide which factor contexts are reliable enough to promote and which show that my factor vocabulary is under-specified.
```

The main signal is negative but useful. R0's compression created reuse, but on
the M11.7 replay most reused contexts are not reliable. Four of five reused
factor contexts contradict themselves. Two contradictions are direct no-op
contradictions: the same coarse factor context/action sometimes moves the actor
and sometimes does nothing. One contradicted context even mixes ordinary actor
motion with a status increase. This says the current factor key is missing
load-bearing distinctions, likely local geometry/adjacency and contact relation
features, not that the planner should trust majority outcomes.

Known limit: R1 diagnoses representation debt; it does not repair the factor
representation. The correct next cheapest check is to add one candidate
refinement pressure at a time and require it to reduce contradicted contexts
without destroying reusable stable contexts.

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-12r1-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-12r1-events.jsonl`
- `experiments/artifacts/0001-ls20-observation-option-agent/factor-counterexamples.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/factor-stability-audit.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-12r1-freeze.md`

## M12.R2a: actor-centric local-geometry repair assay

R2a tests the first narrow representation-repair pressure after R1:
pre-action, action-relative local geometry around the tracked/control candidate.
It does not name walls, doors, terminals, or object roles. It adds only a small
frontier geometry key to the R0 factor key and compares it against an absolute
coordinate-bucket negative control.

Toy assay result: local geometry repairs a constructed movement/no-op alias.
Two open-front transitions and two blocked-front transitions share the same R0
factor/action context. R1 marks them contradicted. R2a splits them into two
stable reused local-geometry contexts, while exact coordinate buckets fragment
into unsupported singletons.

Replay result:

```text
status: partial_refinement_fragmented
transitions: 23
raw states: 23
base reused contexts: 5
base stable contexts: 1
base contradicted contexts: 4
local geometry states: 22
local geometry reused contexts: 1
local geometry stable contexts: 1
local geometry contradicted contexts: 0
local geometry prediction accuracy: 1.0
coordinate control states: 19
coordinate control reused contexts: 2
coordinate control stable contexts: 1
coordinate control contradicted contexts: 1
coordinate control prediction accuracy: 0.5
resolved aliases: 0
terminal aliases split: 1
forbidden source findings: []
```

Interpretation:

```text
M12.R2a:
  I can test whether pre-action actor-centric local geometry repairs movement/no-op aliasing without collapsing back to raw-state memory.
```

The real replay does not satisfy the R2a acceptance criteria. Local geometry
eliminates contradictions, but does so by making `22` refined states from `23`
raw states and leaving only `1` reused context. That is not repair; it is
fragmentation. The alias ledger is still useful: it shows that local geometry
can split R1 counterexamples, including the terminal alias, but the splits are
mostly unsupported singletons rather than reusable factor contexts.

This result blocks the tempting conclusion that "local geometry solved walls."
It did not. The next cheapest check is to make the local geometry key less
positional and more relational: keep action-front openness/blockage, but remove
or coarsen distinctions that encode absolute local layout. If that still fails,
R2b should add contact/adjacency relation factors rather than more local pixels.

Artifacts:

- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-12r2a-summary.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-12r2a-events.jsonl`
- `experiments/artifacts/0001-ls20-observation-option-agent/local-geometry-repair-audit.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/alias-resolution-ledger.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/coordinate-control-audit.json`
- `experiments/artifacts/0001-ls20-observation-option-agent/milestone-12r2a-freeze.md`
