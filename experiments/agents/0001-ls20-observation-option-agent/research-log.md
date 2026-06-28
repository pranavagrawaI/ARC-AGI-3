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
