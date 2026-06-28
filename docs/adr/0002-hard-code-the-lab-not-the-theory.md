# Hard-Code the Lab, Not the Theory

Accepted.

ARC-AGI-3 contestant experiments may use concrete, game-general machinery for
observation transport, memory, bookkeeping, uncertainty, graph search, option
execution, verification, failure detection, and replanning. They must not turn
human-authored game ontologies into permanent contestant logic.

Concrete mechanisms are scaffolding and instrumentation. Learned hypotheses own
the world model.

## Decision

Keep algorithmic:

- admissible observation and action interfaces;
- history buffers, event logs, and transition graphs;
- candidate generation and object tracking;
- uncertainty and evidence accounting;
- search over a learned model;
- execution, verification, failure detection, and replanning;
- reproducible evaluation and artifact generation.

Learn from admissible interaction:

- which visual proposals behave as persistent entities;
- which latent variables explain observed changes;
- which changes are action-caused;
- which entities or relations have useful affordances;
- which observations predict progress or terminal outcomes;
- which experiments, options, and routes are worth attempting.

Human-readable labels such as resource, hazard, gate, changer, key, or goal may be
used for analysis. They are not contestant runtime concepts unless an experiment
is explicitly reported as human-supervised.

Generic geometric and temporal measurements may produce candidate features or
pseudo-labels. They are evidence, not ground truth. A connected component is a
region proposal, not an asserted object. A conditional visual transformation is
an affordance hypothesis, not a named mechanic.

## Consequences for 0001

`0001` is an admissible, instrumented scaffold whose purpose is to prove and log
the observation-only learning loop. Its parser, tracker, event differencer,
contact geometry, and conditional-rate tests are data-refinery mechanisms.

No further LS20 mechanic detectors should be added to `0001`. In particular, do
not add dedicated clock, changer, reset-region, trap, or goal rules.

The next model-bearing experiment should consume neutral histories and learn a
generic conditional transition prediction:

```text
history + candidate entity/relation + action
    → distribution over future visual/state changes
```

It must be evaluated against simple concrete baselines, on held-out temporal or
level splits, and by recovery after deviation rather than trace replay alone.

## Rejected Direction

Continuing to add LS20-specific recognizers and option rules would likely improve
one-game performance while obscuring whether abstractions transfer. That path is
rejected because it converts research scaffolding into a hand-authored ontology.

