# Separate Solvers From Experiments

Accepted. This decision supersedes the source-location consequence of
`0001-experiments-after-lessons.md` and the integration direction in
`0003-executable-theory-lane-for-0159.md`.

## Context

The numbered `experiments/agents` directories accumulated two different kinds
of identity:

- a Solver Candidate and its implementation lineage;
- an Experiment and its chronological research question.

That made a new experiment inherit the previous experiment's directory and then
modify it while placing its card, specification, analysis, and artifacts in
other numbered directories. The result preserved chronology but destroyed
locality: understanding one mechanism required reconstructing several folders,
and changing one experiment silently changed the implementation used by later
ones.

The 0159--0161 line also kept NOOA's persistent actor while replacing Tycho's
native workspace-tool loop with progressively larger CodeAct broker surfaces.
Those surfaces were available but were not used by the tested models. Adding
more mechanisms inside the same interface did not recreate Tycho.

## Decision

Solver implementation and experimental evidence are different modules.

### Stable ARC host

The Python 3.14 environment boundary lives in `/arc-host`. It owns the ARC SDK,
public-frame projection, action validation and SDK translation, episode stopping,
and host-visible evidence. It knows nothing about NOOA, Tycho, providers,
prompts, or experiment numbers.

Solver processes communicate with it through one versioned persistent JSONL
protocol. This keeps the Python-version boundary and the authority boundary in
one small module instead of reimplementing a host inside each experiment or
solver family.

### Stable solver modules

Runnable source lives in `/solvers/<solver-family>`. A solver-family directory
is the one place required to understand, build, and test that implementation.
It may depend on pinned external projects such as NOOA or Tycho. It must not
import a numbered experiment.

The first family is `/solvers/tycho_nooa`. Its target is deliberately literal:
Tycho's one-growing-conversation, native workspace-tool loop implemented as a
NOOA generation strategy. It starts from the smallest tracer bullet that keeps
the decisive interaction form:

1. the same actor survives for the game;
2. observations and tool interactions remain in one event history;
3. workspace files survive across turns;
4. file, Python, and action operations are native model tools;
5. `take_action` commits exactly one environment action and ends the turn.

Optional builders, planners, compactors, and forensic machinery are absent
until this loop works. They are additions to a stable solver module, not new
copies of it.

### Thin experiments

An Experiment owns one question, its precommitted contrast, candidate manifests,
and its results. It does not own solver implementation.

A candidate manifest freezes a solver source revision plus configuration. An
experiment arm references a candidate manifest. A later experiment may reuse a
candidate manifest, but no experiment imports another experiment's Python.

The locality tests are:

- **behavior:** `/solvers/<family>`;
- **candidate identity:** one manifest;
- **research question:** one experiment directory/card;
- **observed result:** one immutable run artifact.

Chronological numbers identify studies and runs only. They do not encode module
inheritance or architecture lineage.

### Historical material

Existing numbered implementations and artifacts remain unchanged as research
evidence. They are legacy inputs to forensic analysis, not the location for new
solver code. No mass migration is authorized by this ADR.

## Consequences

- New production/runtime code must not be added beneath numbered experiment
  directories.
- `/arc-host` must not import a solver or an experiment.
- New solver modules must not import from `/experiments`.
- New experiment code must not import another experiment.
- A Mechanism Test changes one named mechanism or interface condition against a
  frozen candidate. A bundled rebuild is a Capability Build and must be labeled
  as such.
- Tycho and NOOA reuse is pinned and recorded in the solver module's
  `PROVENANCE.md`; experiments reference that module rather than copying it.
- `0003` remains a historical specification of the WM0 treatment, but its
  decision to retain the 0159 CodeAct actor is no longer the active integration
  direction.

## Rejected Alternatives

### Continue the numbered inheritance chain

Rejected because it makes chronology a dependency graph and forces one idea to
be understood through several directories.

### Move all historical files immediately

Rejected because it would rewrite evidence provenance before the replacement
has earned parity.

### Add another broker method to 0159

Rejected because the failed behavior is tool uptake, not the absence of one
more capability behind the same interaction form.
