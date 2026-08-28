# One Tycho Nooa Solver

Accepted for the unified integration line based at
`fa190780ac20dd50b3b45a9ec7b63abbfda8bd79`.

## Context

The repository had one host protocol and one worker entry point, but several
worktrees looked like competing solvers. They contained four different things:

- the production runtime;
- behavior configurations and candidate mechanisms;
- experiment launch, replay, and audit code;
- historical implementations and evidence.

Directory names and runnable entry points did not distinguish those roles. In
one incident, a current Solver audit ran against the old 0159 raw Muse harness
because that harness was available. The result belonged to the wrong program.

The clean repair line at `fa190780` supplies the integration base. Its root tree
is `d7ef189d2856f8f2751faf02f2dc02ed45af56b1`, byte-identical to the separately
reviewed `f7a47c5d` tree. Dirty worktrees remain evidence sources. They are not
merge targets.

## Decision

The Tycho Nooa Solver has one production path:

```text
arc-host
  -> JSONL protocol v3
  -> tycho-nooa-worker
  -> WorkerSession
  -> one persistent TychoNooaAgent
  -> NativeToolStrategy
  -> one ActionDecision
```

An ordinary run uses `trigger/control` unless an experiment explicitly binds a
different candidate. This default is the incumbent. It is not a claim that all
other configurations are inferior.

Every source module has one disposition in
`solvers/tycho_nooa/MODULE-DISPOSITION.json`:

- `kernel`: required to implement or protect the worker contract;
- `incumbent`: behavior used by the ordinary Solver;
- `challenger`: a candidate mechanism imported by the runtime but not part of
  the ordinary control;
- `research`: experiment, replay, compatibility, or analysis machinery that the
  worker cannot reach;
- `historical`: a frozen implementation retained for old evidence.

A source-level test recomputes the worker's transitive internal imports. It
requires every reached module to be `kernel`, `incumbent`, or `challenger`. A
research or historical import into the worker fails the test.

## Candidate selection

A new mechanism does not become part of the Solver because it is implemented or
available. Its experiment must identify the incumbent behavior it challenges,
keep unrelated conditions fixed, and define what counts as markedly better.

The first gate is uptake. If the model did not use the mechanism at the relevant
judgment point, the run says nothing about its quality. After uptake, behavior
must improve under the precommitted measure without weakening protocol,
authority, trace, sandbox, or spend controls.

A winning challenger replaces the challenged incumbent behavior. The old
behavior becomes research or historical material. The runtime must not
accumulate several co-equal ways to make the same decision.

## Source ownership

Production behavior belongs under `solvers/tycho_nooa`. The host remains
ignorant of Tycho, NOOA, providers, prompts, modes, and experiment identities.
Experiments may bind a Solver revision and configuration, but the Solver must
not import their launchers, cards, authorizations, or results.

The dirty core, main, and 0161 worktrees remain unchanged. Local recovery
bundles preserve their patches and untracked files before this integration.
The still-running GLM Arm A v4 directory is excluded from the first core source
snapshot and must receive a complete snapshot after its producer closes.

## Consequences

- `fa190780` is the fixed comparison point for unified changes.
- Whole-branch merges from live micro or dirty core are prohibited. Useful
  behavior is ported through a failing test onto the unified line.
- Safety properties such as credential isolation, read-only generated Python,
  host-owned planner publication, and lossless trace capture apply to every
  candidate.
- Typed candidate goals and decision discipline remain challengers until their
  own comparisons justify promotion.
- On-policy and interface-audit launch policy stays outside production imports.
- Historical files and worktrees are not deleted during this integration.
- Physical movement of research modules waits until source commitments and
  evidence consumers are inventoried. Import closure gives the production
  boundary now without rewriting provenance.

## Rejected alternatives

### Merge every worktree

Rejected because live micro predates later trace and authority repairs, while
dirty core mixes reusable source with one-off experiment policy and growing run
artifacts. A branch merge would erase the reason for selecting each change.

### Use dirty core as the base

Rejected because its tests and runtime bundle change while experiments run.
The clean repair line already preserves its validated continuity, trace, Gate,
and typed-evidence repairs.

### Use live micro as the base

Rejected because it contains valuable sandbox, planner, and context work but
lacks the final repair stack. Those capabilities are smaller and safer to port
than the later repairs are to reconstruct.

### Delete research modules now

Rejected because old manifests and run evidence may bind their exact bytes.
The worker import boundary can be enforced without changing those files.
