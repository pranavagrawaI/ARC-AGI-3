# Frozen evidence-query parity — 2026-08-23

## Judgment

**Supported within the tested domain:** the sterile `evidence.query` path is
value-equivalent to the frozen workspace `wmlib.py` path for ordinary
current-attempt frames and transitions.

This is an interface result, not evidence that CodeAct improves reasoning or
gameplay.

## Contract

For the same frozen workspace, the two paths must produce identical:

- current evidence frontier;
- ordered frame grids and `(level, turn)` keys;
- ordered transition grids, actions, rows, and columns;
- derived per-frame color counts;
- derived per-transition changed-cell counts and bounding boxes.

The harness hashes the canonical JSON values independently in T1 and inside a
real sterile T2 invocation. It records every mediated capability call. A
positive control changes one cell in a copied legacy frame and must make the
same comparator report a mismatch.

## Results

| Frozen checkpoint | Frontier | Frames | Transitions | T2 calls | Result |
|---|---:|---:|---:|---:|---|
| AR25 turn 0 | L0/T0 | 1 | 0 | 2 | passed |
| AR25 turn 18 | L0/T18 | 19 | 18 | 2 | passed |
| Synthetic pagination control | L0/T39 | 40 | 39 | 4 | passed in pytest |

Both frozen checkpoints passed raw-value and derived-analysis hashes with no
mismatches. Both positive controls were detected. The synthetic control forced
two frame pages and two transition pages, establishing that frontier-bound
pagination does not omit, duplicate, or reorder records.

Exact hashes and checkpoint identities are preserved in `report.json`.

## Reproduction

From the repository root:

```bash
env -u TMP -u TEMP \
  TYCHO_SANDBOX_RUNTIME=host \
  PYTHONPATH=solvers/tycho_nooa/src \
  solvers/tycho_nooa/.venv/bin/python \
  -m tycho_nooa.evidence_parity \
  --manifest analysis/codeact-evidence-parity/20260823/checkpoints.json \
  --output analysis/codeact-evidence-parity/20260823/report.json \
  --repo-root .
```

The pagination and real-checkpoint controls are also executable with:

```bash
cd solvers/tycho_nooa
env -u TMP -u TEMP TYCHO_SANDBOX_RUNTIME=host \
  .venv/bin/pytest -q tests/test_evidence_parity.py
```

## Boundaries

- Both real checkpoints are AR25 frontiers and contain the same frozen wmlib
  revision. This does not establish cross-game or historical-wmlib transfer.
- The present capability intentionally omits archived attempts, death events,
  terminal events, and animations; parity for those channels is **untested**,
  not failed.
- The harness establishes information and computation parity, not that an LLM
  will request the right evidence or use it well.
- Native tools and CodeAct have not yet been compared as live Actor treatments.

## Consequence

The next architectural step may now add T1-generated NOOA execution events.
Actor shadow integration should still wait until those events preserve cell,
capability, result, and return-validation provenance.
