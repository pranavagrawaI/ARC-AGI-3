# Bounded planning and prospective probes

This local experimental overlay adds a model-owned graph search helper and a prediction-before-action operation to the hash-pinned B1 bugfix source. It does not change the incumbent or deploy a solver. Source provenance is in `provenance.json` and `NOTICE.md`.

Use the existing CPython 3.12 environment, which already contains the harness dependencies:

```bash
solvers/tycho_nooa/.venv/bin/python -B solvers/m2_planning_tools/qualify.py
```

`build.py DESTINATION` composes a fresh local source tree. It refuses an existing destination and verifies the parent before copying. Set `ARC3_PLANNING_VARIANT` before importing that tree: `control`, `planner`, `probing`, or `both`. The default candidate variant is `planner`.

Search takes pure model-authored successor, state-key, goal and optional heuristic callbacks. `FOUND` certifies only a path in the supplied model; check real outcomes during execution. `HORIZON_EXHAUSTED`, `GRAPH_EXHAUSTED`, `RESOURCE_LIMIT`, and `CALLBACK_ERROR` have separate meanings. Cooperative search timing cannot interrupt a hung callback; the enclosing sandbox deadline does.

```python
plan = bounded_search(start, successors, state_key=key, is_goal=goal,
                      heuristic=estimate, max_depth=40, max_seconds=10)
```

A probe uses one action and an unused saved-data name. Callbacks receive the current frame, a JSON-compatible before-state and the normalized action. The host acknowledges storage before the real action runs.

```python
check = probe_transition({'action': 'MOUSE', 'row': row, 'col': col},
                         observe=read_state, predict=predict_successor,
                         state_key=key, name='probe_001')
```

`MATCH` and `MISMATCH` compare only the state variables represented by `observe` and `state_key`. `TERMINAL` avoids comparing a restarted/new-level board against the prior model. `REFUSED` and `ACTION_UNCONFIRMED` never count as agreement. On interruption, saved `PREDICTED` means execution is unconfirmed, not necessarily absent; `ACTED` has an acknowledged receipt without a completed comparison. Inspect receipts and state before any further action.

The factorial continuation runner and evidence are in `analysis/planning-transfer-20261014`. It restores historical contexts with archived replies, verifies request/tool/prefix fidelity, then changes sandbox dispatch and the named system addendum at the branch. Restoration is the default mode. Explicit live mode connects to a separately provisioned endpoint; the runner does not start a server or launch a GPU. CPU fixtures prove wiring, not model adoption or score efficacy.

```bash
solvers/tycho_nooa/.venv/bin/python -B analysis/planning-transfer-20261014/run_matrix.py \
  --output /tmp/planning-restoration
```

The 16-episode manifest pairs two archived checkpoints and two seeds across four variants. A future inference screen has a maximum 524288 generated tokens, 32768 per episode, 32 calls per episode, 900 seconds per episode and a 14400-second matrix deadline. No failed/ambiguous episode is retried. GPU admission and launch require their own bounded experiment card; none is performed here. The audit rejects fixtures, empty coverage, unresolved exposure and native replay disagreement. Two completion gains and no losses across four valid pairs only pass a checkpoint screen; full-game matched score validation remains necessary.
