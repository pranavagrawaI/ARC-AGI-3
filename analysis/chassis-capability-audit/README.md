# Chassis capability audit — Tycho → NOOA

Started 2026-09-04, after the symmetry library was found unreachable from every run
since the line moved to the NOOA chassis.

## Why this exists

`Questions and Ideas/sources/Tycho-NOOA Architecture - What Remains to Steal.md` §4.7
and P6 already name *"symmetry operations, planner diagnostics"* as Tycho capabilities to
carry over. The list was correct and nothing enforced it. q2b, q2c, q2d, q3a, q3b, q3c
launched against a chassis missing both, and no card noticed, because every card compares
the current chassis against its own previous run.

A list of things to steal cannot catch this. A check that runs against the live source can.
That is `reachability.py`.

## Method

Audit at the **point of use**, never the point of definition. A capability counts as present
only if the actor could reach it from inside a CodeAct cell in the chassis that actually
ran. Existing in `solvers/tycho_nooa/` proves nothing about the NOOA line: the two solvers
share no runtime.

Live NOOA source for the q3 series is `analysis/local-probes/20260904T-host-control-loop-q3/patched-source/after/`
(the reference clone was uncommitted and `experiments/agents/0158-.../third_party/source-snapshots/nooa/`
is stale — it has no `harness.py`). Audit against `after/`, not the snapshot.

## Findings, 2026-09-04

| Tycho capability | where it lives | reachable from the NOOA actor | note |
|---|---|---|---|
| `symmetry` — canon / congruent / transporter / stabiliser / orbit / primitive | `tycho_nooa/symmetry_template.py`, seeded by `trusted_runtime.py:28`, briefed in `runtime_notice.py:58` | **no** | 0 hits in live `solver_agent.py` + `harness.py`. Only appearance of the word anywhere the actor can read is one bullet in `grid-game-solver/SKILL.md` — a skill these runs do not use. |
| bounded planners — `plan_bfs`, `plan_astar`, `plan_subgoals`, `validate_plan`, all with `max_nodes` and `prune_game_over` | `tycho/workspace/wmlib_template.py:1543-1770` | **no** | 0 hits for `plan_bfs`/`astar`/`max_nodes`/`visited`. |
| segmentation — `segment`, `segment_summary` (connected components, bboxes) | `wmlib_template.py:110,180` | **no** | 0 hits for `segment`/`connected`. Actor gets `grid_array` and numpy and nothing above it. |
| `dynamics.key_counterexample`, `action_laws` | `tycho_nooa/dynamics_template.py:41,125` | **no** | 0 hits. |
| evidence queries — `death_events`, `terminal_events`, `solved_actions` | `wmlib_template.py:893,1024,1038` | **partial** | q3 added host-owned `observations()` and `level_terminal()`, which cover terminal frames and blocked pairs. Not the same API; the gap is closed in substance for terminals, open for death indexing. |
| scribe at `LEVEL_COMPLETE`; trigger-builder at `LEVEL_START` | `tycho/agent/agent.py:652`; `tycho_nooa/trigger_builder.py:494` | **partial** | q3c ported the scribe (P2 = 1.000). `LEVEL_START` re-check named as next port in q3c's RESULT, not done. |
| per-action animation frames | `wmlib_template.py:912,989` | **yes** | `trajectory(event_frames=...)`, `harness.py:712`. Not a gap. |
| exact provider-request checkpoints | `tycho_nooa/decision_checkpoints.py` | **partial** | `agent_logs/**/messages` holds the rendered prompts; not the JSON-safe request commitment. |

### The planner gap is load-bearing, not cosmetic

`interactive-game-solver/SKILL.md` step 3 instructs: *"search (BFS / greedy / best-first)
over action sequences in latent space."* The chassis supplies no search. So the actor writes
one in a cell, under a 60-second kill and a prompt line that says *"Keep cells short."*

- q2a hung 9.5 minutes inside one unbounded cell (`PRECOMMIT.md` Addendum 3), reproducing
  the failure the NOOA paper reports for 2 of its 25 games: *"ad-hoc, in-cell searches that
  lacked the bounds (max_depth, visited sets, node budgets) their own persisted planners
  carried."*
- q3c's model then hand-wrote `plan_to` in `helpers/game_e5a3ff.py:48` — `while q:`, no node
  budget, no depth cap, no game-over pruning. It survived only because the level-1 reachable
  set is 5×5.

`plan_bfs(model, start, target, max_nodes=20000, prune_game_over=True)` is the exact
counter-mechanism, written, tested, and one chassis away.

## What this audit does not license

Reachability is not value. Nothing here says a ported capability improves RHAE; the symmetry
uptake result is n=1 plus one same-configuration replication, on one game and one seed, in
the *other* chassis. Each port needs its own card and its own contrast.

The audit is also incomplete. Tycho has 131 modules under `solvers/tycho_nooa/src/tycho_nooa/`
and this pass checked the ones named in "What Remains to Steal" plus the runtime brief.
`verifier`, `optionality`, `deadlines`, `action_batch`, and the compatibility ladder are
unexamined.

## Standing rule

Run `reachability.py` against the live chassis source before freezing any card that assumes
a capability. A card that names a tool the actor cannot reach is measuring the wrong thing.
