# CPU fault-injection evidence, 2026-10-21

Harness: `run_competition_cpu.py` runs a notebook's own env, source-path, pickle, customization and run cells in
competition mode (`TRUE_SUBMISSION=True`, concurrency 120, per-game 31,920 s unless `--per-game-s`). It plays against the
audited taaf 110-clone competition simulator. The SGLang launch is replaced by `--llm dead|http500|http400`, or by
`--llm process`: `fake_llm.py`, run as a process under the notebook's own startup wait loop and, if present, the guard
supervisor cell.
Source tree: `build_replica.py` executes F1x's patch, overlay and F1 cells on the downloaded dfranzen bundle. All
overlay pre-hashes and the F1 hash matched.
Per-run artifacts: `runs/<name>/` (final.json, lifecycle.jsonl, guard-lifecycle.jsonl, filtered log excerpt).
Full raw outputs: `/tmp/f1x-faults` (not retained in the repo).
Fake actions are random arrows, so levels are not a gameplay signal.

## Production timing of a run that fails from the start (parent F1x, unguarded)

| run | fault | wall | games quit | levels |
|---|---|---|---|---|
| dead-prod | no listener on the model port | **12,863 s (3.57 h)** | 110/110 "analyzer failed 11 times" | 0 |
| 500-prod | every request HTTP 500 | **12,862 s (3.57 h)** | 110/110 | 0 |
| 400-prod | every request HTTP 400 | **1,107 s (18 min)** | 110/110 | 0 |

Mechanism: `ARC3_MAX_ACTIVE_STREAMS=10` holds a stream slot through each game's own 900 s first-request grace.
Connection errors and 5xx get that grace; 4xx skips it. So 11 waves × ~1,150 s ≈ 3.5 h, against ~100 s per wave for 4xx.
Consequence: D12 was COMPLETE within ≤ 2.2 h of submission, which includes queueing and the ≥ 12 min startup. That is
inconsistent with a server that was dead or answering 5xx from startup under the reproduced settings. It does not exclude
every startup fault or hidden-environment variation. It is consistent with a fast-rejection path
(4xx-type answers or game-side exceptions), or with a crash after some minutes followed by fast failures. Not established
on hidden data.

## Parent vs guard

The guard builds used here changed during review. Rows marked "pre-review" ran builds made before commit 731f63fb.
"current" is `kernel/f1x/m2-guard-f1x.ipynb` sha `38823f9f…`.

| run | notebook | fault | result |
|---|---|---|---|
| t2-parent | parent | SIGKILL of the server at 111 s, never restarted | 10 games quit "analyzer failed 11 times"; the other 100 were never served before the 900 s cap; silent zero; 7 games acted |
| t2-guard | pre-review | same kill | supervisor: server_exit −9 → relaunch 3 s later → server_ready 25 s later; 0 games quit; 36 games acted |
| t1-guard | pre-review (90 % of all games) | server dies 8 s after launch, every launch | 2 relaunches → server_abandoned; 80 quit, 30 never served; **end guard did not raise**: the defect that led to the served-games rule |
| t1-guard2 | pre-review (served rule, final-token count) | same | end guard raised "80/80 served games …, 0 tokens, no level" |
| h-parent | parent | healthy fake server, 600 s cap | 24 games acted, completed |
| h-guard | pre-review (final_generated_tokens only) | same | **end guard raised on a healthy run** ("0/0 served, 0 tokens"). End-to-end reproduction of the review blocker |
| h-guard-fixed | current | same | served 31, failed 0, tokens 122,350, completed, no raise. A random action completed 1 level, so this run does not exercise the zero-level branch; the real-record unit test covers that |
| guard-healthy | pre-review | healthy, 120 s cap | no raise; 11 games acted |

## Real-hardware screen (fault-screen-1, collected by the coordinator thread)

The original partial-collection result is preserved in `solvers/m2_guard/experiment/fault-screen-1/gate-evaluation.json`.
Selective terminal-console recovery on 2026-10-22 establishes G1 and G6, but the overall result remains NOT_QUALIFIED:
`gate-evaluation-recovered-20261022.json`. The console lifecycle exactly matches the retained lifecycle file, with no second
READY. The observed post-kill horizon ends before 900 seconds. No new GPU attempt or full output re-download was made.
Facts from the retained output:
- Target weight load took 1,997 s. First
  READY came at t = 2,343 s; the benchmark had been released at 722 s by the 12-minute deadline.
- Kill at t = 3,547 s, relaunch at 3,563 s. The relaunched server started its load with `avail mem=94.17 GB`, so process
  group cleanup supports the observed release of VRAM; successful HTTP port binding was not established. The retained
  log reaches 6/38 shards, 6m21s into the relaunched load. No second READY or explicit relaunch-failure event was collected
  before observation ended. This is censored, unqualified recovery evidence, not proof of successful relaunch.
- No game quit. With the parent's patience of 10, the first stream wave would have quit at ≈ 1,900 s, before READY at
  2,343 s. This is arithmetic on the observed times; no counterfactual was run.

Design implication: complete reload duration must be measured against failure patience and the remaining screen horizon.
The observed initial load took 1,997 seconds; the configured patience is described as roughly 17 minutes of failed turns.
The screen did not establish that a reload fits that tolerance. `DROP_CACHE_AFTER_LOAD` makes a cold reload plausible, but
the run did not measure cache state or establish a predictive reload-time distribution.

Independent review, 2026-10-22: the single 1,996.97-second initial load is verified. The owner's claimed 40-session rank/median
has no retained sample inventory in this packet and was not independently reproduced; it must not be used to size a new
budget as an established distribution. The cache explanation is source/config reasoning, not measured cache-state attribution.
The fixed healthy CPU run records one completed level, so zero-level behavior is established separately by the real-record
regression. Historical executed-notebook hashes were not recorded; matching the current notebook is not retroactive execution
provenance. Full review: `analysis/guard-review-20261022/review.md`.
