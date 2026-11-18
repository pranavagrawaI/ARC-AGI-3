# Learned-model planning qualification

**A* found a native-valid 29-action recovery using the agent's frozen learned simulator.** All 29 predicted successors matched the engine, and L2 completed. The two deeper breadth-first variants exhausted their fixed time budget. This is a retrospective, analyst-assisted planning result on one checkpoint; it is not a live model intervention or a competition score improvement.

| Search arm | Result | Expanded | Discovered | Search seconds |
|---|---|---:|---:|---:|
| Archived BFS, depth 24 | No route within horizon | 158,413 | 211,935 | 48.64 |
| Same generated BFS, depth 40 | Time cap | 287,503 | 373,137 | 90.00 |
| Incremental BFS, 24 then 28 | Time cap | 283,533 | 368,161 | 90.00 |
| A*, depth at most 40 | Found 29-action route; native completion | 23,562 | 42,035 | 7.25 |

## What was held fixed

The actual archived b1-2903-reminder call-8 search code, its starting state, action order, goal predicate, static obstacles, state-key function, and final retained sim_step were frozen. The depth-24 arm reproduced the original stdout exactly, including best-distance/path output and the original 211,935/158,413 counts. Each arm used the same model-file hash. The actual native prefix was replayed and its state matched the generated planner's original setup.

The depth-40 arm changes only the original depth comparison. Incremental BFS retains its frontier while extending the horizon; it does not introduce true game dynamics. A* is an analyst-written replacement with predecessor links and the agent's existing Manhattan-distance notion. Its heuristic is ceil(distance-to-the-agent-selected-goal / 3), using the frozen model's three-cell action increments. The known reference route and four-length reference simulator were not supplied to any planner. Native-engine source was used only to restore the prefix and check returned actions.

The learned model's other known transition errors remain. This result demonstrates sufficiency for the returned route, not complete correctness of its reachable graph. No conclusion about BFS impossibility follows from a resource cap. These timing observations are single runs on this host; no general speedup or benchmark effect is estimated.

## Verification

- Exact baseline reconstruction, not merely agreement on a no-route label.
- Frozen runner/card/model hashes and protected source hashes checked after completion.
- All four arms respected 90 seconds, 300,000 expansions and 500,000 discovered states; supervisors allowed at most 110 seconds per process.
- Native receipts contain all 29 before/predicted/actual successor states, frame hashes and final levels_completed=2.
- Nine independent runner checks exercise a literal 26-edge toy chain, horizon extension, state caps and a zero-time deadline. Resource limits remain separate from no-path claims.
- All 81 files in the qualified bugfix successor's verification inventory remain unchanged.
- No provider calls, GPU use, training, submission, queue change or solver deployment.

Raw outcomes: `results/*.json`; native outcomes: `native-receipts.json`; provenance: `manifest.json`; runner checks: `runner-validation.json`; summary: `RESULT.json`.

The v1 attempt failed before any search because the loader rejected sim_step's import copy. Its source/card/log are preserved in the sibling directory. V2 accepts that exact existing import; learned functions, original search, action prefix and experiment limits were unchanged.

## Interpretation and next test

The immediate failure is not only the depth-24 horizon: raising it alone did not recover under the tested budget. Search strategy is an independently demonstrated bottleneck here. A simple guided search converts the already-learned model into completed play without additional rule feedback.

Next, qualify a game-general bounded search helper with model-authored successor, key, goal and heuristic callbacks; preserve FOUND, horizon exhaustion, graph exhaustion and resource-limit outcomes separately. Test additional frozen learned models/checkpoints before a live comparison. The host must not supply s5i5 object names, coordinates, action rules or the analyst's route. Only then test whether the agent adopts the helper and whether equal-budget completion improves. Prospective action-family coverage remains a separate necessary investigation for incorrect models.
