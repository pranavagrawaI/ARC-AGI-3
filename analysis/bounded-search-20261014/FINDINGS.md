# General bounded-search implementation and frozen-model screen

**The implemented helper converts the correct-enough reminder model into native completion, but guided search does not recover the other five frozen models under the same caps.** The utility is exposed through the real Python sandbox in `solvers/m2_search`, with model-owned successor/key/goal/heuristic callbacks and no game priors in the helper.

## Outcome

| Frozen model and actual final state | Unguided helper | Guided helper | Native completion |
|---|---|---|---|
| b1-1701-control | Node cap, 12.33 s | Node cap, 11.55 s | Neither |
| b1-2903-control | Node cap, 11.46 s | Node cap, 8.87 s | Neither |
| b1-1701-reminder | Node cap, 14.14 s | Node cap, 11.91 s | Neither |
| b1-2903-reminder | Time cap, 20.00 s | 29-action route, 8.48 s | Guided, 29/29 successors match |
| b1-1701-counterexample | Node cap, 11.23 s | Node cap, 9.48 s | Neither |
| b1-2903-counterexample | Node cap, 11.63 s | Node cap, 9.92 s | Neither |

The paired arms share each model, native action prefix, start, action order, goal and resource limits. Zero heuristic versus model-derived Manhattan guidance is the contrast. The known reference route and successful-run reference simulator are absent from both arms. Limits:20 s,100,000 stored nodes,100,000 expansions,depth60; absolute process cap40 s. All12 arms execute; total measured search time141.0 s. These are six selected states/models from one game, not a general success-rate estimate.

The positive guided arm expands23,562 states, discovers42,035 unique keys, and stores42,965 records including reopens. Counting stored nodes separately keeps the memory cap honest. The returned path predicts all29 native transitions and completes L2.

## What the failures tell us

A separate post-screen diagnostic uses the already-archived successful contestant model as an analyst reference. It finds28–31-action recovery routes from all six states and validates every transition in the native engine. The frozen generated models are then checked along those routes, without supplying the routes/reference rules to any screen arm or to the solver.

| Frozen model | Native-valid witness length | Model matches | First wrong consequence |
|---|---:|---:|---|
| b1-1701-control |29|9|Orange growth:yellow/green position mismatch |
| b1-2903-control |28|9|Yellow shrink:green position mismatch |
| b1-1701-reminder |31|23|Yellow shrink:green position mismatch |
| b1-2903-reminder |29|29|None on this route |
| b1-1701-counterexample |29|23|Yellow shrink:green position mismatch |
| b1-2903-counterexample |31|23|Yellow shrink:green position mismatch |

This supports concrete remaining rule-model defects and demonstrates physically feasible recovery within the horizons. It does not isolate those defects as the sole cause of each node-cap failure, prove that the defective models contain no valid route, or justify blind increases in search budget. Guided planning and prospective coverage of untested action families remain distinct research questions.

## Implementation and verification

One public operation hides heap traversal, stable predecessor records, shorter-path reopening, tie breaking, node/expansion/time/depth limits, and diagnostics. FOUND certifies only the supplied model. Horizon, graph, resource and callback outcomes remain separate. Arbitrary heuristics are allowed as priority guidance without claiming shortest routes. Callbacks must be pure and keys must identify equivalent future behavior; the host does not supply an ontology or game rules.

The sandbox blocks actions and nested helper calls during callbacks, then releases the guard on every exit. User wrappers survive subsequent calls via the inherited function cache. A supported new integration bug was found and repaired during review: if snippet code shadows bounded_search, a dependent wrapper must not be retained and silently rebound to the real helper next turn. The regression fails before the fix and passes afterward.

Qualification runs47 behavioral tests, including200 independent random-graph BFS checks, plus12 build/receipt/evidence/pin/package/integration checks. The package installs a new utility file from the donor tree, verifies all source hashes, executes the real ToolAgent search/retention gate against installed source, rejects drift before mutation, and applies the source patch with real Git. The full hash-protected bugfix parent remains unchanged. The search-risk estimate uses the explicit fixed test universe with no new dependency or threshold; it is not a claim of general code quality or reduced complexity.

An analyst-authored program also runs through the actual sandbox helper and action IPC, executes29 real native actions, matches every successor and clears L2. This checks the complete implemented capability path. It does not demonstrate that an LLM independently chooses the helper or authors valid callbacks.

Artifacts:paired raw metrics in results/, native-receipts.json, reference-diagnostic.json, sandbox-native.json, cases.json, manifest.json and verification.json. The successful route and diagnostics are test-only artifacts; they are absent from the solver package and prompts.

## Decision

Retain as an isolated, locally qualified candidate. Do not promote on this retrospective screen. No model calls, training, GPU launch, submission, queue mutation or incumbent replacement occurred.

The next discriminating live test is interface uptake under the same frozen core model: does it supply a useful heuristic, use the helper, and execute a checked route at a matched total budget? Separately, compare ordinary continuation with a prospective prediction/test of the missing yellow-shrink coupling. Success needs completed play; invocation counts and historical agreement are mechanism evidence only.
