# Hybrid integration review — 2026-10-04

Candidate reviewed: untracked working hybrid, its generated adapters, inherited call chains and q4 evidence. The scope is integration/execution validity, not an exhaustive review of every canonical or donor module. Old candidate, tests and local evidence are preserved in revisions/local-v2-reviewed/snapshot.json. Frozen q4 artifacts/card were not changed.

## Supported findings and disposition

| Priority | Finding and evidence | Disposition |
| --- | --- | --- |
| P1 | IPC writes and reads restarted the same timeout. A real pipe reproduction took 1.701 seconds under a 1-second budget. | transport.py now shares one absolute deadline; regression passes. |
| P1 | Closing the worker left its Python child running. Real subprocess reproduction failed before the change. | Generated worker starts in its own process session; synchronized close kills remaining group members before closing inherited streams. Regression passes. Provider-side cancellation is not claimed. |
| P1 | compute.json write failure bypassed cleanup/bookkeeping and gate release. Injected failure reproduced stranded state. | Nested finally blocks close the worker, remove bookkeeping, finish game and release admission; concurrent task failure also drains remaining tasks. Regression passes. |
| P1, policy mismatch | q4 SB26 requested construction but received zero time. Ordinary Builder allocation also expired 240 seconds after decision start even when substantial host time remained. | v3 uses invocation-time allocation, capped at 240 seconds, with Actor reserve min(120 seconds, half remaining usable time) and 5 seconds for host return. This is a declared policy change, not a behavior-preserving fix. Late-request arithmetic and actual 60-second construction/repair fixture pass. |
| P2 | Expected game-budget expiry was classified like an early model-call failure. | Explicit game_deadline receipt and gave_up game state retain the timeout and unknown provider usage; earlier analyzer failure still crashes. Tested. |
| P2 | candidate-config always reported gate=10 and omitted exact selected games; old comparison documents retained public25 despite the latest instruction. | Report effective gate, games, concurrency, passes and allocation. Add small mode: fixed three games, 2400 seconds/game, no action ceiling. Update plan/draft. |
| P2 | Virtualenv under working output expanded the remote inventory and contributed to slow collection. | v3 places it under /tmp. Summary-first retrieval remains a pre-launch task; this path change alone does not repair the collector. |

During review, added game-name reporting broke the generic ExampleGame runner. The full runner test detected it; review-runner-regression.log preserves the failure. Reporting now falls back to the game label, and the test passes. This was not a GPU incident.

## Verification

- Eight original qualification methods pass against v3 (24.849 seconds): real worker/native tools and donor computation, generated runner, construction/repair over loopback HTTP, cancellation, tool-pair projection, image estimates, error accounting and successful/error HTTP trickling.
- Six review regressions pass (4.601 seconds): shared IPC deadline, child cleanup, artifact-failure cleanup, late/short Builder allocation, expected game timeout classification and small-runner configuration.
- verify.py passes notebook schema/size, 93 payload file hashes, source provenance, unchanged donor serving/hardware/model references and local dependency checks.
- All 55 downloaded q4 requests inspected for orphan tool results and unfinished tool exchanges across user boundaries: zero found. This checks actual historical requests, not every possible future long context.

Reproduce from repo root:

```bash
python3 solvers/tycho_dfranzen/build.py
solvers/tycho_nooa/.venv/bin/python solvers/tycho_dfranzen/verify.py
solvers/tycho_nooa/.venv/bin/python solvers/tycho_dfranzen/tests/qualify.py
solvers/tycho_nooa/.venv/bin/python solvers/tycho_dfranzen/tests/review_regressions.py
```

Raw evidence: evidence/review-before.log, review-after.log, deadline-before.json, review-runner-regression.log, qualification-v3.log and q4-request-integrity.json. The original two zero Builder budgets are preserved in deadline-before.json.

Current generated notebook SHA256: c43384f030fdd698991190345312eeb84b638722a8c7f2513878e8e19406bccc. Source metadata uses the distinct private source-v3 slug; it has not been uploaded or GPU-qualified in this review.

## Remaining limits

No v3 GPU result, no matched score comparison, no mechanism efficacy conclusion. Process-group tests are lifecycle checks, not a general adversarial sandbox guarantee. The three-game plan avoids queued donor admission; public25 scheduling/cancellation fairness is not established. Exact future ablation semantics, voluntary uptake, statistical variation and server-side cancellation remain unknown. See PRE-MORTEM.md for the experiment risks and decisions. Before the next GPU launch, finish summary-first collection, freeze its card and check the exact run artifact. This review found/fixed specific defects; it does not certify the system bug-free.

## Launch preparation update — 2026-10-04

The preceding scope and findings describe the first v3 review snapshot. A later independent board review identified additional paths, which were reproduced and fixed before launch:

- A game's escaping persistence/cleanup exception previously stopped its siblings. It is now caught per game; only cancellation triggers arm-wide cleanup. The old behavior is preserved in evidence/sibling-failure-before.json; the new sibling completes normally in launch_regressions.py.
- A long newest tool exchange could exceed context with no user boundary available to drop. Projection now removes older reasoning first, then older tool bodies while retaining arguments, matching tool IDs, current observation and latest result. Applied reductions persist across subsequent requests and are logged. Actor/Scribe tool bodies are capped at 12 KiB instead of 96 KiB. An unshrinkable required input raises a recognizable context-window error. This is an additional declared context policy change.
- A fresh policy-first import hit a circular import. Generated package exports now resolve the agent lazily; the fresh-process regression passes.
- Runner SIGTERM uses the asyncio loop; notebook cleanup now kills a runner that ignores its termination grace. Provider endpoint configuration is retained when adding donor sampling defaults, including top-k on the explicit thinking-disabled action-demand path.
- Critical-first collection is implemented and exercised against real completed q4 output: 34 retrieved files match preserved bytes; three 16-action chains and token ledgers reconcile. New runners emit a hash-bound evidence index. Missing/corrupt core evidence stays pending and prevents bulk transfer. A persistent user service owns collection independently of this conversation.
- A separate notebook watchdog is tested with real parent/child processes. GPU q5 soft stop is 3300 seconds after its first cell; forced stop is 3590 seconds; the API cap is 3600. Installed Kaggle SDK code explicitly sets session_timeout_seconds from --timeout, correcting the reviewer's claim that q4 had only metadata enforcement. A submitted API cap is not a direct measurement of platform enforcement, hence the second limit.

Twenty local test methods now pass: eight original, six review, three launch regressions and three collector/watchdog tests. All 55 actual q4 provider requests were also replayed at a conservative one-character/token estimate and retained valid tool pairs within the envelope. Frozen base notebook hash is now 6b1e91690130ee532f2df2b665a49e2b531718697034e9b01cf8356bcbb5a6c3; frozen run identities are in experiment/cpu-q5 and gpu-q5. Their source payload is identical. CPU q5 is a zero-GPU offline-install/hardened scripted rehearsal; GPU q5 launches only after its preserved outputs pass inspection. No score efficacy conclusion follows from these checks.

## q5 runtime failure and local correction — 2026-10-04

GPU q5 ended ERROR before gameplay. After serving was ready, runner.py evaluated game.label as the default argument to getattr while writing game_names. Actual GameAPI has env_name and no label. The CPU runner fixture used ExampleGame, and the small-set configuration test used a stub with both fields. Those tests failed to represent the production input contract. See gpu-q5/RESULT.md and FAILURE-AUDIT.json for preserved evidence.

The same AttributeError was reproduced through the actual generated runner using pickled GameAPI objects (evidence/gameapi-regression-before.log). The source now uses a conditional that evaluates label only for game types without env_name. The small-set test now constructs actual GameAPI objects. A new runner test executes one legal scripted action through each actual offline ar25/ls20/sb26 engine and checks both benchmark actions and host receipts. An initial shared-UP fixture was corrected because sb26 does not advertise UP; this fixture error is retained in gameapi-regression-after.log. The corrected fixture and existing ExampleGame runner test pass in gameapi-regression-verified.log; the real-GameAPI small-set configuration test and packaging verification also pass. No full-suite rerun or GPU success is claimed from these affected checks.

Current rebuilt notebook SHA256 is e41f8573fbe8bc0d7a8912f46ea00d53197cbacfe3a97c64c752c383d8c8a154. Its source bundle uses the distinct, unuploaded source-v4 dataset identity. Frozen q5 inputs and raw receipts are unchanged. Before another GPU launch, run the packaged CPU rehearsal against the same GameAPI shape and freeze a new card. No retry or matched pair has launched; eight campaign GPU-hours remain under conservative accounting.
