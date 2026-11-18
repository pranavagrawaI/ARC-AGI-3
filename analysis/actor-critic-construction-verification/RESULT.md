# Actor–critic with construction: verified local integration

## Result

The new overlay is `analysis/actor-critic-construction-integrated/`. It retains the incumbent v11 actor and epistemic critic and adds construction from the same run's host-recorded executed transitions. It does not route gameplay through GameplaySolver.

The actor can request construction, query the frozen executable model, submit its own game action, and assimilate the matching subsequent executed transition. Candidate execution uses the existing sealed RPC boundary. Construction runs in a separate killable process; timed-out work is reaped and cannot subsequently publish a freeze into the actor run. Frozen publication is exclusive and source digests are checked before use.

## Independently executed checks

Final combined suite: **203 passed in 93.98s**.

This includes the entire incumbent runtime suite, new integration tests, and coordinator-owned behavioral regressions. Ruff passed for the new implementation and coordinator scripts.

The final source-hash verification reported:
- Candidate changed during execution: false.
- Construction donor changed during execution: false.
- Incumbent changed from baseline: none.

`integration-check.json` contains the exact command and source SHA256 values. `integration-check.stdout` is actual captured test output. `run-03/` preserves the test workspaces and receipts.

Independent HTTP worker proof:
`run-03/test_http_constructor_receives0/coordinator-http-proof.json`

That check used explicit constructor HTTP CLI configuration through the real worker launcher. It recorded one constructor request over the worker's own executed prefix at frontier 1, thirteen scripted actor requests, and one scripted critic request. The later fixture observation was absent from the constructor input. Prediction and assimilation results reached subsequent actor requests, and normal cleanup completed.

Failure regressions cover missing prefix without consuming the opportunity, transport failure with continued play, digest tampering, source-module isolation, unscored rather than fabricated compatible verdicts, prediction-branch handling, executed action/step binding, and cancellation without late publication. The late-publication regression initially failed by observing a FROZEN.json written after FAILED was returned; it passes after replacing the abandoned thread with a terminated and reaped process.

## Run the focused checks

From the repository root:

    unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -m pytest analysis/actor-critic-construction-integrated/tests analysis/actor-critic-construction-verification/test_contract.py analysis/actor-critic-construction-verification/test_http_worker.py analysis/actor-critic-construction-verification/test_timeout.py -q

Entry point: `analysis/actor-critic-construction-integrated/headless.py`.

It wraps the incumbent headless actor and accepts explicit `--constructor-transport`, `--constructor-fixture`, and `--constructor-url` arguments through `NOOA_AGENT_COMMAND`. Ambient temporary fixture files are not used as configuration.

## Scope and limits

- Model responses and observations in these checks are scripted fixtures. HTTP transport and the actor/critic/constructor execution paths are real; model intelligence and gameplay benefit are not established.
- Visible replay checks the executable ABI. It does not establish informative construction completion, held-out prospective support, or general competence; records explicitly retain `supported_model: false`.
- The initial implementation permits one terminal construction episode per game after a nonempty prefix, not repeated reconstruction throughout gameplay.
- The donor's 45-second construction budget is soft. A separate process timeout, default 50 seconds, terminates and reaps overdue construction work.
- This is a local experimental overlay, not a qualified Kaggle submission. No paid or remote model run, commit, or push was performed.

## Collaboration

- Implementer: terminal `b5fc8aaf-b5c8-4f80-9724-faabb705986d`, owned the new implementation directory.
- Reviewer: terminal `015a2502-15f5-48a8-9505-c0f14656b761`, read-only architectural and lifecycle review, with executable reproductions.
- Coordinator: independent regressions, real HTTP worker check, full-suite execution, source-hash binding, and this report.

The reviewer approved local machinery on the final service/job/deadline-test hashes and independently ran the two deadline/publication tests. The final 203-test report binds those exact hashes, with no source drift. That approval does not extend to live-model competence or deployment.
