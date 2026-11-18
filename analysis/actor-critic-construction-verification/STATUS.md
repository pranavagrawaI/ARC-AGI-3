# Actor–critic with construction: coordinator verification

User-requested collaboration:
- Implementer: terminal b5fc8aaf-b5c8-4f80-9724-faabb705986d; owns only analysis/actor-critic-construction-integrated/.
- Independent reviewer: terminal 015a2502-15f5-48a8-9505-c0f14656b761; read-only source review.
- Coordinator: independent integration execution and this verification directory.

Acceptance boundary:
1. Existing actor continues through its actual CodeAct/worker path and remains the only submit_actions authority.
2. Existing epistemic critic actually runs and remains tool-less, without constructor or action authority.
3. Constructor receives only chronological host-recorded executed transitions, not submitted batches, synthetic executed_actions files, or future observations.
4. Constructed executable initialize/predict/assimilate source is frozen, executed behind a fail-closed containment boundary, and usable by the continuing actor.
5. Construction failure, reset/level scope changes, and subsequent observation assimilation are exercised.
6. Scripted transport proves local integration machinery only; no claim of model construction quality, actor adoption by a real model, gameplay benefit, or Kaggle qualification follows.

Verified baseline before integration:
- deployment/kaggle/.venv/bin/python -m pytest deployment/kaggle/submission-runtime-20260912-v11/test_epistemic_critic.py deployment/kaggle/submission-runtime-20260912-v11/test_done_refusal.py -q
  Result: 10 passed in 10.65s.
- deployment/kaggle/.venv/bin/python -m pytest deployment/kaggle/submission-runtime-20260912-v11 -q
  Result: 175 passed in 79.39s.
- Root uv run --with pytest uses Python 3.14 and cannot import tycho_nooa in the real worker test. The deployment venv is the correct existing environment; no dependency changes were made.

No remote/paid launch, commit, incumbent overwrite, or prior-identity edit authorized to these workers.

Current status: final local integration independently verified — 203 tests passed in 93.98s; candidate and constructor donor remained unchanged during the run, and the incumbent matches its baseline. See RESULT.md and integration-check.json. This is not live-model or Kaggle qualification. Architecture review identified ipc/transitions.jsonl as authoritative and rejected the prior GameplaySolver bypass and in-process candidate exec.

Additional independently verified prerequisites:
- Local mapped namespace: unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python analysis/actor-critic-construction-verification/check_local_seal.py => passed; unchanged sealed runner executed the fixture at uid/gid 65534.
- Unchanged constructor test_integration.py: 10 tests passed in 3.255s under the mapped namespace, including host-canary isolation and chronology/freeze checks.
- Incumbent actor/critic and DONE-refusal tests also passed under the mapped namespace: 10 passed in 12.17s.
- Historical real IPC audit: recorded-prefix-audit.json. CD82 has 17 transitions and 17 state notifications, independently falsifying the N+1 assumption on that recording.

Independent regression reproduction on the initial new candidate:
  deployment/kaggle/.venv/bin/python -m pytest analysis/actor-critic-construction-verification/test_contract.py -q
  Result: 3 failed in 0.04s.
  Failures: no observation grids and no inspection tool in default constructor request; inherited LS20-specific SYSTEM; altered source digest reaches candidate RPC.

Read-only reviewer agrees NOT APPROVED and additionally requires no fabricated compatible verdicts, no silently discarded prediction branches, reset/level scope handling, and actual actor/critic integration. Implementation approval requires exact source hashes and rerun evidence, not only the proposal.

Subsequent verification:
- Independent constructor contract regressions now pass: 8 tests (evidence access, generic prompt, digest rejection, empty-prefix opportunity, transport failure, donor isolation, unscored replay, branch handling).
- Independent HTTP worker test passes. Explicit HTTP constructor configuration traverses the real worker; constructor receives its own executed prefix at frontier 1, without the later fixture observation. The actor receives prediction and assimilation tool output, the critic runs, and cleanup completes.
- Combined candidate/coordinator/incumbent suite passed 201 tests in 85.75s, with no source drift and no incumbent changes. That pre-timeout-fix result is preserved separately; it was NOT sufficient for approval.
- Independent real HTTP timeout regression reproduced a late FROZEN.json after FAILED had returned (1 failed in 2.54s). A daemon thread with join(timeout) was not cancellation.
- Implementer replaced that thread with a killable constructor process and staged publication. Post-fix candidate plus coordinator suite: 27 passed in 21.63s, including the delayed HTTP/no-late-freeze check.
- Publication now uses exclusive creation; the constructor job installs parent-death SIGKILL and parent interruption reaps the job. The final suite includes the exclusive-publication regression.
- Final source-bound combined suite: 203 passed in 93.98s. Candidate and donor source drift flags are false; incumbent baseline differences are empty. Ruff passed. See RESULT.md for the current deliverable and limitations.

