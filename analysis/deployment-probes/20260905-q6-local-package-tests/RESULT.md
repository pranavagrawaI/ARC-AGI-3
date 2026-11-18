# Local testing of queued Kaggle package — 2026-09-05

27 checks passed: nine against the notebook's extracted host payload, and 18 resolution/scoring/integrity checks. The queued package was not edited or launched. Its final prepare.py --check passed; the original contract remains pinned.

The extraction reads literal embedded assets without executing notebook cells. It verifies all manifest runtime hashes and loads the extracted MyAgent, shared LocalSolver, worker and chassis. Seven existing host tests passed against these bytes, including level-boundary tail discard, native evidence parity, no-impact interruption, prediction evidence, reset/budget lifecycle, worker-death cleanup and real CodeAct rejection repair with fake replies. A deliberately corrupted extracted stall_contract.py was refused at worker startup.

The added real local ls20 test completed 200 Solver decisions, 201 environment calls (one GAME_OVER reset), two worker sessions and zero cleared levels. All 200 transitions were retained; both sessions stopped normally, shared one run identity, and had distinct session identities. The policy was a deterministic stub. Zero levels is not a model-performance result. The official Agent base was represented by the existing test fixture; its frame converter was extracted from the exact hash-pinned official source. This does not exercise the complete GPU notebook, HTTP server or official launcher process.

The scorer rejected the stub trace as INVALID because policy_attempts.jsonl was absent in both sessions. This is the expected negative result: successful host transport alone must not become research evidence. The separate 18-test suite covers valid authority traces and tamper rejection.

An initial driver failure is preserved in INITIAL-DRIVER-FAILURE.txt. The direct local environment returns NumPy frames; the official launcher converts them to FrameData lists before MyAgent. Adding that exact pinned converter fixed the test harness. No production patch was required.

Environment provenance limit: the local ls20.py and metadata.json hashes differ from the Kaggle public manifest. Removing the leading license header did not establish a source match. No pinned downloaded copy was found in the inspected deployment outputs. Therefore this is real local-game host qualification, not exact Kaggle environment parity. The Kaggle notebook retains its existing strict environment hash verification. The saved ENVIRONMENT-CHECK.json records expected and actual hashes; no environment files were modified.

Commands run from repository root:

```bash
deployment/kaggle/.venv/bin/python analysis/deployment-probes/20260905-q6-local-package-tests/run.py
deployment/kaggle/.venv/bin/python -m pytest analysis/local-probes/20260905T-stall-authority-q6/test_q6_contract.py analysis/local-probes/20260905T-stall-authority-q6/test_integrity_guard.py -q
python3 analysis/local-probes/20260905T-stall-authority-q6/score_q6.py analysis/deployment-probes/20260905-q6-local-package-tests/pytest-artifacts/test_real_ls20_full_budget0/workspaces --output analysis/deployment-probes/20260905-q6-local-package-tests/STUB-SCORE.json
python3 deployment/kaggle/q6-control-preflight/prepare.py --check
```

Results: 9 passed in 24.79 s (pytest plugin rewrite warning); 18 passed in 1.74 s (two existing set_dynamic deprecation warnings). Ruff passed on the local driver and added tests. Raw traces are under pytest-artifacts/, test output in TEST-OUTPUT.txt and SCORING-INTEGRITY-TESTS.txt, action/session totals in LS20.json, and notebook identity in CHECK.json. The scorer returned the expected invalid result, recorded in STUB-SCORE.json.

No Kaggle quota or paid model credit was consumed. Remaining external uncertainties are actual FP8 model behavior, GPU startup/performance and stall incidence under model-driven play. The pending Kaggle run remains necessary to settle those.
