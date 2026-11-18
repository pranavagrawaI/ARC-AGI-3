# Local all-game budget repair: passed

Command: `deployment/kaggle/.venv/bin/python -m pytest deployment/kaggle/submission-20260910-v2/test_submission.py -q --basetemp=analysis/deployment-probes/20260910-first-submission/entrypoint-v4`

9 passed in 23.07s. Six budget cases (1, 3, 25, 55, 110, 135 supplied games), embedded payload identity, and both real direct/gateway entrypoints passed. Both entrypoints executed 10 Solver actions with controlled model responses. Worker PIDs 233239 and 233486 were absent after completion. Raw subprocess logs and IPC are retained in entrypoint-v4.

Notebook SHA256: 15c6ab29cc97f74a317f01060c0e0d3bc68e210426d391aef5cdce0743c3a056.

No live local 27B inference, performance conclusion, or competition readiness claim follows. Public v2 qualification remains required. Production client/worker timeouts (90s/120s) need live qualification; local controlled entrypoint tests use 30s/40s.
