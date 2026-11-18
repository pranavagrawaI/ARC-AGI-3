# Budget-aware submission candidate — 10 September

Public GPU budget/lifecycle qualification passed as Kaggle version 4 on 10 September.
No competition submission was made. Prior remote version 3 remains failed.
Evidence: `analysis/deployment-probes/20260910-budget-public-v1/RESULT.md`.

Build: `deployment/kaggle/.venv/bin/python deployment/kaggle/submission-20260910-v5/build.py`.
`BUILD.json` pins the notebook and `submission-runtime-20260910-v3` contract.
The model, reasoning-off configuration, sequential loop, deferred successor
publication and disabled background summarizer remain unchanged.

## Budget policy

Before each game, calculate from remaining gameplay time R and remaining games N:

    per-game work allowance = (R - 180) / N - 30
    model-call admission window = work allowance - request timeout - 15

Reserves: 180 seconds for final accounting/output, 30 seconds per game for
cleanup, and 15 seconds for tools/protocol work after an admitted request.
These are policy margins, not measured worst-case bounds. A negative window
fails preflight rather than pretending all games fit. Unused time feeds the
next allocation. Competition mode has no fixed 75-second cap; public mode
caps admission at 75 seconds to bound qualification.

At 25,140 remaining seconds, 110 games and a 90-second request timeout, the first
admission window is about 91.9 seconds, followed by a 105-second request/tool
reserve and 30-second cleanup reserve. Worker startup, environment creation,
analysis and transitions consume the allowance. This is fair time allocation,
not an estimate of the time required to solve each game.

An absolute monotonic deadline reaches the Solver. Before each model call it
either admits and records that call, or records exhaustion and parks without
making a request. The worker acknowledges exhaustion; the host finishes the
game without inventing an action and continues. The game context displays
remaining admission time. The watchdog uses remaining game time, not a fresh
allowance for every decision. Client retries must be zero.

## Evidence and limits

`budget-allocations.jsonl` preserves each allocation. `game-progress.jsonl`
records stop reason, elapsed time, actions and levels. Session IPC contains
`request_lifecycle.jsonl` (admitted/completed/failed) and, when applicable,
`budget_exhausted.json`. Accounting rejects unfinished, failed, late-admitted
or unrecorded calls, including on budget-stopped games. The outer reconciler
permits zero calls only for explicitly acknowledged budget stops.

Operational completion permits zero-action budget stops. It does not imply a
game was solved or enough solving time was available. Transport timeouts,
unbounded tool execution, startup failures and cleanup failures remain failures.
The 15-second tool margin is not a hard sandbox guarantee. Live SGLang request
retirement passed the three-game public qualification; competition output and
full-set budget behavior remain unqualified.

Run local checks:

    deployment/kaggle/.venv/bin/python -m pytest deployment/kaggle/submission-20260910-v5/test_submission.py deployment/kaggle/submission-20260910-v5/test_budget_accounting.py -q

These exercise the official Agent loop, offline LS20 and a local HTTP ARC
gateway with controlled model responses, plus accounting rejection cases.
They do not establish performance or solver-quality gains.
