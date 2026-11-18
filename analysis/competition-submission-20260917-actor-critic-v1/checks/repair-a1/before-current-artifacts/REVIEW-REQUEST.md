# T4 independent review request — no approval granted

Review the frozen manifest and exact notebook of `pranavagrawai/arc3-ac-competition-20260917-v1` for competition feasibility. This is not a request to launch automatically. Neither authorization.json nor final-review.json exists in the real run directory.

Required review files (relative to this directory unless prefixed):

- manifest.json, EXPERIMENT-CARD.md, launch_control.py, freeze.py, test_gate.py, exact_version.py.
- ../../deployment/kaggle/submission-20260917-actor-critic-competition-v1/{submission.ipynb,kernel-metadata.json,ASSETS.json,DONORS.json,build.py,budget.py,entry.py,play.py,install.py,official-hashes.json}.
- Candidate runtime/{CHASSIS.json,adapter.py,shared_local_solver.py,local_solver.py,nooa_worker.py,headless.py,nooa_public_probe.py,exact_request_outer_runtime.py,chassis/solver_agent.py,chassis/epistemic_critic.py,chassis/objection_ledger.py} plus all remaining manifest-bound sources.
- qualify_gateway.py, test_local.py, test_entrypoints.py, test_adapter.py, test_deadlines.py, test_budget_accounting.py.
- RESULT.md, checks/final-checks.json, checks/inherited-cp312.log, checks/entrypoints-cp312.log, checks/new-tests-cp312.log, checks/sdk-request-shapes.log.
- checks/gateway-110-cp312/RESULT.json, gateway-scorecard.json, model-requests.jsonl, notebook/working/submission.parquet, notebook/working/qualification/{launcher-result.json,game-progress.jsonl,budget-allocations.jsonl,import-origins.json,request-budget-*.jsonl}, and worker IPC model_calls/request_lifecycle/transitions records.
- checks/gateway-critic-cap-cp312/RESULT.json and raw records, exercising the four-call critic ceiling.
- checks/gateway-reserve-cp312/RESULT.json and corresponding raw records, showing one Actor slot exhausted while the reserved critic still executes.

Discriminating review questions:

1. Does notebook commit avoid model/bootstrap costs, and does default rerun reach only the real gateway and pinned competition mount?
2. Are bootstrap, startup/warmup, two possible request tails, per-game cleanup, outer cleanup, final output and unknown provider overhead funded within 32400 seconds? The 900-second provider uncertainty reserve is policy, not a measured guarantee.
3. Does shared per-game request accounting survive resets, count summarizer calls against Actor slots, and retain a critic request and time reserve? Does supervisor reconciliation reject every unrecorded or extra request?
4. Do allowlisted materialized bytes and import origins match the freeze? Can any fixture override accidentally be inherited in ordinary rerun?
5. Does the atomic one-push gate fail closed, use the SDK-normalized title, and bind independent review/authorization to the exact identity, manifest, card and notebook? Submission remains a separately authorized future action tied to an exact pushed version.
6. Is the public-mode qualification requirement in the T2 handoff still required before competition submission? T3 intentionally implements cheap commit mode, and local fixtures do not discharge a remote public-run gate.

Limits: these are CPU-only synthetic local games and scripted model responses on Python 3.12 with site-packages origins checked and all pinned tycho_nooa source hashes matched. The official REST API and official Agent.main execute; parquet is written by a local callback implementing the published sidecar contract. Kaggle's private gateway writer, private game inventory, pinned GPU/container installation, real model latency/capability, remote timeout semantics and remote output completeness remain unexercised. No remote or scientific success is claimed. An independent reviewer must decide whether these remaining boundaries permit a separately authorized provider qualification. Do not self-author approval from these local results.
