# Pair-1 status, 2026-10-05

Both private notebooks accepted as version 1 and currently QUEUED.

- Control: https://www.kaggle.com/code/pranavagrawai/tycho-dfranzen-pair-1-control (kernelId 137182958).
- Hybrid: https://www.kaggle.com/code/pranavagrawai/tycho-dfranzen-pair-1-hybrid (kernelId 137182970).

Control accepted 13:34:22 UTC; hybrid 13:34:27 UTC. Card and immutable identities frozen before API calls. Three GPU-hours reserved, 4.1626601988325 campaign hours unreserved. Notebook queue time is not GPU execution time. No score or capability result yet.

Collector `tycho-dfranzen-pair-1-collector.service` is active under the user systemd manager. It polls both arms, records status-history.jsonl, downloads hash-bound critical outputs at terminal status, reads account quota after both runs, and never retries or launches follow-up work. A score/action/Builder mechanism audit is still required after collection; collection alone is not an efficacy verdict.

Candidate includes bounded same-cap Builder construction, early commit/verification/handoff, and interrupted-provider usage uncertainty. 40 local checks passed, deployment payload hash read back from private source-v7, reviewed, commit 86004ed. Exact source/test/qualification artifacts are preserved here. Original q6 and previous candidate remain untouched.

One local launcher command failed before budget reservation or any kernels_push call because the frozen script looked for the campaign ledger in its own folder; the canonical qualified launcher beside budget.json then launched both arms exactly once. Evidence in launch-preflight-failed.log and launch-attempt/response receipts. Frozen collector is invoked in watch mode and does not use the ledger path.

Next: inspect downloaded raw receipts, recompute per-game scores, audit first-shot predictions and recovery, then apply frozen margins/result map. No conditional mechanism run automatically authorized by a collector.
