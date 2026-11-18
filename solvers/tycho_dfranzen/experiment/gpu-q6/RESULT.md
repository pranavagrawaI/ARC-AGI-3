# GPU q6 result

Private Kaggle version 1 is COMPLETE, verified live on 2026-10-04. The notebook finished normally within its one-hour cap; its watchdog recorded notebook completion. Startup consumed about ten minutes, followed by forty minutes of concurrent gameplay. This was a qualification run, with no matched donor control.

| Game | Levels completed | Score | Actions |
| --- | ---: | ---: | ---: |
| AR25 | 7/8 | 77.78 | 219 |
| LS20 | 2/7 | 10.13 | 158 |
| SB26 | 1/8 | 2.78 | 60 |

Mean score: **30.23015873**. All games reached the configured 2400-second game deadline; none crashed. Their `gave_up` states and worker termination reflect that resource cutoff. The 437 host action receipts reconcile with benchmark history, action IDs, click coordinates and successor observations. Original evidence remains unchanged.

## Construction

Only LS20 requested the Builder: twice, at decision sequences 30 and 72, while on its second level. Both requests were admitted with a 240-second allocation. They consumed 40 completed Builder model calls and 46,161 completion tokens. Both ended in `BuilderDeadlineExceeded`, after 23 and 18 tool calls respectively. The first attempt read and ran inspection code without writing a model. The second wrote `features.py` after repairing a tool argument error; the file was rolled back at the deadline. Neither attempt retained a candidate or produced a valid Builder report.

Thus requested=2, admitted=2, completed successfully=0. There is no accepted Builder model or independently checked prospective Builder prediction in this run. First-shot mechanism correctness and recoverability remain unmeasured. Tool argument repair does not count as mechanism recovery. Builder timeouts do not establish that executable construction is useless; they establish that this implementation/model combination did not complete construction within its allocation.

## Accounting defect

The automatic audit correctly failed: AR25 and SB26 host call/token counts disagree with their durable traces. The host records only the most recent IPC stats snapshot. During the final pending decision, AR25 completed three further model responses and SB26 five, before the game deadline terminated the worker. Those responses were absent from `compute.json` and benchmark token totals.

The host totals match exact trace prefixes, including every per-role count and token total. All eight additional responses have hash-verified raw provider receipts whose usage matches the corresponding durable trace. Completed usage is **418 calls / 693,060 completion tokens**, versus the host's **410 calls / 675,426 tokens**: a difference of **17,634 known completed tokens**. Any outstanding provider request remains unresolved; this audit does not claim its cancellation or assign it zero usage.

See [AUDIT.json](AUDIT.json) and the reproducible [reconcile_results.py](reconcile_results.py). The original `AUTOMATIC-AUDIT.json`, benchmark and compute artifacts are preserved, so their failed accounting checks remain visible. The frozen notebook and metadata identities match the launch identity; retrieved indexed artifacts pass their hash checks.

## Judgment and next task

Real-model gameplay and Actor-controlled Builder admission are demonstrated. Follow-up: host deadline accounting is now fixed in the editable candidate and passes local regression/replay checks; q6's original failed accounting artifacts remain unchanged. See [the repair review](../../evidence/deadline-accounting/REVIEW.md). The completed [Builder examination](BEHAVIOR.md) identifies invisible budget and late finalization as a failure path; changing that lifecycle would be a distinct mechanism intervention.

This run proves neither a hybrid advantage nor construction efficacy. AR25's score is promising, but q4 had a different horizon and no matched control exists. No additional GPU run was launched by this inspection.

The notebook log ends at 3014.423284 seconds: about **0.83734 GPU-hours** under Pranav's approved elapsed-log fallback charging policy. This is not an account billing receipt; startup is included, queue waiting is excluded. Earlier run charges remain untouched.
