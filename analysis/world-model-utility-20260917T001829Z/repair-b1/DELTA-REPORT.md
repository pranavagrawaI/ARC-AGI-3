# B1 deadline repair — ready for exact delta review

Original independent HOLD and all reviewer evidence are unchanged. Superseded inventory: `1306bf4aa80c0e4a5cf9ae7bbc089c66ede8427c3634e004a68037001c35599d`. All46 inventoried files are archived, and3014 original reviewer-file hashes reverified. Card, scientific design, actor, model, prefix, budgets, provider, credentials policy and scoring are unchanged.

## Reproduction and correction

The preserved reviewer probe was copied into repair-b1/before/ with only candidate/output locations adjusted. Before editing, actual Transport/supervise reproduced elapsed 23.551839652995113s for a21s hard limit, hard_timeout:false. The peer supplied successive bytes inside the inactivity timeout; the synchronous supervisor could not check elapsed wall time until the full body returned.

run_utility.py now arms an absolute kernel ITIMER_REAL/SIGALRM watchdog for the phase, including time spent in HTTP I/O. Expiry immediately kills the worker group and raises a BaseException-derived HardDeadline across ordinary transport exception handlers. Supervisor always restores prior signal handling, reaps, truthfully marks expiry and rejects promotion. The CLI fails closed if the main-thread alarm facility is already owned/unavailable.

transport.py wraps both success and HTTP-error body reads with the same interrupted-I/O receipt path. It closes response I/O, records status/latency/incomplete response, retains the durable possibly-dispatched reservation and unknown cost, sets the global stop, and rethrows. No complete response/hash or zero billed cost is fabricated. Partial bytes internal to the interrupted buffered read are not claimed recovered. No background transport thread, retry, fallback or remote cancellation claim is introduced.

## After-fix actual slow-body checks

| HTTP | Hard limit | Return/reap elapsed | hard_timeout | Body complete at return | Unresolved fixture exposure |
|---|---:|---:|---|---|---:|
| 200 |21s|21.00926811998943s|true|false|$0.335766910 |
| 429 |21s|21.01039801700972s|true|false|$0.335766910 |

Both tests assert one dispatched request, worker returncode SIGKILL/reaped, no late action, no root freeze promotion even with child-written freeze files, and no second HTTP admission. Known fixture spend remains0, actual cost is unknown; these are manufactured no-auth loopback requests, not paid provider calls. The ~0.01s over the21s timer is local interruption/receipt/reap overhead; the worker is killed on timer expiry while the response is unfinished.

Existing37 tests passed in83.05s; two new slow-body cases passed in42.51s. Exact commands/log hashes are CHECKS.json. Full raw requests, incomplete-response receipts, ledgers, supervision and child forensic markers are retained under repair-b1/. No additional speculative tests or infrastructure were added.

## Exact delta and next action

REPAIR-DELTA.diff contains executable changes to run_utility.py, transport.py, test_utility.py and the preparation-only make_inventory.py. The generator change only permits explicit hash-verified supersession of the archived HOLD when regenerating inventory; it does not alter production approval gates. README changes the decision path/basetemp to independent-review/delta-review/ so the original reviewer evidence is not overwritten. Current reports/checks/evidence index are refreshed; their predecessors are archived.

No paid calls, credentials, live state, launch, submission, R10 edit or R10b action. Ready for the independent focused delta review. A fresh reviewer PASS and PM/Hermes release must bind the new inventory before the exact README command runs once.
