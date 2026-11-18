# Independent B1 delta review — WORLD-MODEL-UTILITY

## Scope and exact bindings

Focused review of hard-deadline repair B1 against the preserved original HOLD. No experiment redesign, implementation edits, paid calls, credential access, release or launch. Writes are confined to this independent-delta-review directory. Test network traffic is credential-free loopback HTTP.

New inventory: `ac86862b5de9ddb3598205580417655ce3349ba3bc9309ee793e9b15892ea71b`.
Previous inventory: `1306bf4aa80c0e4a5cf9ae7bbc089c66ede8427c3634e004a68037001c35599d`.
Unchanged card: `efab867dd44b9d1d365884ed5ce99c2b750a67af8ce864026d9ab4ab79880939`.

`audit.py` independently rederives the four-file executable delta from the archived source bytes and confirms exact equality with REPAIR-DELTA.diff. Executable changes are limited to run_utility.py, transport.py, test_utility.py and the preparation-only inventory generator. Actor/model/game/worker/reconciliation/auth/request contracts, budget constants and runtime bindings are unchanged. Updated reports and evidence index are separately enumerated in bindings.json.

Verified: 47 current inventory members, all 46 archived original members, 164 runtime bindings, 19 donor-source bindings, 3014 unchanged original reviewer files and 12495 retained evidence bindings. The old independent-review/REPORT.md and HOLD decision remain byte-identical. No live directory exists. The candidate checker passes.

## B1 repair inspection

`run_utility.py:37–57` rejects an unavailable main-thread timer, then arms ITIMER_REAL for the remaining absolute phase deadline. The alarm handler kills the worker process group and raises HardDeadline, a BaseException subclass that cannot be swallowed by ordinary transport Exception handlers. This remains active during the synchronous provider read that caused the original finding.

`transport.py:72–91` covers both normal response reads and HTTPError-body reads. Interruption closes the response context, persists latency/status and response_complete:false, retains possibly_dispatched with unknown exposure and no actual-cost value, marks the global ledger stopped, and rethrows. It does not claim to have recovered partial buffered bytes, fabricate a complete response, infer zero cost, retry, or cancel the remote request.

`run_utility.py:80–96` catches the deadline, restores the previous handler, clears the timer, reaps the worker, marks timeout truthfully and prevents root promotion. The preparation-only make_inventory change accepts a hash-bound archived HOLD for regeneration; production gates are unchanged and still require review/release/credential approval bound to the new inventory.

## Validation

**PASS — B1 resolved; no remaining concrete blocker identified in this focused delta.**

39 passed in 130.92s (0:02:10) under the declared interpreter, fresh reviewer basetemp.

| HTTP | Hard limit | Observed return/reap | Body finished | Unknown exposure |
|---|---:|---:|---|---:|
| 200 | 21 s | 21.150184 s | false | $0.335766910 |
| 429 | 21 s | 21.009641 s | false | $0.335766910 |

Both receipts have hard_timeout:true, returncode:-9, reaped:true and exactly one request. Elapsed time includes local interruption, receipt writing and reaping overhead. Neither remote cancellation nor zero actual billing is inferred.


The fresh full suite exercises the original actual actor/official-game/query-exposure/containment paths plus the two added deadline cases. The slow-body cases use the actual Transport and supervise, return HTTP 200 and 429 separately, keep sending below the socket inactivity timeout, and assert interruption while the body is unfinished. Each asserts worker SIGKILL/reaping, no late action, no root freeze despite child-written forensic freeze, preserved unknown exposure, refusal of a second request, and restored signal state. These are local mechanical results, not actual provider cancellation or utility evidence.

## Reproduction and handoff

From repository root, using a fresh basetemp:

```bash
python3 -B analysis/world-model-utility-20260917T001829Z/independent-delta-review/audit.py
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/world-model-utility-20260917T001829Z/test_utility.py --basetemp=analysis/world-model-utility-20260917T001829Z/independent-delta-review/NEW-UNUSED-BASE
```

The README's example points to independent-review/delta-review/DECISION.json. The user's latest instruction places the actual decision at **independent-delta-review/DECISION.json** instead. This is an accepted runtime gate path, not a source change: use that absolute path for `--review` and bind Hermes release's review_sha256 to this actual file. Preserve the original HOLD. No README/source edit is necessary.

Actual review path:
`/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/world-model-utility-20260917T001829Z/independent-delta-review/DECISION.json`.

This affirmative package review resolves B1 for these exact bytes only. Separate Hermes release and PM credential approval remain required before the once-only live command. No new design/budget ruling is requested. Changed candidate bytes require re-review. Original scientific limits and R10 HOLD remain unchanged.
