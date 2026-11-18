# Independent utility package review — HOLD

Task: WORLD-MODEL-UTILITY. Review against `/tmp/world-model-utility-ruling.md` and the candidate card. No paid calls, credentials, implementation changes, release or launch. Only independent-review files were written; all network activity was local fixture HTTP.

## Exact candidate

- Inventory SHA-256: `1306bf4aa80c0e4a5cf9ae7bbc089c66ede8427c3634e004a68037001c35599d`.
- Card SHA-256: `efab867dd44b9d1d365884ed5ce99c2b750a67af8ce864026d9ab4ab79880939`.
- Independently verified 46 inventoried files, 164 runtime bindings, 19 donor-source bindings and 6407 retained evidence files. Full hashes in `bindings.json`. This is a bounded donor preservation check, not a snapshot of all concurrent repository activity.
- `live/` absent. Read-only candidate checker passed. All **37 tests passed in 90.18 seconds**, using the documented interpreter and a fresh reviewer basetemp.

## Blocking finding B1 — hard supervision stops while reading provider response

**Location:** `run_utility.py:59` calls transport synchronously inside the only hard-deadline loop (`:43–60`). `transport.py:74–77` uses a socket timeout and then blocking body reads, including HTTP-error bodies. The supervisor cannot reach its deadline check or kill/reap path (`run_utility.py:65–70`) while those reads continue. A peer delivering successive chunks before each socket timeout can exceed the total phase deadline. A late-response check after the entire body arrives is not an enforced wall-clock stop.

**Reproduction:** `probe_hard_deadline.py` runs the actual `Transport` and `supervise` with a credential-free loopback HTTP server that supplies one valid provider-shaped response slowly. Soft deadline 20.5 seconds; supervisor hard deadline 21 seconds. The actual receipt was:

```
elapsed_seconds: 23.544551312050316
hard_timeout: false
returncode: -9
reaped: true
requests: 1
failure: late response: no tools/publication
```

The worker was only killed after transport returned, 2.54 seconds beyond the declared hard wall. The scaled deadline makes this test bounded; production uses the same synchronous path for constructor 240/245 and actors 1800/1805. The issue can extend beyond this particular delay. Raw fixture request/response, ledger and supervision receipts remain in `hard-deadline-probe/`; summary in `hard-deadline.json`.

**Consequence:** the declared phase hard caps and timely cleanup are not qualified during real HTTP I/O. The receipt can also misclassify an actual deadline overrun as `hard_timeout:false`, because Transport raises its own late-response Stop before the supervisor's post-call check. The probe did preserve validated fixture usage and reject the late result; it did not execute a late action or demonstrate a dollar-cap breach. This finding concerns enforceable local time/cleanup, not an impossible demand for remote cancellation.

**Minimal resolution:** keep an independent absolute-deadline supervisor active while provider I/O is blocked, with bounded termination/reaping of the worker and transport activity. Preserve the already durable possibly-dispatched reservation; unreceipted exposure must stop all future spending. Do not convert timeout into a retry or claim remote cancellation. Mark elapsed hard expiry truthfully and prevent late freeze/action promotion. Add a real loopback slow-body regression (including the error-body path or a shared deadline mechanism covering it) and demonstrate deadline cleanup while the response is still unfinished. Preserve this HOLD and submit changed bytes for a focused delta review. No change to the scientific design is needed.

## Utility and integration review: no other blocking finding identified

- **Actual game path:** this is a live-capable parent/worker actor pipeline, not the earlier constructor-only test. `game.py:30–56` instantiates the installed official SDK in OFFLINE mode, records canonical observations, executes the fixed eight-action prefix and rejects prefix level/terminal changes. Each actor independently repeats and matches the preparation prefix before its first request (`actor.py:21`). The full-100-action fixture passed through official game stepping and scorecard reconciliation; fixtures are not model utility evidence.
- **Common compact actor / exposure:** both actors use the same policy, observation encoding, rolling history, executed-prefix retrieval and limits. Treatment alone receives one mandatory RIGHT query and subsequent query capability. `actor.py:25–37` includes the forced result in the first request. Decisions preserve the exact input hash (`:46–47`), and action joins preserve the exposed results and actual transition (`:59–63`). Real integration tested retrieval, a second query, and an action different from the initial RIGHT query. Exposure establishes delivery, not cognitive reliance or efficacy.
- **State/action binding:** `model.py:58–82` reuses only a query matching actual action, step and pre-action observation; otherwise it predicts the committed action before the game step. It never assimilates a discarded hypothetical branch. Actual successors receive an unscored verdict; null branch state and abstention are tested. Reset/level/terminal changes disable the model without repair or reconstruction. State-advance RPCs are additional to the 20 exposed query attempts, explicitly counted separately in the card.
- **Outcomes:** `reconcile.py:6–25` checks transition continuity/count, final observation, official scorecard identity/run/count/levels/status and actor input/response/query bindings. Primary levels/status and secondary official score remain separate. Ties are inconclusive, and incomplete arms are identified. Constructor failure produces unavailable-query exposure and continued treatment with no oracle; the final constructor result allows the workflow-failure interpretation. No efficacy, R10b readiness or original R10 identity clearance follows from these fixtures.
- **Information/credential boundary:** `game.py:5–7` disables SDK dotenv loading before import. Parent launches workers with a fresh allowlisted environment and network namespace (`run_utility.py:29–40`); generated source uses the inherited chroot/seccomp RPC. Malicious-source fixtures were rejected. The credential loader was inspected as code only and never called; it is reached by main after inventory/review/release/credential-approval checks. No credential is forwarded through worker argv/environment or persisted in request JSON. Official game source remains in trusted runner code and outside sealed candidate evidence.
- **Once-only entry:** fixed live path, exclusive directory/CONSUMED creation, hash-bound owner-specific gates and no retry/fallback are present. A partial live directory prevents another main invocation. Runtime/inventory drift is checked before key loading. No implementer-authored release or reviewer PASS was used.
- **Budget/treatment:** sequential requests reserve full advertised 131072-token provider completion exposure plus conservative prompt exposure before dispatch. Unknown exposure blocks further requests. Nominal actor allocations are symmetric and not borrowed; known actuals are preserved even on later routing/treatment failure. Reasoning OFF is validated for both actors and ON evidence for construction. Public endpoint metadata, rates and optional-parameter support were checked from preserved discovery bytes, not a new API request. Hosted provider identity and cap uncertainty remain the declared limitations; there is no exact checkpoint claim.

## Reproduction and evidence

From repository root:

```bash
python3 -B analysis/world-model-utility-20260917T001829Z/independent-review/audit_bindings.py
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/world-model-utility-20260917T001829Z/test_utility.py --basetemp=analysis/world-model-utility-20260917T001829Z/independent-review/NEW-UNUSED-BASE
```

`probe_hard_deadline.py` is a preserved standalone no-auth reproduction. It intentionally creates `hard-deadline-probe/` exclusively; to repeat, copy the script into a fresh reviewer-owned subdirectory and adjust its candidate-root reference. Never overwrite the original evidence. `tests.log` and the reviewer pytest tree retain the completed suite and real-boundary artifacts. `check-candidate.json`, `bindings.json`, `hard-deadline.json` and `DECISION.json` bind the review evidence.

## Handoff

**HOLD solely for B1.** Repair the blocking-I/O hard-deadline enforcement, add the real slow-response regression, regenerate the candidate inventory and request a focused exact-byte delta review. No new experiment-design or budget ruling is requested. After affirmative candidate review, the separate PM/Hermes gates still govern actual execution. This report grants no spend or release authority.
