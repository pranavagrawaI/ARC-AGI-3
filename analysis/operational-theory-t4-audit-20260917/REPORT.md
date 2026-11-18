# T4 independent post-run audit — INVALID_RESULT

The paired scientific result is **invalid for comparison**. The launch, routing, budgets, accounting and artifact checks pass, but treatment encountered an executed CodeAct persistence defect. Both observed constructions were incomplete; this is not a clean “no advantage” result.

| Arm | Requests | Input tokens | Output tokens | Total tokens | Cost | Proposals | Accepted theory versions |
|---|---:|---:|---:|---:|---:|---:|---|
| Control | 6 | 93,174 | 1,951 | 95,125 | $0.020498798 | 0 | Not applicable |
| Treatment | 6 | 95,638 | 538 | 96,176 | $0.017322592 | 0 | 0 |
| Total | 12 | 188,812 | 2,489 | 191,301 | $0.037821390 | 0 | — |

All usage and costs were recomputed from raw responses and checked against receipts/ledger, including 139,552 cached input tokens. All twelve responses are HTTP 200, finish `tool_calls`, zero reasoning, one direct Reka routing attempt. Unresolved exposure is zero. No provider calls, reruns or source changes occurred during T4.

## What happened

Control spent its six calls inspecting the grid and writing `step_log.md` and `hypotheses.md`. It never submitted an action. Treatment inspected the grid, successfully called `workflow_skill()`, and read the canonical store/schema. It never called observe, reconcile, revise, prepare or submit. The host recorded two successful theory reads; final phase was idle with zero repair errors and no lineage. Both action logs are empty. Neither arm revealed the second recorded transition or executed gameplay.

Treatment call 4 ran `import json as _json` and successfully printed the store/schema. Call 5 reused `_json` and raised `NameError`; call 6 reimported it and reached the store again. The actual system prompt promises “state persists across cells.” Yet pinned `nooa/runtime/actor.py:1650` excludes every name starting with `_` from captured locals. The in-process wrapper therefore discards this alias between cells, and CodeAct reinjects only what was captured. The prompt does not disclose this exception. This is an inherited runtime/interface defect, not a failed schema operation or inaccessible workflow. See the exact response/request pointers and pinned source hashes in [T4-AUDIT.json](T4-AUDIT.json).

The defect wasted the fifth treatment request and prompted a sixth-call workaround within a six-call budget. That does **not** prove a corrected run would finish: four calls had already been spent inspecting, the model chose to truncate its schema printout, and it had not begun a theory revision. The preserved trace supports both an inefficient model trajectory and a concrete machinery defect; it cannot assign the entire failure to either one.

## Mechanical audit

The card, candidate manifest, independent review, authorization and all 165 manifest-listed launch files match their claimed hashes. Candidate and predecessor inventories remain unchanged (5,092 / 36 / 37 / 40 files). The sealed launch has no extra files, symlinks or writable entries. No live artifact mtime is later than launch end; both workers were reaped and no current bwrap process was found. Read-only permissions and matching hashes do not imply storage an owner cannot modify.

Authorization was recorded at **2026-09-17 13:52:52.784645 UTC**, before launch start at **13:52:52.800615 UTC** and first provider creation at **13:52:54 UTC**. Control provider creations span 13:52:54–13:53:37; treatment spans 13:53:45–13:54:00; launch ended 13:54:03.323896. Provider timestamps have second precision; file timestamps describe persistence, not exact dispatch. The preserved user-task record explicitly authorizes the run and is hash-bound by the authorization.

Fresh preserved metadata resolves exactly `qwen/qwen3.8-27b`, canonical revision `qwen/qwen3.8-27b-20260814`, at `reka/fp8`. All twelve receipts agree. One claim file, one consumption marker and one sequential pair are present; no fallback or paid retry is observed. Each Actor attempted a seventh local admission, but the transport rejected it at the cap before another HTTP dispatch. These are not fourteen provider requests. SIGKILL return codes reflect the designed budget-exhaustion cleanup.

The complete first requests are identical after removing only the declared treatment context. All twelve nonmessage settings and tool schemas match; sampling fields are omitted identically. Treatment’s first request adds 732 bytes / 136 prompt tokens; later workflow/schema output and arm-authored histories explain legitimate size differences. Total input differs by 2,464 tokens. Equal caps do not imply equal actual compute or spend. Output relocation was documented and changed the driver’s output path without changing frozen source or requests.

No recognizable credential patterns were found in 166 launch files or 126 archived preflight members, and request bodies contain no credential headers. T4 did not load secrets; exact-value credential exclusion remains the inspected operator sealing check rather than a independently repeated secret-value scan.

## Interpretation and next boundary

The common rubric is incomplete in both arms, with treatment invalid for comparison. Control retained measurements and scoped uncertainties; neither produced a complete prospective decision, post-observation revision or replayable next policy. Treatment’s lack of canonical versions is not independently a comparative statistic.

Retain this as operational/accounting evidence and a runtime-contract defect, not as evidence for or against Operational Theory efficacy. No gameplay competence or counterfactual completion claim follows. A separately scoped offline persistence repair and qualification should precede any fresh identity/review. **No rerun or further spend is authorized.**
