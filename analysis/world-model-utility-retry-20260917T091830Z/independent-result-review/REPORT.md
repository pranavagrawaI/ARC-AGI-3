# Independent result audit — HOLD, incomplete utility comparison

**The actor requested nonexistent transition14; the host's predeclared fail-fast policy aborted the pair.** Treatment made10 receipted requests and5 new official actions (RIGHT, DOWN, DOWN, DOWN, DOWN), completed0 levels, remained NOT_FINISHED, official score0. Control was never started. There is no positive or negative paired utility conclusion and no matched20-action milestone.

## Exact failure and interface

Raw `live/treatment-raw/010.response.raw` contains:

```json
{"name":"retrieve","arguments":"{\"count\": 1, \"notes\": \"verify step14 state\", \"start_step\": 14}"}
```

At this request,8 prefix+5 new actions yielded13 transitions. `DECISION-010.json` records step_before13. Decoding the actual outbound request shows executed_summary steps1–13, recent transitions ending13 and new_actions5. Legal ranges were count1 with start1–13, or count2 with start1–12. Transition14 did not exist.

The actual outbound schema requires an object with integer start_step>=1, count in1..2, notes string<=64, all three required and no extra fields (`actor.py:17`). **These arguments satisfy that static schema.** It does not encode the current-history upper bound or the joint start+count restriction. The actual tool description says executed prefix only, and the system prompt explicitly says already-executed transitions, never future evidence. Thus there is a schema/runtime coverage gap, but no evidence that the host rejected a genuinely available transition. The compressed history did contain the correct frontier; no dynamic maximum or explicit legal-range field was supplied in the schema.

`actor.py:45–54` parses and records the raw arguments unchanged, then raises `ValueError('retrieval future/range')` because14>13. The outer handler writes ACTOR-FAILURE and rethrows (`:72–76`), so no structured retrieval error is returned to the actor. Worker returncode2 blocks promotion and prevents control startup. Supervision reports24.518846seconds, reaped:true, hard_timeout:false; this is not the prior HTTP timeout problem. Its failure:null describes absence of a supervisor exception, not successful actor execution.

The request error is **operationally recoverable in principle**: the host could reject the read without changing game state and allow another bounded decision. This frozen implementation instead terminates on invalid actor requests under its card. Correctly refusing future data and aborting the experiment are distinct choices; the latter amplified an invalid retrieval into an incomplete comparison. No recovery was implemented or attempted in this audit. A different error policy would require a separately authorized candidate, not continuation of this consumed run.

The saved provider output proves what reached the adapter. It does not establish checkpoint-only causation, independently exclude provider serialization, or prove whether compressed-history interpretation, step bookkeeping or another model error produced14. There is no observed local argument coercion.

## Accounting and query evidence

All10 HTTP200 responses, raw hashes, usage and receipts reconcile: **24997 prompt tokens,470 completion tokens,0 reasoning tokens; $0.006547858 known; $0 new unresolved exposure**. Requests use reasoning-off and pinned Reka routing. No11th request or control directory exists. This was a host validation abort, not a budget exhaustion or unresolved provider response.

Three actual frozen-model queries were delivered before relevant commitments: mandatory RIGHT query1 plus voluntary RIGHT query2 before step9, and DOWN query3 before step10. Raw query hashes join to actor input packets and actual transitions. The notes on request2 say “commit RIGHT per predicted frame”; that is recorded actor wording, not a causal measurement of help.

| Query | Executed transition | Changed cells | Exact | Cell accuracy | Gain over identity |
|---|---:|---:|---|---:|---:|
|1,2 RIGHT|9|201|false|0.950927734375|0|
|3 DOWN|10|0|true|1.0|0|

Independent arithmetic confirms these particular predictions equal the current frame. Queries1 and2 score the **same** transition, not two independent outcomes. Query3 correctly predicts an unchanged frame, with undefined changed-cell accuracy. There are two distinct scored transitions, no baseline improvement and no evidence of general predictive competence. No efficacy inference follows from exposure, actor notes or these scores.

## Reproduction, bindings and preservation

`audit.py` recomputes costs, token totals, request/response/decision hashes, legal history, query exposure/score arithmetic, official scorecard agreement and package preservation. It executes the actual `actor.run` retrieval branch with the saved response and a historical-state adapter. The adapter supplies13 preserved rows and forbids game stepping; it is explicitly an offline boundary reproduction, not a rerun of gameplay/model/provider calls. The exact ValueError and failure JSON match; no action join is created. Fresh local evidence is under the directory recorded in AUDIT.json.

Run from repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 deployment/kaggle/.venv/bin/python -B analysis/world-model-utility-retry-20260917T091830Z/independent-result-review/audit.py
```

Final inventory remains `6222b21ed809bd4e1a41043593e4a594b9390f7a69a18d4dad1dc5213bdeafc7`; all inventory/runtime hashes match. The old released package and exact old ledger hash match prior review. Its **$0.170458010 known and $0.345637232 unresolved** remain separate and unchanged. Old HOLD and consumed states were not reset. AUDIT.json preserves hashes of the live evidence inspected and the original package decision.

**Permitted next decision:** accept this as an incomplete run caused by an invalid actor retrieval plus fail-fast host policy. Any proposal for bounded recoverable validation feedback belongs to a new authorized scope. No retry, repair, utility claim or new launch approval is issued. Only independent-result-review artifacts were written; no credentials/API/model calls or game actions occurred.

SUPERSET_WORKER_DONE
Task: WORLD-MODEL-UTILITY-RETRY-RESULT-REVIEW
Summary: Failure reproduced; known accounting and exposure/scoring reconciled; incomplete comparison, no utility conclusion.
Files: independent-result-review/REPORT.md, audit.py, AUDIT.json, fresh offline boundary evidence.
Checks: Exact raw argument/schema/history join, real actor validation branch, all10 usage receipts,3queries/2transitions, inventory/runtime/old preservation.
Handoff: Read-only outcome accepted; any changed recovery policy requires a new authorized candidate.
