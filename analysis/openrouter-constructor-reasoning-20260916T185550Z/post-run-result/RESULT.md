# OR constructor reasoning pair — RESULT

SUPERSET_WORKER_DONE task=OR-CONSTRUCTOR-REASONING-PAIR; summary=authorized once-only pair completed descriptively, neither arm froze; checks=12 raw request/response hashes verified, accounting reconciled, initial shared request parity verified, all inventoried sources unchanged; handoff=read-only outcome review, no additional run/repair authorized.

## Observations

| Measure | Reasoning off | Reasoning on |
|---|---:|---:|
| Requests / HTTP 200 receipts | 6 / 6 | 6 / 6 |
| Actual reported USD | 0.028654528 | 0.040803382 |
| Prompt tokens | 135733 | 144565 |
| Completion tokens INCLUDING reasoning | 1499 | 5404 |
| Reasoning tokens | 0 | 2250 |
| Successful inspections | 2 | 4 |
| Submissions / rejected | 4 / 4 | 2 / 2 |
| Accepted freezes / repairs | 0 / 0 | 0 / 0 |
| Constructor elapsed seconds | 54.1396 | 107.1196 |
| End state | BUDGET_EXHAUSTED | BUDGET_EXHAUSTED |

Total validated actual spend **$0.069457910**; unresolved exposure **$0**. All 12 calls reported tool_calls finish reason; no reported truncation or per-request completion overrun. Both supervised workers exited 0 and were reaped without a hard timeout. The six-request ceiling was reached; reported token and wall ceilings were not reached. No retries, provider substitutions, additional runs, Kaggle calls or gameplay.

Actual wire requests pin reka/fp8 with no fallbacks. Off requests explicitly disable reasoning; all six responses report zero reasoning tokens and no reasoning payload. On requests enable medium reasoning; all six report positive reasoning counts with returned payload (16, 18, 25, 807, 1097, 287). Thus the provider-reported treatment contrast was observed, beyond accepting an outgoing flag. Initial wire bodies are exactly equal after removing only reasoning. Later feedback and trajectory naturally diverged.

All six submitted work_item values were **strings**, not objects. Each received the existing `ValueError: work_item must contain the exact required fields`. Neither arm corrected that representation before its request budget ended. Validation rejected these submissions before source compilation/sealed candidate ABI evaluation; no conclusion about source validity, dynamics, or the historical candidate_source stringified-list defect follows. Inspection did execute in the sealed boundary. No frozen candidate exists in either arm.

## Interpretation

This ordered singleton pair showed an applied provider-reported reasoning contrast but no successful construction/repair advantage: neither arm reached a freeze. On used more completion/reasoning tokens and inspections and had only two submission opportunities. Those are descriptive observations, not a causal estimate or gameplay efficacy result. Hosted model/checkpoint equivalence to Kaggle remains unestablished. No R10a identity HOLD or R10b gate is cleared.

## Allowed next action

Read-only independent review of these outcomes and raw receipts. Preserve consumed state, source/card and failures. No retry, code repair, new endpoint, additional pair, launch, submission or gameplay is authorized by this result.

## Evidence

- `../live/RESULT.json`, `../live/ledger.json`: pair outcome and durable actual/unknown accounting.
- `../live/off/RESULT.json`, `../live/on/RESULT.json`: authoritative arm outcomes.
- `../live/<arm>/child/raw/`: exact numbered request, raw response and receipt files.
- `../live/<arm>/child/tool-events.jsonl` and `PROGRESS.json`: actual inspection/submission feedback and transcripts.
- `SUMMARY.json`: per-call cost, usage, treatment, latency and factual submission types.
- `EXECUTION.json`: exact absolute once-only launch command and exit status.
- `EVIDENCE-HASHES.json`: post-run hash inventory; originals remain unchanged.
