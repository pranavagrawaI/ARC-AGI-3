# Fixed constructor interface pair — RESULT

SUPERSET_WORKER_DONE task=OR-CONSTRUCTOR-SCHEMA-FIX-REPEAT; summary=authorized once-only pair completed, both arms reached ABI-only freeze; checks=9 request/response hashes verified, costs reconciled, initial shared request and exposed work_item schema parity checked, candidate/prior/R10 hashes unchanged; handoff=read-only independent outcome review, no further run or gameplay authorized.

## Observations

| Measure | Reasoning off | Reasoning on |
|---|---:|---:|
| Requests / HTTP 200 receipts | 5 / 5 | 4 / 4 |
| Actual reported USD | 0.023971450 | 0.025998344 |
| Prompt tokens | 114279 | 86596 |
| Completion tokens INCLUDING reasoning | 912 | 3912 |
| Reasoning tokens | 0 | 2898 |
| Successful inspections | 4 | 3 |
| Submissions / rejected | 1 / 0 | 1 / 0 |
| Accepted freezes | 1 | 1 |
| Repairs after failed submission | 0 | 0 |
| Constructor elapsed seconds | 32.5160 | 84.5082 |
| End state | CANDIDATE_READY | CANDIDATE_READY |

Total validated actual spend **$0.049969794**; unresolved exposure **$0**. All 9 calls reported tool_calls finish reason. No reported truncation, token overrun, timeout, retries, substitutions or extra runs. Both owned worker processes exited 0, were reaped, and timely root promotion made the freezes authoritative.

Every off response reported zero reasoning tokens and no reasoning payload. Every on response reported positive reasoning with returned payload: 1140, 11, 1731, 16 tokens. Actual outbound requests explicitly disable versus enable medium reasoning, with the same reka/fp8 singleton pin/no fallbacks. Initial request bodies are identical after removing only the declared reasoning object. All exposed nested work_item schemas are equal across arms.

Both submitted work_item as an object and candidate_source as a list of source lines. No representation coercion was applied. Both reached actual sealed initialize, predict, assimilate on the recorded step-1 RIGHT transition and passed generic visible ABI replay. No rejection feedback or repair was needed in either episode.

## Freeze identities

- Off candidate source SHA-256: `f21b6413295bf114804eae769526fae6bcb47c8608533f25b08cef7ddc7364d2`.
- On candidate source SHA-256: `05926d38fd80b74176f64d4c71a592d071cf1bbb7d7c69aeb50d491da5a209fa`.
- Authoritative files: `../live/off/FROZEN.json` and `../live/on/FROZEN.json`; file hashes and complete evaluator receipts in SUMMARY.json / EVIDENCE-HASHES.json.

## Interpretation

The fixed interface was passed in both arms; the card's boundary-passage prediction was observed. The provider-reported reasoning contrast was applied. Both froze, so this pair establishes no successful-freeze advantage for reasoning on. On used one fewer request but more output tokens, wall time and slightly higher cost. These are descriptive single-pair observations, not a causal estimate.

The prior pair had no freezes and rejected all six submitted work_item strings. This pair submitted objects successfully under the joint schema-plus-feedback interface change. It is not a replicated old prompt, cannot isolate schema from feedback, and does not control serving/time variation. Neither episode used repair feedback, so no live repair success is claimed.

An accepted freeze is source/interface execution plus visible ABI replay only. Commitments were unscored, no prospective transition or gameplay ran, and no dynamics accuracy or gameplay efficacy follows. Hosted weights/runtime equivalence to Kaggle remains unestablished. R10a HOLD and R10b gates remain unchanged.

## Preservation and allowed next action

All inventoried candidate bytes, all 1195 snapshotted prior-experiment files and all 70 frozen R10 source hashes still match. Consumption markers and raw receipts remain intact. No secret was printed or recorded.

Next allowed action: read-only independent review of this outcome. No additional pair, edit, retry, source repair, Kaggle launch, submission or gameplay is authorized by these results.

## Evidence

`../live/RESULT.json` and `ledger.json`: pair result and actual/unknown accounting. `../live/<arm>/RESULT.json`, `FROZEN.json`, `SUPERVISION.json`: authoritative outcomes and lifecycle. `../live/<arm>/child/raw/`: numbered exact request/response/receipt bytes. `tool-events.jsonl` and `PROGRESS.json`: actual tool/evaluator evidence. SUMMARY.json reconciles per-call costs, treatment, latency, types and evaluation. EXECUTION.json records the exact once-only absolute command. EVIDENCE-HASHES.json binds all result artifacts without rewriting originals.
