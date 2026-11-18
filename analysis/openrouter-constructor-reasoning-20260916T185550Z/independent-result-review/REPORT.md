# Independent result review — DONE

## Narrow conclusion

The retained evidence supports a completed, provider-reported reasoning off/on constructor pair, with identical initial inputs apart from reasoning configuration. **Neither arm produced an accepted work item, reached candidate evaluation, or froze a model.** Both exhausted six requests. The common observed failure is an empty-string `work_item`, rejected at the local work-item boundary.

**There is also an inherited interface defect:** the outbound tool schema omits the nested work-item contract that the prompt describes and the host enforces. It does not permit a string, but does permit `{}` and arbitrary object fields. This schema/validator mismatch and possible provider-side tool serialization prevent attribution to checkpoint-only behavior. No local adapter mutation of the received arguments was found. The result does not establish that reasoning is generally ineffective, game-model correctness, treatment efficacy, or R10b readiness. Existing R10 HOLD is unchanged.

## Recomputed result

| Measure | Off | On |
|---|---:|---:|
| Receipted HTTP 200 requests | 6 | 6 |
| Prompt tokens | 135733 | 144565 |
| Completion tokens, including reasoning | 1499 | 5404 |
| Reported reasoning tokens | 0 | 2250 |
| Successful inspections | 2 | 4 |
| Rejected submissions | 4 | 2 |
| Accepted freeze | 0 | 0 |
| Provider-reported charged cost USD | 0.028654528 | 0.040803382 |
| Local episode seconds | 54.139554 | 107.119621 |

Pair cost sums to **$0.069457910**, matching the ledger, with zero unresolved exposure. All twelve individual costs also reconcile exactly from reported prompt/completion/cache counts at the retained Reka rates: uncached input $0.214/M, cached input $0.15/M, output $2.55/M. Off reports 65856 cached prompt tokens; on 61152. Cost is provider-reported accounting, not an independent account-statement audit. Reasoning tokens were not double-counted.

Every request asked for 4096 completion tokens; all observed completions fit. All prompt/output/request ceilings and local times fit. Each reservation recomputes from exact outgoing byte length plus framing and 131072 advertised output exposure; prior known actual cost plus each reservation fits $2. Ledger order is off 1–6 then on 1–6. No retry or extra call appears in this retained run evidence; this is not a claim to have independently enumerated all provider account activity.

## Exact rejection and representation

Evidence paths below are relative to this candidate directory. JSON pointers identify fields in single-line raw bodies.

- **Off calls 003–006:** `live/off/child/raw/003.response.raw` through `006.response.raw`, `/choices/0/message/tool_calls/0/function/arguments`, decode to `work_item: ""`. Call 003 puts all nine intended work-item fields at the outer argument level; its list-valued declarations are themselves strings such as `"[1]"`. Calls 004–006 retain only work_item and candidate_source and keep work_item empty. All four source strings contain real newlines and statically define initialize/predict/assimilate. That is syntax evidence only; none was evaluated.
- **On calls 005–006:** the same raw argument location contains `work_item: ""`. Call 005 likewise flattens nine fields into the outer object with stringified list declarations. Call 006 has only work_item and candidate_source. Static parsing of 005's stringified/escaped source fails with `unexpected character after line continuation character`; 006 contains multiline Python defining the ABI functions. This is a secondary, unexercised source observation, not the recorded rejection cause.
- Both arms' `child/tool-events.jsonl` contain exactly the raw response tool calls. `child/PROGRESS.json` transcripts retain identical parsed responses and matching logical-to-wire messages. Each subsequent outbound request includes the exact preceding feedback/tool-call IDs; on also preserves returned reasoning details/string through continuation. No local transformation moved work-item fields or replaced an object with the empty string.
- Every submission feedback is exactly status `rejected`, error `ValueError: work_item must contain the exact required fields`. `engine/constructor.py:313` calls `_validate_work_item` before obtaining candidate source at line 314, compiling at line 322, or evaluating at line 325. The rejecting predicate at lines 143–144 checks object type and exact key set. Thus the observed error is **not** an ABI NameError, candidate syntax rejection, dynamics failure or frozen-model result. The final rejection in each arm was retained, but no later model request exists to show the model received that last feedback.

All root results agree with child results except the added root authority field. Both progress records have evaluation=null and candidate_source=null, BUDGET_EXHAUSTED root status and prospective=not_run. No root or child FROZEN.json exists at the authoritative paths. Supervisors report returncode 0, reaped=true, hard_timeout=false (54.260852 and 107.243271 seconds). These are local process observations; no remote cancellation claim is needed.

## Actual outbound interface — confirmed defect and limits

`OUTBOUND-INTERFACE.json` extracts the actual first submission request in each arm: `live/off/child/raw/003.request.json` and `live/on/child/raw/005.request.json`. The shared schema is:

```json
{"type":"object","properties":{"work_item":{"type":"object"},"candidate_source":{"oneOf":[{"type":"string"},{"type":"array","items":{"type":"string"}}]}},"required":["work_item","candidate_source"],"additionalProperties":false}
```

The extraction retains the omitted source-description text too. There is **no** nested properties, required, additionalProperties restriction, minProperties, or field-type schema under work_item. Outer additionalProperties=false does not propagate into work_item. Consequently:

- Empty string work_item violates the advertised object type; the interface does not explicitly permit string/empty-string work items.
- Empty object `{}` and arbitrary objects satisfy this nested schema yet fail the host's exact nine-field validator. **That is a concrete schema/host mismatch.** Top-level flattened fields also violate outer additionalProperties=false.
- The system prompt explicitly names exactly nine fields and list requirements. `/messages/1/content`, parsed as JSON, contains required_submission.work_item with those nine nested keys, expressed as prose descriptions rather than a formal nested tool schema. Therefore the necessary instructions were delivered, but their machine-readable structure was incomplete.
- The validator's generic “exact required fields” feedback does not distinguish wrong type from missing keys or restate the expected nesting. The next requests preserve the full original instructions and error, but the error alone is not a targeted repair explanation.

The prior package PASS did not flag this inherited schema under-specification. The observed shared failure makes it material to interpretation now. It was shared by both arms and preserved intentionally; it is not evidence of a between-arm adapter difference. No schema/prompt/source repair was made in this review.

**Attribution boundary:** raw OpenRouter bodies prove the malformed representation existed before this local adapter handled it. They do not expose upstream model tokens, provider chat-template expansion, tool parser or serialization internals. “Hosted tool output was malformed” is supported; “the checkpoint alone caused it” and “provider serialization is excluded” are not. The schema gap is independently established, but this trace does not isolate its causal contribution.

## Treatment, parity and bindings

Every off response explicitly reports zero reasoning tokens and no reasoning payload. On reports reasoning counts **16, 18, 25, 807, 1097, 287**, each with returned reasoning evidence. The configured contrast is enabled=false versus enabled=true/effort=medium, with exclude=false in both. All retained response routing metadata reports direct Reka selection, attempt 1, canonical `qwen/qwen3.8-27b-20260814`, and no transformation pipeline. Requests pin reka/fp8. This verifies the contrast at the **provider-reported** level, not hidden computation or physical checkpoint identity.

The complete first outgoing bodies, after removing only the reasoning object, are equal and hash to `78b1575ca9361084dacac80f8bd66c95a6f87d9f4997301a793e6ef98e05889d`, matching both ledger start bindings. All twelve request/response hashes and per-call receipts match the ledger. All transcript responses, event calls, feedback and continuation messages reconcile.

Candidate inventory remains exactly `fa134897d4c592335ef9f7f047838773bf0c5cb4e74ca24921ee6d402c215dee`; every inventoried file still matches. INPUT-HASHES.json binds read evidence and source; CHECKS.json contains per-call calculations and observations. Inputs were checked again for drift before completion.

## Reproduction and permitted next decision

This command only rereads local evidence and regenerates this review's checks/hash manifest; it does not import or run candidate code, call an API, read credentials, or rerun the experiment:

```bash
python3 -B analysis/openrouter-constructor-reasoning-20260916T185550Z/independent-result-review/audit.py
```

The pair shows no successful construction in either condition; on consumed more tokens/time and changed the inspection/submission allocation. One ordered pair with a common interface failure cannot estimate a reasoning benefit or deficit. PM may decide whether separately authorized work should isolate the nested-schema/serialization boundary. No implementation, retry, spend, source fix, launch approval or R10b clearance follows from this result review.

SUPERSET_WORKER_DONE task=OR-CONSTRUCTOR-REASONING-RESULT-REVIEW; summary=provider-reported treatment/accounting/parity reconciled, both arms rejected empty-string work_item before source evaluation, shared nested-schema defect identified; files=REPORT.md,CHECKS.json,OUTBOUND-INTERFACE.json,INPUT-HASHES.json,audit.py; checks=offline exact-byte reconciliation; handoff=separately scoped decision on interface/serialization diagnosis only, R10 HOLD unchanged.
