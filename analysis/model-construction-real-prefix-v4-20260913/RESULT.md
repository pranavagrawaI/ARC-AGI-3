# V4 result: provider routing repaired; local continuation validator aborted

**INTEGRATION_INVALID_LOCAL_CONTINUATION_CONTRACT.** The sole authorized episode made one actual request and received a successful **Reka** completion from the same exact `qwen/qwen3.8-27b`. Thus the corrected first request passed strict provider parameter routing. One model-selected visible-prefix inspection completed. Before a second request could be sent, the new local validator rejected response-only tool-call metadata. No retry, alternate provider, candidate submission, suffix evaluation, deployment or commit followed.

## Actual execution

| Measure | Observed |
|---|---:|
| Actual requests / completed provider responses | 1 / 1 |
| Inspections / candidate submissions | 1 / 0 |
| Provider-reported prompt / completion tokens | 1,871 / 78 |
| Provider-reported reasoning tokens | 0 |
| Provider-reported total cost | **$0.000599294** |
| Request latency | 2.221329 seconds |
| Inspection wall time | 0.027761 seconds |
| Total episode wall time | 2.261 seconds |
| Current completion reservation, not actual usage | 4,096 tokens |
| Conservative prompt reservation, not actual usage | 11,476 tokens |
| Structured work item / candidate identity | None / none |
| Candidate visible checks / suffix evaluation | Not run / not run |

All actual attempts have a receipt; **there is no unknown-cost attempted request in this episode**. The original runner's generic exception summary incorrectly says usage is unknown. `POST-RUN-AUDIT.json` derives known usage and cost from the unchanged raw receipt, rather than rewriting that original result.

The model autonomously chose `execute_python` to inspect the visible case's keys, initial/current grid dimensions and first visible transition. The isolated inspection returned successfully. Its rendered output is durable in `raw/tool_events.jsonl`, but was **not delivered to the model**, because construction stopped before sending the continuation. It did not submit a structured work item or code candidate; there is no candidate hash or semantic check outcome to report.

## What changed and what was established

Public endpoint/model discovery plus official documentation are preserved in `provider-endpoints.json`, `model-metadata.json` and `docs-*.txt`. Reka `reka/fp8` remained advertised active and supported all essential optional fields. The outgoing request removed only gratuitous unsupported `parallel_tool_calls:false`; it retained tools, required tool choice, `max_tokens`, and the OpenRouter unified `reasoning:{enabled:false}` control. `only`/`order` remained pinned to Reka, `allow_fallbacks:false`, `require_parameters:true`, and unchanged max-price caps.

The complete top-level and nested request review is in `REQUEST-CONTRACT.md`. Model metadata said reasoning is not mandatory but defaults on, so omitting the explicit control would be wrong. No unadvertised `reasoning_effort:none` substitution or display-only `exclude`/`include_reasoning` workaround was used. The actual receipt reports provider Reka, exact model, zero reasoning tokens, and null message reasoning.

**First-request provider compatibility is established; complete multi-turn integration is not.** V3's error did not name the rejected parameter. The v4 success is consistent with the removal fixing it, but this is not a controlled causal comparison across time.

## Exact new blocker

`RESULT.json` records:

```
ValueError: request contract: tool call envelope
```

Reka's actual tool-call envelope contained the standard function-call fields **plus `index:0`**:

```
choices[0].message.tool_calls[0].index = 0
```

The new validator required exactly `id`, `type`, and `function`. The constructor retained the raw assistant message for the next turn, so this response metadata tripped the outbound check after the inspection ran. **This was a host integration bug introduced by the stricter validation, not a provider transport rejection or model failure.** The prelaunch continuation fixtures omitted this field and failed to cover the real provider response shape.

`post_run_audit.py` reproduces the exact validator failure using the actual recorded response and tool result in an explicitly reconstructed, offline continuation. Removing only response `index` from a local copy makes that schema check pass. This is a diagnostic, not a repaired or resent request, and it makes zero provider calls, inspections or suffix evaluations. It does not claim to recover exact unsent telemetry.

## Verification and audit limits

Before launch, **28 reference tests, 36 local tests, reference SHA256 checks and all 17 prelaunch checks passed**. The preflight exercised explore and commit request shapes, unsupported essential/extra parameters, reasoning controls, context limits, sequential multiple-tool budgeting, mock HTTP success/failure, and isolation tests. Its passing result did not establish the untested real response shape above.

The frozen original `AUDIT.json` is **failed: 32 of 35 checks passed**. Its three failures are retained and explained:
- `usage_reconciles`: generic exception handler omitted known successful usage from `RESULT.json`.
- `request_allowance_reconciles`: exception summary omitted the already reserved allowance.
- `exact_transcript`: construction exception bypassed aggregate transcript/summary persistence, although the actual request, response and inspection event survived.

The separate failure-specific `POST-RUN-AUDIT.json` passes **18 of 18 checks**: frozen and prior hashes, exact wire/receipt preservation, known accounting, actual provider identity, reasoning receipt, bounded abort, model-selected inspection identity, no candidate/suffix, and offline root-cause isolation. **This does not override the failed original audit or upgrade the episode's invalid classification.** Machinery consistency and semantic evidence remain distinct. The v4 generic audit's zero-response regression now explicitly returns `NOT_ESTABLISHED` for provider match instead of vacuous success.

Baseline/v2/v3 manifests remain unchanged. The core host-owned verifier and trace loader are byte-identical to v3; candidate current-argument-only ABI isolation and singleton joint commitments were not weakened. All v4 source/card/preflight/doc/test hashes in `STARTED.json` remain frozen. The post-run audit script and report were added afterward and are explicitly not prelaunch evidence. Unrelated dirty work was not edited. `PERFORMANCE.json` remains historical inherited evidence, not a v4 speed claim.

## Interpretation and permitted next action

Retain this as first-request routing success plus a local continuation-contract failure. Do not infer candidate ability. A future separately authorized run must project provider responses into reviewed outbound message fields while retaining raw responses, test that path with this exact `index`-bearing fixture, and persist/reconcile progress on local pre-send failures. No paid continuation or new episode is authorized by this report.

Historical prefix steps 1–5 and withheld suffix 6–8 were already known to designers; the constructor received prefix evidence only. No untouched holdout, full finite-domain prototype, or game-general competence claim is supported.

## Absolute artifact paths

- Report: `/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/model-construction-real-prefix-v4-20260913/RESULT.md`
- Original failed audit: `/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/model-construction-real-prefix-v4-20260913/AUDIT.json`
- Failure-specific audit: `/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/model-construction-real-prefix-v4-20260913/POST-RUN-AUDIT.json`
- Raw evidence: `raw/requests/00.body`, `raw/responses/00.body`, `raw/request_attempts.jsonl`, `raw/model_calls.jsonl`, `raw/tool_events.jsonl` in the same directory.
