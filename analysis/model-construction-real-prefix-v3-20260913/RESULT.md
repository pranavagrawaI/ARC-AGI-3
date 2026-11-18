# V3 result: alternate provider selected, request blocked by parameter routing

**INTEGRATION_INVALID.** One authorized episode was launched and stopped on its first request: OpenRouter HTTP 404, “No endpoints found that can handle the requested parameters.” No retry, fallback, second episode, candidate evaluation, or deployment followed.

## Observed outcome

| Measure | Actual |
|---|---:|
| Request attempts / completed responses | 1 / 0 |
| Inspections / candidate submissions | 0 / 0 |
| Failed-request latency | 0.369683 seconds |
| Episode wall time | 0.378 seconds |
| Current completion reservation | 4,096 tokens |
| Conservative prompt reservation | 11,506 tokens |
| Conservative request dollar reservation | $0.052466 |
| Provider-reported prompt / completion usage | Unknown / unknown |
| Billed cost | **Unknown, not zero: no receipt** |
| Work item / candidate / visible assessment / suffix assessment | None / none / not run / not run |

## Provider evidence and blocker

The unauthenticated exact-model endpoint API advertised active **Reka**, endpoint tag **`reka/fp8`**, supporting `tools`, required `tool_choice`, and `reasoning`. Advertised prices were $0.214/M prompt and $2.55/M completion, inside unchanged $1/M and $10/M routing caps. This was the lowest advertised input-price non-DekaLLM endpoint satisfying those selection criteria. The exact metadata is in `provider-endpoints.json` and the selected endpoint/routing in `PROVIDER-SELECTION.json`.

The request kept **`qwen/qwen3.8-27b`**, reasoning `{"enabled": false}`, `tool_choice: required`, and `parallel_tool_calls: false`. Routing pinned both `only` and `order` to the discovered exact tag, disabled fallbacks, and enabled `require_parameters`.

The raw error records **15 initial endpoints → 1 allowed endpoint → failure at “Filter by Parameters.”** Thus the pin identified an endpoint, but the complete parameter combination did not pass routing. Public tools/reasoning support did **not** establish compatibility with the entire request. `parallel_tool_calls` is present in the request but absent from the advertised supported-parameter list; this is a plausible contributor, **not an established diagnosis**, because the error does not identify the rejected parameter. No further paid diagnostic call was made. **Actual serving provider is unknown/not established; no provider completion was returned.** Requested reasoning-off was verified, but actual reasoning usage cannot be measured without a response.

## Bounds and verification

Unchanged limits: 16 requests, 16,384 actual completion tokens, 300,000 cumulative prompt tokens, 900 seconds, 10 inspections, four submissions, $0.50; retain at least two requests and 4,096 completion tokens for commit/repair/evidence. Actual-token charging and current-request reservations remain distinct. All construction/sandbox logic is unchanged except explicit routing. Host-owned assessment/version/ancestry and current-argument-only ABI isolation remain intact.

Before launch: reference hash checks and **28 reference tests passed; 27 local tests passed; all 15 prelaunch checks passed**. New tests mock HTTP, exercise the real transport/constructor/artifact audit on both completed and first-request HTTP-failure paths, and check empty/partial/malformed receipts. No test fixture is presented as a live result.

After launch: **all 33 artifact-audit checks passed**. This proves bounded abort and artifact consistency, not successful integration, actual provider service, or known spend. The audit now returns null total usage/cost after a failed request instead of vacuous zero sums; attempt latency is recorded on both paths. Exact wire request bytes, JSON payload, raw error body and attempt ledger are durable.

V2 and baseline hashes remain unchanged. V3 source/card/tests and provider selection remain frozen in `STARTED.json`. `PERFORMANCE.json` is explicitly inherited historical baseline/v2 evidence, not a v3 speed measurement. No live throughput or speedup claim is supported.

## Interpretation and next action

This episode yielded routing diagnostics only, not a work item or model-capability evidence. Preserve it; do not repair routing in this identity or make another live attempt. A future separately authorized identity should verify the complete outgoing parameter set, including optional flags, before freezing. No suffix feedback was exposed to construction or used for repair. Historical prefix steps 1–5 and suffix 6–8 were already known to designers; no untouched holdout claim.

## Artifacts

- `RESULT.json`: unchanged original runner outcome.
- `POST-RUN.json`: explicit counts, reservations, timing and unknown usage/provider fields.
- `AUDIT.json`: independent artifact consistency checks.
- `PRELAUNCH.json`, `STARTED.json`: verification and frozen identity.
- `raw/requests/00.body`, `raw/requests/00.json`: exact request bytes and decoded payload.
- `raw/responses/00.error.body`, `raw/request_attempts.jsonl`: exact failure and latency.
- `PROVIDER-SELECTION.json`, `provider-endpoints.json`: public discovery and routing evidence.

Absolute report directory: `/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/model-construction-real-prefix-v3-20260913`.
