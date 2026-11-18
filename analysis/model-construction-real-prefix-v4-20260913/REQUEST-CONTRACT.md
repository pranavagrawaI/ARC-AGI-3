# Complete request contract: v4

## Evidence and interpretation

Public, unauthenticated exact-model endpoint discovery is preserved in `provider-endpoints.json`; the exact model record from GET `/api/v1/models` is in `model-metadata.json`. Selected Reka `reka/fp8` remains active, advertises tools, required tool choice, max_tokens and reasoning, $0.214/M input and $2.55/M output, 262144 context and 131072 completion capacity. No need to change provider or model to meet advertised essentials. Discovery does not guarantee a completion.

Official documentation read in the browser (direct markdown fetch was HTTP 403; generic extraction backend unavailable):
- https://openrouter.ai/docs/api_reference/parameters — `docs-parameters.txt`
- https://openrouter.ai/docs/guides/best-practices/reasoning-tokens — `docs-reasoning.txt`
- https://openrouter.ai/docs/guides/routing/provider-selection — `docs-routing.txt`

## All outgoing top-level fields

| Field | Treatment / evidence |
|---|---|
| model | Required routing identity; exact `qwen/qwen3.8-27b`, no aliases, alternate model or model fallback list. |
| messages | Chat envelope, text-only system/user/assistant/tool content; checked roles, message options, function call/result associations; initial compact prefix and later inspection results only. |
| provider | OpenRouter routing envelope, **not** a model supported-parameter name. Nested fields exactly `only`, `order`, `allow_fallbacks`, `require_parameters`, `max_price`. `only`/`order` pinned to `reka/fp8`, fallbacks false, require_parameters true, max_price.prompt=1 and completion=10 ($/M). Official provider routing documentation establishes semantics. |
| tools | Essential optional model field advertised by endpoint. Function name/description/object parameters only, no strict tool option or server tools; parameter shapes checked. |
| tool_choice | Essential optional field advertised; specifically `supports_tool_choice.required=true`. Required can return **one or more** tools. Each is processed sequentially and charged against host attempt budgets, including repairs; no parallel execution. |
| max_tokens | Essential optional field advertised. Actual allowance at most 4096, lower in bounded commit phase; cumulative actual completion limit 16384. Conservative prompt reservation plus completion must fit endpoint context; max_prompt_tokens checked if advertised. |
| reasoning | Essential OpenRouter **unified gateway control**, advertised as `reasoning`, exactly `{enabled:false}`. This does not assert upstream field-name equivalence. Model metadata explicitly says mandatory=false and default_enabled=true, so omitting control would be wrong. `supported_efforts` does not include none; do **not** invent equivalence by replacing enabled=false with reasoning_effort=none. Exclude/include_reasoning controls display, not disabling, and is not a substitute. |

Removed `parallel_tool_calls:false`: unnecessary optional flag absent from this endpoint's advertised support. V3's 404 identified only the parameter-filter stage, not the exact rejected field. This change tests a documented compatibility correction, not a previously proved sole root cause.

No temperature, top_p, top_k, stream, usage extension, seed, cache directives, include_reasoning, reasoning_effort, response_format, strict, or extra HTTP feature headers are added. Every top-level non-envelope field must be endpoint-advertised **and** in the minimal frozen field set. Unknown nested transport options fail closed. HTTP uses JSON POST with Content-Type and authorized Bearer header only; no SDK retries.

## Preflight and episode

`request_contract.validate_request` gates each actual request before transport and verifies field values, budgets, schemas and message associations. `REQUEST-PREFLIGHT.json` captures the real constructor's explore/commit request shapes via a mocked transport, clearly labeled zero paid calls. Tests also cover continuation with multiple tools, every essential parameter removed from advertised support, v3 unsupported flag, unknown fields, context limits and disabled-reasoning semantics.

The first actual construction request is the sole episode's compatibility test; no paid smoke request precedes it. If transport fails, stop without retry or fallback, preserve the exact request/error and attempt latency, and treat unreceipted usage and cost as unknown. A machinery audit pass is not semantic provider evidence: zero returned responses yields `NOT_ESTABLISHED` for actual provider match and actual reasoning-off.
