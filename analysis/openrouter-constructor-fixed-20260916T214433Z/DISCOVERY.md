## Inherited discovery provenance

The files below are copied, hash-bound prior discovery, not fresh observations for this candidate. Transport/provider/prices remain unchanged by this task. A separate public endpoint refresh, if present, is recorded in discovery/fixed-endpoint-refresh.json. No credential loader, paid model call or preflight runs in preparation.

# Public transport discovery

Independent unauthenticated endpoint/models GETs returned HTTP 200; full original bytes and UTC/hash receipts are in `discovery/endpoints.raw`, `endpoints.receipt.json`, `models.raw`, `models.receipt.json`. Public documentation was read using the web tool. Separate direct urllib documentation fetches failed with HTTPError; errors are retained, not model calls. The reasoning/routing/metadata `.md` and HTML route attempts produced no retained bodies. No completion/preflight was sent.

Selected endpoint: **reka/fp8**, active status 0, quantization fp8, context 262144, maximum completion 131072; prompt $0.000000214/token, completion $0.00000255/token. Chutes FP8 was also advertised; it is not a fallback. Model canonical alias and reasoning options are extracted in `discovery/selected.json`. Metadata advertises reasoning optional, default enabled, efforts xhigh/medium/low. It does not prove actual disable enforcement or exact serving/checkpoint equivalence.

## Complete outgoing field audit

| Field | Authority / disposition |
|---|---|
| model | exact requested `qwen/qwen3.8-27b`; no model fallback array |
| messages | shared copied prompt/evidence and original tool feedback; OpenRouter chat envelope |
| tools | advertised endpoint supported_parameters; original tool schemas |
| tool_choice required | advertised supports_tool_choice.required = true |
| max_tokens | advertised; <=4096 requested combined budget, observed threshold |
| reasoning.enabled | model reasoning not mandatory; official nested reasoning control; live treatment verified from receipt |
| reasoning.exclude false | documented common control; exposes rather than hides reasoning |
| reasoning.effort medium | on only; model supported_efforts includes medium |
| stream false | documented gateway nonstreaming envelope |
| provider.only/order | documented full endpoint-tag pin; both singleton reka/fp8 |
| provider.allow_fallbacks false | documented routing envelope; no client retries either |
| provider.require_parameters true | documented supported-parameter filtering, not proof every nested flag works |
| provider.quantizations fp8 | documented routing filter |
| provider.max_price | documented dollars per MILLION token filter: prompt .214, completion 2.55 |
| X-OpenRouter-Metadata enabled | documented routing evidence request header; retained response metadata |
| Content-Type application/json / Authorization | HTTP/auth envelope; key never retained in request artifacts |

No other optional body fields or sampling settings. Inherited actual constructor wire has no sampling knobs to preserve. No parallel_tool_calls (absent from selected endpoint list), chat_template_kwargs, include_reasoning alias, reasoning.max_tokens, seed, transforms, plugins or response_format.

References: [endpoint metadata](https://openrouter.ai/api/v1/models/qwen/qwen3.8-27b/endpoints), [model metadata](https://openrouter.ai/api/v1/models), [reasoning](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens), [routing](https://openrouter.ai/docs/guides/routing/provider-selection), [router metadata](https://openrouter.ai/docs/guides/features/router-metadata), [chat schema](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion).

PM resolved the public documentation's provider-qualified combined-token behavior by explicitly treating 4096 as requested/observed and reserving the advertised maximum provider output allowance before each attempt. No hard provider-stop guarantee or remote cancellation guarantee is claimed. Returned routing metadata can corroborate the singleton route, not expose a hash-bound physical backend.
