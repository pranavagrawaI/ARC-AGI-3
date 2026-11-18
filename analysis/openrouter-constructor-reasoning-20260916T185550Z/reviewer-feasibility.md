# OR-CONSTRUCTOR-REASONING-REVIEW — feasibility stage

Sentinels acknowledged: **OR-REVIEW-NO-SPEND**, **OR-PAIR-REVIEW-BEFORE-SPEND**. Independent reviewer; implementation owner separate. Both dispatch files read in full. Public unauthenticated GETs only; no paid request, credential access, remote mutation, or source/control edit. Existing R10 HOLD unchanged.

## Disposition

**DONE: preparation is feasible from advertised capabilities; candidate/spend PASS is not issued.** Reka FP8 and Chutes FP8 both advertise the necessary broad capabilities. Endpoint-specific enforcement of reasoning disable and the combined output ceiling remains unproven. The candidate must make those uncertainty boundaries explicit rather than claim discovery metadata establishes the treatment. If strict pre-spend proof of a hard combined-token ceiling is required, the targeted public evidence below does not supply it: obtain endpoint-specific authoritative semantics before spend, or report that blocker. Do not silently weaken a cap.

This is a constructor protocol comparison, not a reproduction of identical Kaggle serving. A same-provider pair can be useful even though hosted weights, revision, quantization implementation, tokenizer/template, runtime, batching and physical backend are not hash-bound to Kaggle's snapshot. FP8 and the Qwen model name do not close that identity gap. Successful source/ABI replay would not establish model dynamics, gameplay efficacy or R10b acceptance.

## Live public evidence

Public GET `https://openrouter.ai/api/v1/models/qwen/qwen3.8-27b/endpoints` returned HTTP 200 at **2026-09-16T18:52:20.053266+00:00** (September 17 IST); response SHA-256 `0cdf91466fe10840843907774abd683b0ee49fe897eb7edf8251bb4f98250e6b`. The web reader rejected this URL; Python urllib performed the successful unauthenticated GET. No completion was requested.

| Endpoint tag | Quantization | Context | Maximum completion | Prompt/output USD per million | Required tool choice |
|---|---|---:|---:|---:|---|
| reka/fp8 | fp8 | 262144 | 131072 | 0.214 / 2.55 | advertised true |
| chutes/fp8 | fp8 | 262144 | 65536 | 0.24 / 2.20 | advertised true |

Both had status 0 and advertised reasoning, include_reasoning, max_tokens, tools, tool_choice and reasoning_effort. This is advertised service metadata, not an executed treatment check. [Live endpoint metadata](https://openrouter.ai/api/v1/models/qwen/qwen3.8-27b/endpoints).

The matching entry in [models metadata](https://openrouter.ai/api/v1/models) reports canonical_slug `qwen/qwen3.8-27b-20260814`, reasoning mandatory=false, default_enabled=true, supported_efforts=[xhigh, medium, low], default_effort=xhigh; no supports_max_tokens reasoning flag was present. The broad model context is 1,000,000, but the selected endpoints advertise 262,144: use the endpoint bound. A fresh, reduced snapshot is embedded below with its own response hashes/time.

## Specific routing and optional-field review

Use one identical full endpoint tag in `provider.only` and `provider.order`, with `allow_fallbacks:false`, `require_parameters:true`; optionally filter fp8 too. A base slug can match several variants. There must be one model, no model fallback array, no automatic SDK retry, no retry after strict-routing rejection. `max_price` uses USD **per million tokens**, unlike metadata's per-token prices. It filters routing, not total pair expenditure. [Provider routing](https://openrouter.ai/docs/guides/routing/provider-selection).

Recommended candidate routing shape (example only, not endpoint selection or approval):

```json
{"model":"qwen/qwen3.8-27b","provider":{"only":["reka/fp8"],"order":["reka/fp8"],"allow_fallbacks":false,"require_parameters":true,"quantizations":["fp8"],"max_price":{"prompt":0.214,"completion":2.55}},"max_tokens":4096,"stream":false}
```

Audit **every emitted field**, including SDK defaults, against endpoint capabilities or documented gateway-envelope semantics:

- `tools`, `tool_choice:"required"`, `max_tokens` are advertised for both candidates. Check tool-choice mode, not just the presence of tool_choice.
- Sampling fields present in the captured original request must be preserved identically or their shared transport projection disclosed. Do not introduce new seed/sampling knobs casually.
- `parallel_tool_calls` is a documented API field but is **absent from both endpoint supported_parameters lists**. Do not add it assuming false is harmless under strict routing. Preserve reviewed multiple-call handling locally.
- `chat_template_kwargs.enable_thinking` is the old SGLang control, not evidence of OpenRouter support. Do not forward it as an undocumented substitute.
- Do not add response_format/strict schema repair, plugins, transforms, reasoning summaries, reasoning.context, or legacy include_reasoning controls. They either alter the comparison or lack demonstrated need/support here.
- Treat stream/provider/routing fields as gateway envelope settings; do not falsely demand they appear in the provider's generation-parameter list. Conversely, require_parameters is not a theorem that every nested option is honored.

The API documents `parallel_tool_calls` and the opt-in `X-OpenRouter-Metadata: enabled` header. Use routing metadata as evidence without introducing a paid preflight. [Chat schema](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion). Returned metadata can identify selected candidates and successful attempt count; require attempt 1 and matching provider/model when present, retain contradictions, and do not manufacture endpoint identity from a provider display name alone. [Router metadata](https://openrouter.ai/docs/guides/features/router-metadata).

## Reasoning control and continuation

OpenRouter documents reasoning.enabled, effort, exclude, and ordered reasoning_details continuation. Excluding reasoning hides it rather than disabling computation; include_reasoning=false is likewise not an off switch. Default documentation says enabled=true uses medium, while live model metadata says default effort xhigh: pin an advertised effort explicitly for on. Reasoning is billed as output; combined max_tokens behavior is qualified as applying on most providers. Returned reasoning_details must retain the original block sequence. [Reasoning guide](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

Proposed explicit transport contrast: off `{"enabled":false,"exclude":false}`; on `{"enabled":true,"effort":"medium","exclude":false}` (or another predeclared advertised effort). Qwen's own client documentation explicitly uses OpenRouter's nested enabled=false disable field. [Qwen client documentation](https://qwenlm.github.io/qwen-code-docs/en/users/configuration/model-providers/). This supports the request spelling, not proof Reka/Chutes actually disable. Do not use effort=none merely because generic docs mention it: this model's advertised effort list excludes none. Do not infer support for a separate reasoning.max_tokens budget from general reasoning support.

**Concrete existing-engine incompatibility:** preserved `analysis/r10-prep-20260916/r10a-package/constructor.py:227` forces off; lines 252–262 reject nonzero reasoning usage or reasoning fields; lines 265–273 reconstruct assistant messages without reasoning. Its local prompt also says returned reasoning is fatal. The isolated adaptation needs an explicit on-arm protocol policy and lossless continuation, while retaining original prompt/schema as instructed. Do not hide reasoning from the ledger or pretend the preserved prompt has no such sentence. This comparison measures reasoning configuration under that preserved instruction, not a reasoning-optimized prompt.

Reviewer acceptance criteria for candidate fixtures and later actual-arm evidence:

1. Independent episodes start from identical hash-bound prefix/system/schema/evaluator, without historical submissions or error hints. Only the declared reasoning policy differs initially; subsequent trajectories may diverge naturally. Preserve off-first order.
2. Retain raw assistant content (including null), tool IDs/arguments and provider-shaped reasoning_details, including opaque/signature/index fields and array order. Echo the supported reasoning representation with matching tool results; never put reasoning into tool arguments or user prose. A raw reasoning-string response also needs a continuation path. Do not duplicate the same reasoning into multiple aliases gratuitously.
3. Off evidence requires explicit valid reasoning-token accounting of zero and no returned reasoning payload. Missing counts or hidden reasoning cannot be classified as verified off. On requires actual nonzero reasoning usage and compatible returned evidence somewhere in the episode; zero reasoning throughout means treatment not demonstrated. Per-call contradictions must be retained and stop/invalidate under a predeclared policy. A 200 or accepted flag alone is insufficient. Even these checks establish provider-reported behavior, not visibility into undisclosed internal computation.
4. Fixtures must exercise the actual HTTP/constructor path through inspection, tool feedback, another request, malformed candidate repair, accounting and termination. Include details-only reasoning, null-content tool responses, truncation, multiple calls, missing usage, wrong provider, strict rejection and first transport failure. No network calls are needed for these fixtures.

## Caps, money and stopping

At advertised uncached prices, the full pair ceilings cost:

- Reka: `1200000*0.000000214 + 49152*0.00000255 = $0.3821376`.
- Chutes: `1200000*0.00000024 + 49152*0.0000022 = $0.3961344`.

These are calculations under the advertised rates and combined-output assumption, not billed amounts. No discount/cache saving is required. The 600,000 prompt ceiling is cumulative per arm, not an admissible single request. Check the 262,144 context bound on each complete continuation, including tools and reasoning history; serialized bytes plus framing is a conservative engineering reservation, not a certified hosted tokenizer identity.

Usage is automatically included; deprecated usage/include_usage flags add no assurance. Record usage.cost as charged-account cost; upstream cost is a different field. Token counts use native tokenization, and reasoning details are reported when applicable. [Usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting).

Required implementation behavior (review criteria, not claims of existing qualification): reserve endpoint-priced worst case before dispatch, persist attempt/reservation, include all failed/truncated outputs, and release only reconciled unused reservation. Maintain shared **$2 pair** ledger, separate per-arm 6 requests/24576 total completion/600000 prompt ceilings, and 4096 per-request combined completion. Reasoning tokens are a component of completion; do not add them twice or subtract them to make a call fit. Require finite, nonnegative, consistent usage/cost and stop on missing receipt. Preserve unresolved reservations as unknown exposure after timeout; never label cost zero or proceed to arm two after an unreceipted attempt. First calls are arm calls, not free discovery; no warmup/preflight calls.

**Unresolved strict-cap limitation:** docs explicitly qualify combined-token limits by provider and even illustrate a reported total exceeding a requested limit. No targeted source found proves a hard 4096 combined ceiling for these exact endpoints. Response validation detects an overrun after it happened; it cannot prevent it. Likewise a 245-second local hard kill stops local work but does not prove a remote billable request was cancelled. Candidate must distinguish local containment from provider termination and retain monetary reservation for any in-flight attempt. Do not issue an unconditional cap guarantee based on generic docs.

Length-finished reasoning-only output is a paid, consumed request and an interpretable truncation result; do not retry or silently raise the output ceiling. Preserve 240 soft/245 hard seconds, inspection/submission limits, wall-clock admission and late-publication prohibition in the real entry path. Public metadata latency is not a timing guarantee.

## Handoff

No unrelated R10 gate is reopened. Candidate review remains necessary for exact sources/imports, same-prefix bindings, card, cost/reservation logic, reasoning policy/continuation, sealed mapped-user evaluation and local adversarial tests. Minimal protocol adaptation is implementation work for the separate owner; this review changes nothing. Remain NO-SPEND until candidate PASS and Hermes release. If endpoint-specific budget enforcement cannot be established to the required standard, report that precise blocker before paid execution.

SUPERSET_WORKER_DONE task=OR-CONSTRUCTOR-REASONING-REVIEW; summary=bounded public feasibility review complete, no spend approval; files=/tmp/openrouter-constructor-reasoning-feasibility.md; checks=unauthenticated live endpoint/model metadata and official docs, targeted preserved constructor source; handoff=exact-byte candidate review pending, strict combined-cap uncertainty explicit, existing R10 HOLD preserved.

## Discovery appendix: refreshed parsed public metadata

```json
[
  {
    "url": "https://openrouter.ai/api/v1/models/qwen/qwen3.8-27b/endpoints",
    "utc": "2026-09-16T18:55:21.414121+00:00",
    "status": 200,
    "full_response_sha256": "0cdf91466fe10840843907774abd683b0ee49fe897eb7edf8251bb4f98250e6b",
    "scope": "Parsed subset, not full raw response archive",
    "data": [
      {
        "name": "Reka | qwen/qwen3.8-27b-20260814",
        "model_id": "qwen/qwen3.8-27b",
        "model_name": "Qwen: Qwen3.8 27B",
        "context_length": 262144,
        "pricing": {
          "prompt": "0.000000214",
          "completion": "0.00000255",
          "input_cache_read": "0.00000015",
          "discount": 0
        },
        "provider_name": "Reka",
        "tag": "reka/fp8",
        "quantization": "fp8",
        "max_completion_tokens": 131072,
        "max_prompt_tokens": null,
        "supported_parameters": [
          "reasoning",
          "include_reasoning",
          "temperature",
          "top_p",
          "top_k",
          "max_tokens",
          "stop",
          "seed",
          "frequency_penalty",
          "presence_penalty",
          "response_format",
          "structured_outputs",
          "tools",
          "tool_choice",
          "logprobs",
          "top_logprobs",
          "reasoning_effort"
        ],
        "supports_tool_choice": {
          "none": true,
          "auto": true,
          "required": true,
          "function": true
        },
        "status": 0,
        "uptime_last_30m": 99.93528904227783,
        "uptime_last_5m": 100,
        "uptime_last_1d": 99.94787015067827,
        "supports_implicit_caching": false,
        "supports_voice_cloning": false,
        "latency_last_30m": null,
        "throughput_last_30m": null
      },
      {
        "name": "Chutes | qwen/qwen3.8-27b-20260814",
        "model_id": "qwen/qwen3.8-27b",
        "model_name": "Qwen: Qwen3.8 27B",
        "context_length": 262144,
        "pricing": {
          "prompt": "0.00000024",
          "completion": "0.0000022",
          "input_cache_read": "0.000000024",
          "discount": 0
        },
        "provider_name": "Chutes",
        "tag": "chutes/fp8",
        "quantization": "fp8",
        "max_completion_tokens": 65536,
        "max_prompt_tokens": null,
        "supported_parameters": [
          "reasoning",
          "include_reasoning",
          "max_tokens",
          "temperature",
          "top_p",
          "stop",
          "frequency_penalty",
          "presence_penalty",
          "seed",
          "top_k",
          "repetition_penalty",
          "response_format",
          "structured_outputs",
          "tools",
          "tool_choice",
          "reasoning_effort"
        ],
        "supports_tool_choice": {
          "none": true,
          "auto": true,
          "required": true,
          "function": true
        },
        "status": 0,
        "uptime_last_30m": 97.81906300484653,
        "uptime_last_5m": 98.06949806949807,
        "uptime_last_1d": 99.1415146011704,
        "supports_implicit_caching": false,
        "supports_voice_cloning": false,
        "latency_last_30m": null,
        "throughput_last_30m": null
      }
    ]
  },
  {
    "url": "https://openrouter.ai/api/v1/models",
    "utc": "2026-09-16T18:55:21.523533+00:00",
    "status": 200,
    "full_response_sha256": "c11f217a762e7d7cab1049ae4d80cabac947afc55229327f5c87e86e8b1cd94c",
    "scope": "Parsed subset, not full raw response archive",
    "data": {
      "id": "qwen/qwen3.8-27b",
      "canonical_slug": "qwen/qwen3.8-27b-20260814",
      "hugging_face_id": "Qwen/Qwen3.8-27B",
      "name": "Qwen: Qwen3.8 27B",
      "created": 1786722910,
      "description": "Qwen3.8 27B is an open-weight dense vision-language model from Qwen. It is suited for coding, professional workflows, research, multimodal interaction, and long-running agent tasks, with flexible thinking that can be...",
      "context_length": 1000000,
      "architecture": {
        "modality": "text+image+video->text",
        "input_modalities": [
          "text",
          "image",
          "video"
        ],
        "output_modalities": [
          "text"
        ],
        "tokenizer": "Qwen",
        "instruct_type": null
      },
      "pricing": {
        "prompt": "0.000000214",
        "completion": "0.00000255",
        "input_cache_read": "0.00000015"
      },
      "top_provider": {
        "context_length": 262144,
        "max_completion_tokens": 131072,
        "is_moderated": false
      },
      "per_request_limits": null,
      "supported_parameters": [
        "frequency_penalty",
        "include_reasoning",
        "logit_bias",
        "logprobs",
        "max_tokens",
        "min_p",
        "presence_penalty",
        "reasoning",
        "reasoning_effort",
        "repetition_penalty",
        "response_format",
        "seed",
        "stop",
        "structured_outputs",
        "temperature",
        "tool_choice",
        "tools",
        "top_a",
        "top_k",
        "top_logprobs",
        "top_p"
      ],
      "default_parameters": {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 20
      },
      "supported_voices": null,
      "knowledge_cutoff": null,
      "expiration_date": null,
      "links": {
        "details": "/api/v1/models/qwen/qwen3.8-27b-20260814/endpoints"
      },
      "benchmarks": {
        "design_arena": [],
        "artificial_analysis": {
          "intelligence_index": 33.9,
          "coding_index": 68.1,
          "agentic_index": 46.5
        }
      },
      "reasoning": {
        "mandatory": false,
        "default_enabled": true,
        "supported_efforts": [
          "xhigh",
          "medium",
          "low"
        ],
        "default_effort": "xhigh"
      }
    }
  }
]
```

## Local input bindings

- `/tmp/or-constructor-review-dispatch.md`: `be06bcdb374d5185128f48060b44caf826536d9fa567e14468e5fe97a1720367`
- `/tmp/or-constructor-pair-dispatch.md`: `6d10bdeef6830a57af112aa8a1100ea1b033d4ffb0723df777822eb06d7d8f25`
- `analysis/r10-prep-20260916/r10a-package/constructor.py`: `7c4e7ee61b55fb7a449ad27a9d290978605bc21552d8f5b3c3f94cf12bac640c`
