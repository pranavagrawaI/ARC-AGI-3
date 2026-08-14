# Cheaper model replacement research

Date checked: 2026-08-13 UTC

Scope: find a cheaper replacement for the `qwen/qwen3.6-27b` OpenRouter route used by the terminal 44-action LS20 run. This is a source-and-cost investigation only; no candidate model was called.

## Decision

Qualify these two models before changing the agent around either one:

1. **Primary candidate: `deepseek/deepseek-v4-flash-0731`, pinned to `deepseek/fp8`.** It is an official DeepSeek endpoint with the strongest combination of price, cache economics, long context, tool support, and provider reliability. Repricing the completed Qwen run's exact token mix gives **about $0.73**, versus its actual **$5.31**: an 86% reduction. The host must first pass a no-environment tool-loop compatibility check, especially for thinking-mode state and the unsupported `medium` reasoning-effort value.
2. **Matched cost challenger: `qwen/qwen3.7-flash`, pinned to `alibaba`.** It retains the Qwen family and OpenRouter describes it as aimed at multimodal agents and spatial understanding. The same trace costs **about $0.43 with the observed cache-hit mix, or $0.65 with no cache hits**. It has only one OpenRouter provider and does not advertise `structured_outputs` even though it does advertise tools, `tool_choice`, and `response_format`, so it needs a stricter schema/tool preflight.

Do **not** use V4 Pro as the default loop model. `deepseek/deepseek-v4-pro-0813` is real and officially available, but the same token mix is still about **$2.25**. It is a possible later escalation/control model, not the economical engineering substrate.

## DeepSeek V4 is officially available

DeepSeek's official API documentation lists both `deepseek-v4-flash` and `deepseek-v4-pro`, with 1M context, up to 384K output, JSON output, tool calls, and thinking/non-thinking modes. The prices visible when checked are listed below. The same official page announces a pricing change effective **2026-08-16**, so these values must be refreshed immediately before a paid run:

| Vendor model | Fresh input / 1M | Cache hit / 1M | Output / 1M |
|---|---:|---:|---:|
| `deepseek-v4-flash` | $0.14 | $0.0028 | $0.28 |
| `deepseek-v4-pro` | $0.435 | $0.003625 | $0.87 |

Sources: [DeepSeek models and pricing, including the effective-date notice](https://api-docs.deepseek.com/quick_start/pricing/), [DeepSeek V4 announcement](https://api-docs.deepseek.com/news/news260424/), and [DeepSeek live model-list contract](https://api-docs.deepseek.com/api/list-models).

DeepSeek says Flash has reasoning close to Pro and matches Pro on *simple* agent tasks, while falling behind on harder tasks. That is a vendor claim, not evidence for LS20. It makes Flash a plausible candidate, not a demonstrated replacement.

OpenRouter currently exposes versioned V4 IDs. Prefer them over the unversioned OpenRouter aliases: the live inventory currently reports the aliases' canonical snapshots as the older 2026-04-23 preview, while the IDs below bind the later GA revisions.

- `deepseek/deepseek-v4-flash-0731`: GA Flash revision, described as suited to coding, reasoning, and agent workflows; [model page](https://openrouter.ai/deepseek/deepseek-v4-flash-0731), [live endpoint inventory](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-flash-20260731/endpoints).
- `deepseek/deepseek-v4-pro-0813`: GA Pro revision; [live endpoint inventory](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-pro-20260813/endpoints).

The live OpenRouter model inventory reports both IDs as accepting `tools` and `tool_choice`; Flash also advertises `structured_outputs` and parallel tool calls. Both expose optional reasoning and 1,048,576-token model context. This was checked against [`GET /api/v1/models`](https://openrouter.ai/api/v1/models), not inferred from an old model name.

## Exact candidates and routes

Prices and endpoint properties below are from OpenRouter's live endpoint API on 2026-08-13. Uptime is the API's rolling 30-minute field at retrieval time, not an SLA.

| Candidate and exact provider tag | Context / max completion | Input / cache-read / output per 1M | 30m uptime at check | Counterfactual cost |
|---|---:|---:|---:|---:|
| **V4 Flash 0731 `deepseek/fp8`** | 1,048,576 / 384,000 | $0.14 / $0.0028 / $0.28 | 99.997% | **$0.731** |
| V4 Flash 0731 `sail-research/fp4` | 262,144 / 262,144 | $0.072 / $0.016 / $0.144 | 98.832% | $0.439 |
| V4 Flash 0731 `deepinfra/fp4` | 1,048,576 / 384,000 | $0.08 / $0.016 / $0.18 | 98.283% | $0.494 |
| **Qwen3.7 Flash `alibaba`** | 1,000,000 / 65,536 | tiered; $0.03 / $0.006 / $0.13 below 32K, $0.10 / $0.02 / $0.40 at 32K-256K | 100% | **$0.426 cached; $0.653 uncached** |
| V4 Pro 0813 `deepseek` | 1,048,576 / 384,000 | $0.435 / $0.003625 / $0.87 | 99.994% | $2.251 |
| Current Qwen3.6 27B `coreweave/fp8` | 262,144 / 262,144 | $0.60 / $0.12 / $3.60 | 100% | **$5.310 actual and recomputed** |

DeepSeek has already announced new peak/off-peak direct-API prices effective
2026-08-16 16:00 UTC. If OpenRouter passes those prices through unchanged, the
same fixed token mix would put V4 Flash at about **$1.32 off-peak** or **$2.63
peak**, rather than $0.73. This projection must be refreshed from the live
OpenRouter endpoint before any paid run; it is not a quote. Even the projected
off-peak price remains much lower than the completed Qwen run, but Qwen3.7 Flash
would be the more stable cost candidate if its tool/schema qualification passes.

Qwen3.7 Flash source: [OpenRouter model page](https://openrouter.ai/qwen/qwen3.7-flash) and [live endpoint inventory](https://openrouter.ai/api/v1/models/qwen/qwen3.7-flash/endpoints). The endpoint advertises tools and a 1M context, but pricing is per-request tiered. Of the historical run's 539 calls, 54 were at or above 32K input tokens and none reached 256K.

The lowest V4 Flash routes are tempting but not the first choice. `sail-research/fp4` and `deepinfra/fp4` are quantized FP4 routes with materially lower short-window uptime than the official DeepSeek route. Since this project will engineer around the selected model, saving another $0.24-$0.29 per full run is not worth initially confounding model choice with provider/quantization differences. The official DeepSeek route also has automatic caching and exceptionally cheap cache reads. OpenRouter documents DeepSeek caching as automatic; Alibaba Qwen caching may require explicit cache controls depending on the endpoint and API surface. See [OpenRouter prompt caching](https://openrouter.ai/docs/guides/best-practices/prompt-caching).

## Cost calculation from the completed run

The completed Qwen trace reports:

- 539 calls;
- 8,027,530 input tokens;
- 4,314,352 of those input tokens served from cache;
- 3,713,178 uncached input tokens;
- 712,388 output tokens;
- 8,739,918 total reported tokens.

For a fixed-price endpoint, the counterfactual is:

```text
uncached_input * input_rate
+ cached_input * cache_read_rate
+ output * output_rate
```

Applying the current Qwen route's prices reproduces the provider bill exactly:

```text
3.713178M * $0.60 + 4.314352M * $0.12 + 0.712388M * $3.60
= $5.31022584
```

That validates the accounting method for the historical Qwen usage. Candidate totals remain **counterfactual estimates**, not quotes: a different model can produce a different number of reasoning/output tokens, tool iterations, retries, and cache hits. Qwen3.7's cached estimate also excludes any cache-write tokens because the historical ledger reports none; its no-cache estimate is the safer upper comparison for the same token volume.

## Compatibility risks to test before any game action

1. **Thinking-mode continuation.** DeepSeek documents that tool use is supported in thinking mode, but integrations may have to preserve the model's reasoning content on subsequent calls. The current artifact runner uses multi-call tool loops, so test a complete `execute_python` loop before touching ARC. See [DeepSeek tool calls](https://api-docs.deepseek.com/guides/tool_calls/) and its [Copilot integration warning](https://api-docs.deepseek.com/quick_start/agent_integrations/copilot_cli).
2. **Reasoning effort mismatch.** The completed run requested `medium`. The V4 endpoints advertise `low`, `high`, and `max`, not `medium`; Qwen3.7 advertises reasoning but not OpenRouter's effort enum. Do not allow `drop_params: true` to silently choose the experimental condition. Bind and record the intended mode explicitly.
3. **Tool schema fidelity.** Run the real `execute_python` schema, including nested arrays/unions and failed-tool recovery, rather than a toy weather tool. DeepSeek's native strict schema mode is beta and narrower than arbitrary JSON Schema, although OpenRouter's V4 Flash route advertises structured outputs.
4. **Responses API behavior.** The project uses OpenRouter's `/responses` surface. Check tool-call IDs, reasoning continuation, cache accounting, and termination on that exact surface and exact provider pin.
5. **Output-cost divergence.** Flash models are cheap per output token, but an unconstrained reasoning loop can still multiply calls. Preserve the existing call and token caps during qualification.

## Smallest discriminating qualification

Before a 44-action run, replay the same frozen, no-environment artifact inputs against both finalists:

1. one complete tool-loop/schema preflight;
2. the same small set of historical claim/commit artifacts spanning short and greater-than-32K prompts;
3. reasoning configuration bound explicitly rather than dropped;
4. record exact provider tag, usage, cache reads/writes, invalid artifacts, tool-loop completion, and model-authored commitments.

Selection should be based on valid tool-loop completion and the existing semantic commitments, not vendor benchmark prose. If both pass, prefer V4 Flash on the official DeepSeek route for the first bounded live run; keep Qwen3.7 Flash as the matched low-cost challenger. The later game/objective/constraint capability should then be developed against the selected model, rather than mixed into this model comparison.

## Rejected as the main replacement

- **`deepseek/deepseek-v4-pro-0813`:** legitimate and capable-looking, but still around $2.25 on this trace; reserve it for escalation or a control.
- **`openai/gpt-oss-120b`:** very cheap and tool-capable, but its 131,072-token context is below the historical trace's 141,814-token maximum input before allowing for completion, so it is not a drop-in full-run replacement without truncation or interface changes. [OpenRouter model page](https://openrouter.ai/openai/gpt-oss-120b).
- **Free endpoints:** useful for disposable smoke tests, not a stable model/provider substrate around which to engineer this agent without first establishing availability, routing, schema fidelity, and continuity.
