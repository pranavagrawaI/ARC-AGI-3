# GPT-5.6 Luna compatibility run: after action

## Judgment

The run is invalid for compatibility scoring. One Native call was dispatched and accounted at USD 0.000109; CodeAct was not run. The launcher stopped because its synchronous response object exposed the expected model but did not expose provider or reasoning-route fields, so the frozen route predicate failed closed.

## Preserved evidence

OpenRouter's generation metadata, fetched afterward without an inference call, identifies the serving provider as OpenAI and the model permaslug as `openai/gpt-5.6-luna-20260709`. This supports that routing was operationally correct; it does not repair the precommitted observability failure.

The model proposed `return_result({"result": {}})` on its first Native turn rather than the required typed answer. Because the launcher censored that response before the controller processed it and the gate allowed three calls, this is an observation about one attempt, not a termination-gate failure.

## Cause

The route contract assumed provider and reasoning metadata would be available directly on NOOA/LiteLLM's Responses object. For this OpenRouter Responses route, provider identity was available only through the separate generation record, while reasoning was represented as a structured `reasoning` field rather than `reasoning_effort`.

## Next

Do not retry this authorization. Before any new paid run, decide whether route integrity should be validated from request commitments plus post-response generation metadata, or whether the experiment should use a provider interface that exposes those fields synchronously. That is an infrastructure decision requiring a new manifest and authorization.
