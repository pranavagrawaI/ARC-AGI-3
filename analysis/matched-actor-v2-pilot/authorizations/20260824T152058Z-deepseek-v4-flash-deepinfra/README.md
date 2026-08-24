# DeepSeek Matched Actor V2-Pilot authorization

Pranav authorized this exact frozen execution in the task on 2026-08-24.

- requested model: `openai/deepseek/deepseek-v4-flash-0731`
- reported model: `deepseek/deepseek-v4-flash-0731`
- provider endpoint: `DeepInfra` (`deepinfra/fp8`)
- reasoning: enabled; temperature: unset
- hard limits: 72 calls, 16,384 input tokens/call, 4,096 output tokens/call,
  120 seconds/session, and $3.00 total actual, observed, or reserved spend
- theoretical pricing-bound maximum at all token ceilings: $0.147456
- ARC environment actions: zero

`experiment.json` is enabled and hash-bound by `authorization.json`. `run.py --preflight` validates the package and exact live-client configuration without a
provider call. The paid run is single-use and preserves partial failures without
adaptive reruns.
