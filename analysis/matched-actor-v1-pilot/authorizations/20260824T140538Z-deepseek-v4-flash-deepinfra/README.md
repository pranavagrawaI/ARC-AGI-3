# DeepSeek Matched Actor V1-Pilot authorization request

This package binds the frozen six-packet pilot to:

- requested model: `openai/deepseek/deepseek-v4-flash-0731`
- reported model: `deepseek/deepseek-v4-flash-0731`
- provider endpoint: `DeepInfra` (`deepinfra/fp8`)
- reasoning: enabled
- temperature: unset
- hard limits: 72 calls, 16,384 input tokens/call, 4,096 output tokens/call,
  120 seconds/session, and $3.00 total observed-or-reserved spend
- theoretical pricing-bound maximum at all token ceilings: $0.147456
- ARC environment actions: zero

`authorization-request.json` is inert. Paid execution remains impossible until
Pranav explicitly authorizes these exact terms and the request is installed as
`authorization.json` in a subsequent committed change. `run.py --preflight`
then validates the hash-bound package without making a provider call.
