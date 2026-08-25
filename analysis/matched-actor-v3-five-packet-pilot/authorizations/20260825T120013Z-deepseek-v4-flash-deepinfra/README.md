# DeepSeek Matched Actor V3 receipt authorization

Pranav authorized this exact run in the task on 2026-08-25 after the zero-cost
receipt-contract probe and five-packet replay passed.

- sole intervention from V2: congruent model-visible receipt contract
- exact route: `openai/deepseek/deepseek-v4-flash-0731` through DeepInfra only
- reported model: `deepseek/deepseek-v4-flash-0731`
- reasoning enabled; temperature unset
- 60 calls maximum; 16,384 input and 4,096 output tokens per call
- 120 seconds per session; $3.00 total actual, observed, or reserved spend
- theoretical all-ceilings pricing bound: $0.12288
- zero ARC environment actions

`authorization.json` binds the enabled manifest copy. `run.py --preflight`
validates the package and exact live-client configuration without a provider
call. Paid execution is single-use and preserves partial failures without reruns.
