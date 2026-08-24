# DeepSeek Matched Actor V3-Pilot authorization

Pranav authorized this exact frozen request in the task on 2026-08-24.

- exact route: `openai/deepseek/deepseek-v4-flash-0731` through DeepInfra only
- reported model: `deepseek/deepseek-v4-flash-0731`
- reasoning enabled; temperature unset
- 72 calls maximum; 16,384 input and 4,096 output tokens per call
- 120 seconds per session; $3.00 total actual, observed, or reserved spend
- theoretical all-ceilings pricing bound: $0.147456
- zero ARC environment actions

`authorization.json` binds the enabled manifest copy. `run.py --preflight`
validates the package and configured live clients without a provider call. The
paid execution is single-use and preserves partial failures without reruns.
