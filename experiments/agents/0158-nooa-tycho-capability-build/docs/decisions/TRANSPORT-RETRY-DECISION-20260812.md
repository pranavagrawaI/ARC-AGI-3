# Transport retry decision — 2026-08-12

HTTP 429 is retryable at the model-provider boundary because no ARC action occurs inside a provider call. The candidate remains pinned to `coreweave/bf16`, fallback remains disabled, and every crossing remains audited.

For each model call, permit at most five retries after the initial attempt. Honor numeric `Retry-After` values up to 120 seconds; otherwise wait exponential delays of 2, 4, 8, 16, and 30 seconds plus bounded jitter. A successful retry continues the same predeclared run. Exhaustion closes it. Other HTTP failures and uncertain read/write failures remain non-retryable and fail closed.

The 200-crossing global cap, 5,000,000 reported-token cap, 90-minute active-time cap, 12 CodeAct iterations per artifact, one-action protocol, and non-resumable scorecard remain unchanged.
