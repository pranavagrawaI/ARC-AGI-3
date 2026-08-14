# Provider switch — 2026-08-12

## Observation

Three consecutive corrected-candidate LS20 runs stopped on an HTTP 429 from the exclusively pinned `modelrun/fp4` route. The failures occurred on provider calls 4, 4, and 1, with the same error class and redacted-message hashes. No ARC action executed.

## Decision

Switch the next-run candidate to the exclusively pinned `coreweave/bf16` route with fallback disabled.

The live OpenRouter endpoint inventory for `google/gemma-4-31b-it` reported CoreWeave healthy with a 262,144-token context, tool calls, tool choice, structured outputs, and a completion allowance above the frozen 8,192-token requirement. CoreWeave changes both provider and quantization relative to ModelRun; any subsequent result belongs to this new provider configuration and is not a one-factor provider-only comparison.

No live run is authorized or executed by this change.
