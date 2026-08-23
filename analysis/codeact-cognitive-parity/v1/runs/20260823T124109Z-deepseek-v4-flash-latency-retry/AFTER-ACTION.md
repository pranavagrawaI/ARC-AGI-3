# After action: malformed native provider turn invalidated the screen

## Disposition

**Invalid; no cognitive comparison claim.** The first native session terminated after a provider turn returned no tool call.

## Observed trace

- Two native calls completed for 0.00270415 USD. Including prior attempts, observed cumulative spend is 0.01590011 USD.
- Call 1 queried evidence normally.
- Call 2 returned `finish_reason="stop"`, empty content, and no tool calls. Its reasoning described the exact RLE evidence as corrupted or prompt injection.
- The native controller raised `RuntimeError` rather than returning corrective feedback. No CodeAct call, scored report, canonical mutation, or ARC action occurred.

## Causal diagnosis

The preceding repair made a zero-tool CodeAct turn recoverable but left the matched native controller fail-fast on the same provider behavior. That asymmetry makes the treatment invalid: controller robustness, not cognition, determined survival.

## Next discriminating move

Preserve this run; make malformed native turns consume one call, enter the trace as an invalid return, and receive explicit corrective feedback. Test that recovery uses no extra authority. Then authorize a new immutable retry, debiting these two calls and their observed cost from the cumulative 3 USD authorization.
