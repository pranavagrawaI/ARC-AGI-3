# After action: session timeout invalidated the screen

## Disposition

**Invalid; no cognitive comparison claim.** The first native session was cancelled before either arm produced a scored report.

## Observed trace

- Five native calls were reserved; four completed and the fifth was cancelled.
- Completed calls consumed 0.004857385 USD. Including prior attempts, observed cumulative spend is 0.01319596 USD.
- The four completed calls took about 3.25, 47.81, 41.42, and 9.85 seconds. The fifth was cancelled about 17.60 seconds after dispatch.
- The 120-second whole-session cap expired inside the fifth provider call. `remote-success-old-up/failure.json` records `TimeoutError`.
- No CodeAct call, scored report, canonical mutation, or ARC action occurred.

## Causal diagnosis

The provider remained on the frozen route, but the session wall cap was inconsistent with the precommitted allowance of up to eight provider calls: ordinary route latency exhausted the whole-session deadline before the native controller could finish. This is a harness invalidity, not evidence about native or CodeAct cognition.

## Next discriminating move

Preserve this run, then authorize a new immutable retry with the same corpus, prompts, model route, call/output/cost caps, and authority. Change only the per-session wall ceiling to 600 seconds, and debit these five calls and their observed cost from the user's cumulative 3 USD authorization.
