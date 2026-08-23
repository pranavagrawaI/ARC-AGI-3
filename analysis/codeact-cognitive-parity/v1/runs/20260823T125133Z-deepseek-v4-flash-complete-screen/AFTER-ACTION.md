# After action: invalid native query arguments aborted the screen

## Disposition

**Invalid as a corpus screen; no comparative claim.** The first packet recorded a valid CodeAct exhaustion and the second packet then terminated on an unhandled native capability rejection.

## Observed trace

- The first packet completed native and recorded CodeAct call-budget exhaustion without stopping, confirming the prior continuation repair.
- On the second packet, native requested `frontier={"turn":59}` without a level and `limit=100` above the authoritative maximum of 32.
- The tool-facing schema accepted those shapes, but `EvidenceQuery` rejected them inside the controller. The validation error escaped instead of becoming tool feedback.
- Fifteen calls cost 0.005161815 USD. Including prior attempts, observed cumulative spend is 0.026033895 USD.
- No canonical mutation or ARC action occurred.

## Causal diagnosis

The provider made an invalid capability request, an expected event in an agent loop. CodeAct receives capability errors as data, while the native controller treated post-schema query validation as fatal. This remaining asymmetry invalidated the screen.

## Next discriminating move

Preserve the run. Convert native evidence-query validation failures into bounded tool feedback and count them as invalid interactions; do not broaden authority or query limits. Test recovery, then authorize a new immutable retry after debiting the fifteen calls and observed cost.
