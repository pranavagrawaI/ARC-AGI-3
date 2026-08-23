# After action: a valid treatment failure aborted the corpus

## Disposition

**The packet-level reliability observation is valid; the corpus-level screen is incomplete.** No comparative corpus claim is authorized.

## Observed trace

- The first native arm returned a scored result after six calls.
- CodeAct used all eight allowed calls. Its first call executed an evidence query; the next seven produced no tool call and received corrective feedback. It then exhausted its session call budget.
- Fourteen calls cost 0.00497197 USD. Including prior attempts, observed cumulative spend is 0.02087208 USD.
- The experiment runner recorded the CodeAct failure but aborted before the remaining eleven packets. No canonical mutation or ARC action occurred.

## Causal diagnosis

CodeAct call-budget exhaustion is a valid reliability outcome under the frozen treatment, not an infrastructure invalidity. The harness nevertheless treated any arm failure as a fatal screen error. That prevents the precommitted execution-failure metric from being measured across the corpus and creates ordering dependence.

## Next discriminating move

Preserve this run. Make only treatment call-budget exhaustion nonfatal at the packet boundary: preserve the failed arm, still run its matched arm, mark the packet incomplete, and continue the corpus. Route drift, integrity failure, cost failure, and timeouts must remain fatal. Test both arm orders, then authorize a new immutable retry after debiting these fourteen calls and their cost.
