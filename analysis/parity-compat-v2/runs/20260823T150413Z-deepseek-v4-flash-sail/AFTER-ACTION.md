# DeepSeek compatibility ladder: after-action report

## Judgment

**Partial experiment with a decisive advancement result.** Native received three clean responses and failed its frozen termination prerequisite. CodeAct is unresolved because `call-0005` contained a Sail Research 502. The pair is incomplete, but eligibility required both arms to pass, so DeepSeek did not establish eligibility for the matched cognitive experiment.

The machine disposition `rejected_model_protocol` remains preserved as raw output. It must not be broadened into a claim that DeepSeek is generally incompatible with either protocol.

| Dimension | Native | CodeAct | Pair / advancement |
|---|---|---|---|
| Integrity | Valid | Provider-censored | Incomplete |
| Outcome | Failed termination | Unknown | Not eligible |
| Reason | Three clean calls, no valid return | Upstream 502 censored the arm | Native prerequisite failed |

## What happened

| Arm | Calls | Clean calls | Valid terminal returns | Observation |
|---|---:|---:|---:|---|
| Native | 3 | 3 | 0 | No required typed return was issued. |
| CodeAct | 3 | 2 | 0 | One malformed typed return was rejected; the next call ended in an upstream 502; the final clean call did not terminate. |

Later gates were not reached. No claim about CodeAct termination, evidence querying, RLE, representation, cognition, or Native-versus-CodeAct quality is supported.

## Route and cost

All six responses reported the authorized model and provider. There were no route violations. The frozen pricing schedule accounts for **$0.000460885** across **3,917 input** and **1,146 output tokens**. The provider's upstream-cost fields sum to the same amount; its top-level usage cost omitted the failed call and sums to $0.00027650.

## Causal defect

`call-0005` preserved `provider_specific_fields.error` with code 502 and the message `generation ended before the streamed tool call was complete`. The launcher validated route and cost but did not inspect this embedded provider error before allowing another call. That violated the precommitted provider-failure stop condition.

## Next

1. The embedded-provider-failure launcher defect was repaired in `78e0157`.
2. Repair the trusted authorization/recovery quality gate before further paid work.
3. Do **not** rerun this eligibility experiment to rescue DeepSeek: Native already failed the conjunctive prerequisite.
4. If CodeAct compatibility itself is worth resolving, freeze a separate diagnostic and obtain fresh authorization.
