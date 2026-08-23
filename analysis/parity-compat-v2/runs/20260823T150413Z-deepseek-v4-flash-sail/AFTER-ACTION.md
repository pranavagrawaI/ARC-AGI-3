# DeepSeek compatibility ladder: after-action report

## Judgment

**Invalid experiment — provider failure.** The paid run reached only `g0-termination`. The launcher recorded a Sail Research 502 embedded in `call-0005`, conservatively accounted for it, but continued the ladder and emitted the machine disposition `rejected_model_protocol`. Because provider failure was a hard-stop condition, that disposition is preserved as machine output but is not a valid eligibility result.

## What happened

| Arm | Calls | Clean calls | Valid terminal returns | Observation |
|---|---:|---:|---:|---|
| Native | 3 | 3 | 0 | No required typed return was issued. |
| CodeAct | 3 | 2 | 0 | One malformed typed return was rejected; the next call ended in an upstream 502; the final clean call did not terminate. |

Later gates were not reached. No claim about evidence querying, RLE, representation, cognition, or Native-versus-CodeAct quality is supported.

## Route and cost

All six responses reported the authorized model and provider. There were no route violations. The frozen pricing schedule accounts for **$0.000460885** across **3,917 input** and **1,146 output tokens**. The provider's upstream-cost fields sum to the same amount; its top-level usage cost omitted the failed call and sums to $0.00027650.

## Causal defect

`call-0005` preserved `provider_specific_fields.error` with code 502 and the message `generation ended before the streamed tool call was complete`. The launcher validated route and cost but did not inspect this embedded provider error before allowing another call. That violated the precommitted provider-failure stop condition.

## Next

1. Commit this run unchanged as evidence.
2. Add an explicit durable provider-failure transition after a received response is preserved and accounted.
3. Test that embedded provider errors hard-stop the pair and cannot produce a ladder disposition.
4. Do **not** rerun automatically. The completed authorization cannot be reused; a repaired treatment requires a new frozen manifest and fresh authorization.
