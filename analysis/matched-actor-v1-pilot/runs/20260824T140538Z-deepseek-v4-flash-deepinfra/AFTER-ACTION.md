# DeepSeek Matched Actor V1-Pilot — partial after-action

## Judgment

The treatment comparison is **inconclusive**. The provider route remained valid,
but the run stopped during the fourth pair when call 40 reported 51,715 input
tokens against the authorized 16,384-token per-call ceiling. No ARC action was
available or executed.

DeepSeek therefore neither advances nor loses the Native-versus-CodeAct
comparison. Only four of six pairs were attempted, and only three of eight
attempted arms returned accepted terminal results.

## What happened

All 40 responses came from the committed `DeepInfra` route and reported
`deepseek/deepseek-v4-flash-0731`. Calls 1–39 were accounted normally. During
the Native arm of `objective-success-level-boundary`, the model queried all
level-0 transitions without a limit:

```json
{"kind":"transitions","level":0}
```

The capability disclosed 16 full RLE transition records (50,785 characters).
The following provider request contained about 66,072 message characters and
DeepInfra reported 51,715 prompt tokens. The ledger preserved the response,
validated the route, rejected accounting, stopped the run, and recorded
`launch-failure.json`.

## Cost and preserved evidence

- provider calls received: 40
- ledger-accounted calls: 39
- ledger-accounted cost: $0.008180996
- provider-reported cost including rejected call 40: $0.012034884
- prompt tokens reported: 175,240
- completion tokens reported: 17,557
- reasoning tokens reported: 8,352
- ARC actions: 0

The difference between provider-reported and ledger-accounted cost is call 40:
the paid response existed, but the ledger refused to account it as authorized
after discovering the input-cap violation.

## Partial observations, not a treatment result

- `action-failure-illegal-bait`: both arms completed.
- `action-success-grounded`: both arms exhausted six calls.
- `history-success-level1-noop`: Native completed; CodeAct exhausted six calls.
- `objective-success-level-boundary`: CodeAct exhausted six calls; Native was
  censored by the input-cap violation.
- the two composition packets were never started.

These observations are insufficient for the frozen advancement rule and must
not be aggregated into a Native-versus-CodeAct winner.

## Causal diagnosis

The immediate cause was an unbounded full-evidence disclosure followed by a
provider request larger than the authorization allowed. Two distinct controls
are missing:

1. **Disclosure bound:** an evidence query needs pagination or a response-size
   budget so a valid call cannot inject an arbitrarily large page.
2. **Pre-dispatch bound:** the launcher needs a conservative request-size check
   that rejects a request before the provider call when it cannot fit the
   authorized input ceiling.

Raising the input ceiling alone would hide both defects. Any correction must be
frozen as a new treatment version; this run remains immutable evidence.
