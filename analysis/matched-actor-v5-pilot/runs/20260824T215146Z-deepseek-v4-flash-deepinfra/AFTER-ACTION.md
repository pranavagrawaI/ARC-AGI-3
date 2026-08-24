# DeepSeek Matched Actor V5-Pilot after-action

## Judgment

The repaired run was operationally valid but scientifically incomplete. The
context/evidence plane prevented the prior request-size failure, yet only one of
six matched pairs produced completed observations from both arms. CodeAct does
not advance, and the run does not establish a Native-versus-CodeAct cognitive
winner.

Native remains the incumbent by the precommitted burden of proof, not because it
won this pilot.

## Integrity

- authorization: `20260824T215146Z-deepseek-v4-flash-deepinfra`
- route: 62/62 responses reported
  `deepseek/deepseek-v4-flash-0731` through `DeepInfra`
- provider calls: 62 of 72 authorized
- actual provider spend: `$0.007265036` of `$3.00` authorized
- prompt tokens: 88,992; completion tokens: 11,823
- largest conservative input bound: 9,107 of 16,384 tokens
- largest provider-reported prompt: 2,500 tokens
- ledger: six gates started and completed; `LadderCompleted` and `RunCompleted`
  recorded; no provider, route, budget, or integrity stop
- post-run checkpoint verification: the focused V5 checkpoint/manifest suite
  passed (`5 passed`)
- ARC actions: none were exposed by either treatment

The run therefore contradicts the earlier hypothesis that large evidence values
must make the pilot cross its input ceiling. Receipts, bounded views, history
collapse, and provider preflight kept every request within the frozen cap.

## Outcomes

| Packet | Native | CodeAct | Pair status |
| --- | --- | --- | --- |
| `action-success-grounded` | six-call exhaustion | six-call exhaustion | incomplete |
| `action-failure-illegal-bait` | correct in one call | correct in one call | complete |
| `history-success-level1-noop` | six-call exhaustion | six-call exhaustion | incomplete |
| `objective-success-level-boundary` | returned on call six, incorrect | six-call exhaustion | incomplete |
| `cross-level-success-action-response` | six-call exhaustion | six-call exhaustion | incomplete |
| `remote-failure-composition` | six-call exhaustion | six-call exhaustion | incomplete |

Across all attempted work, Native and CodeAct each consumed 31 calls. Native
used 48,260 prompt tokens, 8,032 completion tokens, and `$0.004012224`.
CodeAct used 40,732 prompt tokens, 3,791 completion tokens, and `$0.003252812`.
These totals are operating observations, not efficiency comparisons, because
five pairs lack two completed answers.

## Causal observations

The surviving failure is no longer context overflow. It is inability to finish
the cognitive protocol within six calls.

- CodeAct repeatedly spent its runway discovering or guessing the Python
  evidence contract (`import evidence`, `dir`, `help`, `inspect.signature`,
  positional `evidence.query`, and missing required keywords). It completed no
  non-control packet.
- Native reached authoritative receipts and bounded views, but repeatedly
  repaired invalid limits, cursors, projections, or multi-tool turns. It
  completed one substantive packet only, and that answer was wrong.
- On the final composition packet, the context plane paged the first 32 of 59
  transitions without overflowing. Both treatments nevertheless exhausted
  before acquiring, comparing, and returning all required evidence.

Thus the run supports the context/evidence-plane repair, contradicts the claim
that three calls alone caused CodeAct's failure, and leaves the production
Native-versus-CodeAct judgment unresolved.

## Next gate

Do not pay for the same pilot again. First construct zero-cost scripted golden
paths for the five incomplete packets using only the exact model-visible
contracts. Record the minimum operations and calls required for each arm. Then
compare those paths with the preserved traces to separate:

1. undocumented or incongruent interface behavior;
2. avoidable model repair loops;
3. tasks whose minimum valid path cannot fit the six-call envelope.

Only a demonstrated, versioned repair to one of those mechanisms should earn a
new paid authorization.
