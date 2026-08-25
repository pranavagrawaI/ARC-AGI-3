# DeepSeek Matched Actor V2 five-packet after-action

## Judgment

The run was operationally valid but scientifically incomplete. Only one of five
matched pairs produced accepted answers from both arms. CodeAct does not meet
the precommitted advancement rule. Native remains the production incumbent by
burden of proof, not because this pilot established Native superiority.

Do not repeat this paid pilot unchanged.

## Integrity

- authorization: `20260825T110716Z-deepseek-v4-flash-deepinfra`
- route: 53/53 responses reported `deepseek/deepseek-v4-flash-0731` through
  `DeepInfra`
- provider calls: 53 of 60 authorized
- actual provider spend: `$0.006786980` of `$3.00` authorized
- prompt tokens: 93,907; completion tokens: 8,257
- largest conservative input bound: 10,045 of 16,384 tokens
- largest total request bound: 14,141 of 20,480 tokens
- largest provider-reported prompt: 4,115 tokens
- ledger: five gates completed; `LadderCompleted` and `RunCompleted` recorded
- provider, route, budget, and integrity stops: none
- ARC actions: none were exposed by either treatment

The context/evidence plane therefore kept the treatment inside its resource and
authority boundaries. Context overflow was not the cause of this result.

## Outcomes

| Packet | Native | CodeAct | Pair status |
| --- | --- | --- | --- |
| `action-success-grounded` | six-call exhaustion | six-call exhaustion | incomplete |
| `action-failure-illegal-bait` | accepted on call 3; frozen evaluator incorrect because of unsupported evidence | accepted and correct on call 2 | complete |
| `history-success-level1-noop` | six-call exhaustion | decoded the frames on call 6, then exhausted | incomplete |
| `objective-success-level-boundary` | accepted on call 6, incorrect | six-call exhaustion | incomplete |
| `cross-level-success-action-response` | six-call exhaustion | six-call exhaustion | incomplete |

Native used 27 calls, 46,344 prompt tokens, 4,859 completion tokens, and
`$0.003189500`. CodeAct used 26 calls, 47,563 prompt tokens, 3,398 completion
tokens, and `$0.003597480`. These are operating observations, not comparative
efficiency estimates, because four pairs were incomplete.

## Causal observations

The zero-provider replay proves that all five packets have valid paths within
the six-call envelope. The paid failures therefore reflect model-policy and
interface interaction rather than an impossible deadline.

- CodeAct repeatedly spent calls inspecting receipts or namespace objects. On
  `action-success-grounded`, it obtained the target receipt by call 3, then
  guessed nonexistent `records` and `to_dict` attributes instead of loading the
  granted value. On `history-success-level1-noop`, it loaded and decoded the
  exact frames but used its final call for computation rather than returning.
- Native repeatedly acquired receipts and requested bounded views without
  converging. It returned on only two packets; one answer carried an unsupported
  evidence reference and the other selected the wrong hypothesis.
- The receipt representation still invites a wrong inference:
  `EvidenceReceipt(..., records=2, ...)` visually suggests a `records`
  attribute even though the provider-visible object exposes `record_count` and
  requires `evidence_cache.load(handle)` for records. This is an interface
  incongruence worth repairing, but it does not explain every failure.

The run supports the resource-safety repair and rejects advancement of CodeAct
under this treatment. It does not establish that either interface is reliable
enough for the production Actor.

## Next gate

Retain Native as incumbent. Before another paid comparison, remove the receipt
representation incongruence and run zero-cost model-visible contract probes for
both arms. The separate remote-history stress result remains intact: generic
external acquisition plus bounded Python reduces the Native path from 32 turns
to 3 without making Python the authority language.
