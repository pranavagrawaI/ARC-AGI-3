# DeepSeek Matched Actor V3 receipt pilot after-action

## Judgment

The receipt repair worked, but the paid pilot remained scientifically
incomplete. Only one of five matched pairs produced accepted answers from both
arms. CodeAct does not satisfy the precommitted advancement rule. Native remains
the production incumbent by burden of proof, not because this run established
Native superiority.

Do not repeat this paid pilot unchanged.

## Integrity

- authorization: `20260825T120013Z-deepseek-v4-flash-deepinfra`
- route: 51/51 responses reported `deepseek/deepseek-v4-flash-0731` through
  `DeepInfra`
- provider calls: 51 of 60 authorized
- actual provider spend: `$0.007161076` of `$3.00` authorized
- prompt tokens: 97,942; completion tokens: 8,633
- largest conservative input bound: 14,598 of 16,384 tokens
- largest total request bound: 18,694 of 20,480 tokens
- largest provider-reported prompt: 7,611 tokens
- ledger: five gates completed; `LadderCompleted` and `RunCompleted` recorded
- provider, route, budget, and integrity stops: none
- ARC actions: none were exposed by either treatment

The context/evidence plane kept every request and consequence inside the frozen
resource and authority boundaries. Context overflow was not the cause.

## Outcomes

| Packet | Native | CodeAct | Pair status |
| --- | --- | --- | --- |
| `action-success-grounded` | six-call exhaustion | six-call exhaustion | incomplete |
| `action-failure-illegal-bait` | accepted and correct on call 3 | accepted and correct on call 1 | complete |
| `history-success-level1-noop` | accepted and correct on call 5 | six-call exhaustion | incomplete |
| `objective-success-level-boundary` | six-call exhaustion | six-call exhaustion | incomplete |
| `cross-level-success-action-response` | six-call exhaustion | six-call exhaustion | incomplete |

Across all sessions, Native used 26 calls, 54,409 prompt tokens, 5,242
completion tokens, and `$0.003707032`. CodeAct used 25 calls, 43,533 prompt
tokens, 3,391 completion tokens, and `$0.003454044`. These are operating
observations, not comparative efficiency estimates, because four pairs were
incomplete.

## Causal observations

The V3 intervention repaired the defect it targeted. On two CodeAct packets the
provider saw `record_count` plus the exact `evidence_cache.load(handle)` path,
and the model successfully loaded the granted immutable value. It no longer
guessed a nonexistent receipt `.records` or `.to_dict` attribute.

The remaining failures are not one receipt defect:

- `action-success-grounded`: CodeAct acquired and loaded a metadata-only frames
  receipt, then spent its remaining calls inspecting records and making invalid
  query attempts instead of requesting the needed full projection or returning.
  Native repeatedly queried/viewed evidence and also never returned.
- `history-success-level1-noop`: CodeAct loaded exact frames and performed the
  relevant equality comparison on its sixth call, but emitted no terminal
  result. Native queried transitions and returned the correct grounded answer on
  call 5. This is evidence of CodeAct control overhead under the fixed envelope.
- `objective-success-level-boundary`: CodeAct lost calls to a two-tool rejection,
  positional calls, invalid `kind` values, and introspection. Native consumed all
  six calls acquiring and viewing evidence without returning.
- `cross-level-success-action-response`: CodeAct spent the full budget on
  namespace/API discovery and never issued a valid evidence query. Native began
  with an invalid two-tool turn and never acquired the cross-level evidence
  needed to answer.
- On the trivial illegal-bait control, CodeAct returned correctly in one call;
  Native acquired irrelevant evidence twice before returning on call 3.

The run therefore supports receipt-contract congruence, rejects CodeAct
advancement under this six-call DeepSeek treatment, and shows that both control
policies remain unreliable on the substantive packets. It does not support a
claim that CodeAct is inherently incapable or that Native is cognitively
superior.

## Next action

Retain Native plus bounded Python as the production hypothesis. Stop paid
sampling on this pilot. The next zero-cost work should use these exact traces to
separate task-policy waste from general interface cost, especially the Native
repeated-view behavior and CodeAct's API-discovery/terminal-control tax. Any new
paid comparison should require a redesigned production task protocol or a model
policy intervention with an offline golden path; it should not be another
receipt repair or an unchanged rerun.
