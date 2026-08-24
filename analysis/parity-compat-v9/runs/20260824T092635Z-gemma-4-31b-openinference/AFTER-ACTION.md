# Gemma 4 31B V9 flat-return compatibility: after action

## Judgment

The flat retrieval-only terminal contract solved Gemma's termination failure. Under V8, both arms exhausted three calls while approximating the wrapped discriminated union. Under V9, both Native and CodeAct returned the exact valid answer in one call with zero invalid interactions. This is strong matched evidence that schema depth—not basic tool calling—caused the V8 failure.

Gemma then cleared metadata capability in both arms. Native used two calls. CodeAct first omitted `limit`, repaired from exact feedback, disclosed L0:T3 LEFT, and returned the correct grounded answer on its third call.

Gemma did not clear RLE representation as a matched pair. Native passed. CodeAct queried the correct first transition and returned the correct cell conclusion `11->5`, but cited L0:T2 rather than the disclosed L0:T1. This is an evidence-grounding error after obtaining the relevant record, not a terminal-schema or RLE-value-decoding failure.

The raw alternative was censored during Native: its correct return response reported 17,686 prompt tokens, above the authorized 16,384-token cap. CodeAct raw was not run. Fourteen calls were observed at USD 0.003882850; thirteen were accounted at USD 0.002455720 and the rejected response accounts for USD 0.001427130. Every received call used the exact OpenInference BF16 route.

## Gate outcomes

```text
Gate 0 termination
    Native    one call, zero invalid returns             pass
    CodeAct   one call, zero invalid returns             pass

Gate 1 metadata
    Native    query -> return                            pass
    CodeAct   rejected query -> repaired query -> return pass

Gate 2 RLE
    Native    correct 11->5, cites L0:T1                 pass
    CodeAct   correct 11->5, cites L0:T2                 fail

Gate 2 raw
    Native    correct response, 17,686 input tokens      censored
    CodeAct   not run
```

## Consequence

Keep the flat gate-local return contract for smaller-model treatments. Do not restore the four-type union at the model boundary merely because T1 stores a richer canonical type; T1 can construct that type after validating the smaller proposal.

Do not rerun this authorization. The next discriminating work is no longer terminal-schema design. It is to separate evidence-reference grounding from representation decoding with a tiny value-only RLE control, or to make the returned record reference mechanically available beside the value. The raw arm must be removed or explicitly authorized above its known prompt size before it can produce an interpretable pair.
