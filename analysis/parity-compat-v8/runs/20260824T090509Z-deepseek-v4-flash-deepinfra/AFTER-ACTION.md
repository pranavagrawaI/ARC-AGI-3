# DeepSeek V4 Flash V8 compatibility: after action

## Judgment

DeepSeek cleared the termination and metadata-capability prerequisites in both Native and CodeAct. This establishes compatibility through Gate 1. It does **not** establish that V8's new container representation caused the improvement: the CodeAct trajectory never rendered `repr(evidence)`. It used `dir(evidence)` and the already-existing leaf representation, then queried and returned correctly.

The run continued to representation testing. Native passed RLE. CodeAct retrieved the exact RLE transition, wrote a correct decoder, and computed `11 5 64`, but its third and final call was computation rather than `return_result`; the strict RLE gate therefore failed. This is a terminal-budget failure after successful representation use, not evidence that DeepSeek could not decode RLE.

The alternative raw gate was censored during Native. Its return response was correct, but embedding the raw grid in history produced 18,099 reported prompt tokens, above the authorized 16,384 per-call cap. Reconciliation stopped the run before CodeAct raw. Fourteen provider calls were observed at USD 0.003216436; thirteen were accounted by the ledger at USD 0.001829736, and the rejected response accounts for the remaining USD 0.001386700. All observed calls used the exact DeepInfra route.

## Gate outcomes

```text
Gate 0 termination
    Native    pass
    CodeAct   pass

Gate 1 metadata
    Native    query -> return                         pass
    CodeAct   inspect -> query -> return               pass

Gate 2 RLE
    Native    query -> return                         pass
    CodeAct   inspect -> query -> decode 11->5         fail: no call left to return

Gate 2 raw
    Native    query -> correct return response         censored: 18,099 > 16,384 input cap
    CodeAct   not run
```

## What changed our model

V8 reached the first representation test, so further discoverability work is no longer justified. However, uptake of the V8 container representation was absent; the Gate-1 pass may be sampling variation or more economical use of the V7 leaf surface.

RLE itself was usable. The model constructed the correct decoder and extracted the exact required values. The live problem is the relationship between a three-call ceiling and CodeAct's inspect/query/compute/return lifecycle. Raw grids create a separate, mechanical context-size problem.

## Consequence

Do not retry this authorization and do not interpret the incomplete raw pair. Freeze discoverability. Before another paid run, make the next package answer one narrow design question: whether the representation gate should supply a manifest-derived pure decoder or permit four CodeAct calls so retrieval, local computation, and typed termination can all occur without conflating protocol economy with representation competence. Independently, either raise the declared input ceiling for raw grids with matching cost reservation or remove raw from the compatibility ladder; the current raw path can mechanically exceed 16,384 tokens.
