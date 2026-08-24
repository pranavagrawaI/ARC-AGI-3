# V13 after action: Gemma CodeAct runway

## Judgment

The three-call ceiling was not the sole cause of V12's CodeAct failure. Under six calls, Gemma still did not decode `11->5` or emit `return_result`. The valid result rejects deadline truncation as a sufficient explanation.

This does **not** establish that programmable reasoning is intrinsically worse. It establishes that the current sterile CodeAct interface is operationally inferior to the Native control on this exact task: Native passed on call 2; CodeAct exhausted call 6.

## Observed causal sequence

| Call | Observed action | Consequence |
|---:|---|---|
| 1 | Imported unavailable NumPy, invented an incorrect RLE decoder, queried with no arguments | Cell failed before capability request |
| 2 | Removed NumPy, again queried with no arguments | Exact semantic query was returned in rejection feedback |
| 3 | Made the exact query | Evidence `L0:T1` disclosed; then treated the returned page as a list |
| 4 | Used `records[0]` | Treated the 64 row strings as one RLE string |
| 5 | Indexed row 63 rather than row 0, then requested `evidence_ref` | Printed row-63 encodings and failed because the record field is `ref` |
| 6 | Inspected record keys and row-63 strings | Found `ref=L0:T1`; no decoding and no terminal return |

## What the run settles

- **Supported:** three calls were tight, but extending to six did not make the current pairing compatible.
- **Supported:** transport and evidence authority worked; one correct disclosure occurred and all six actions were structured Python calls.
- **Supported:** most consumed runway was interface repair and representation archaeology, not useful task computation.
- **Contradicted:** a simple call-4 deadline-truncation account.
- **Inconclusive:** whether a redesigned programmable interface can outperform Native on hard ARC reasoning.

## Metric correction

The generated report labels call 6 `computation_completed_call` because V13 precommitted that field as the first completed post-disclosure cell. The trace shows that call 6 was inspection, not completed task computation. `correct_answer_observed_call=null` is the reliable semantic milestone. Future reports should rename the former to `post_disclosure_cell_completed_call` rather than implying semantic completion.

## Engineering implications

The failure surface is cumulative:

1. the available pure-Python environment is not obvious enough to prevent unavailable imports;
2. exact capability arguments are discoverable mainly through rejection repair;
3. the disclosure is a nested dictionary whose shape must be rediscovered;
4. RLE is represented as a list of encoded row strings, while the model repeatedly assumed a flat encoded grid or decoded values;
5. canonical input uses `ref`, while the terminal schema asks for `evidence_ref`.

Each choice is defensible alone. Their composition consumes the smaller model's attention.

## Next action

Stop paid compatibility sampling. First perform a zero-cost interface audit and scripted golden-path proof:

- prove the current exact DTO can be queried, decoded at `prev[0]`/`next[0]`, grounded through `record["ref"]`, and returned in two model turns;
- rename the misleading milestone without rewriting this preserved result;
- then design one matched ergonomic intervention, preferably a pure audited row decoder and a self-describing result shape exposed identically to Native and CodeAct.

Only rerun a model after the new intervention has a single explicit causal claim. Native remains the Actor incumbent.
