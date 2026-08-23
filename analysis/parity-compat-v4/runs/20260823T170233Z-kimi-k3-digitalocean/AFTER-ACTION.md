# Kimi K3 transport-congruence compatibility: after action

## Judgment

V4 fixed the transport defect it targeted. Across Gate 1, Kimi issued two valid structured `execute_python` calls and one valid structured `return_result`; it emitted no plain-prose Python. The V3-to-V4 transport-congruence hypothesis is therefore supported.

The valid ladder nevertheless stopped at Gate 1. Native queried once and returned the correct answer in two calls. CodeAct's first structured query omitted the contract's required `grid_encoding="raw"`, so T1 rejected it. The compatibility capability counts an attempted query before validating its exact shape; consequently the model's corrected second query was rejected for exceeding the one-query cap. Kimi then returned the correct typed answer from the instruction, but zero evidence queries had succeeded, so the arm failed the precommitted capability-use criterion.

Seven provider calls cost USD 0.04262175. Route, model, provider, authority, accounting, and trace integrity passed.

## Causal comparison

V3 Gate 1 CodeAct transport:

- one valid `execute_python` query;
- two plain-prose Python continuations;
- no terminal return.

V4 Gate 1 CodeAct transport:

- two valid `execute_python` calls;
- zero plain-prose continuations;
- one valid terminal return.

The changed historical representation produced the predicted change in action grammar. This is evidence about transport adherence, not yet evidence that CodeAct improves cognition.

## Remaining failure mechanism

The remaining failure is narrower than V3:

1. The packet exposed `grid_encoding="raw"` in the evidence contract.
2. Kimi's first CodeAct query omitted that argument.
3. `_model_evidence_query` supplied its ordinary default, `rle`.
4. The compatibility adapter rejected the resulting query because the frozen gate required exact equality with the raw query.
5. The adapter had already incremented its query counter, so the corrected query could not succeed.

The transport layer behaved correctly throughout. No representation payload was returned, so Gate 1 still does not test grid decoding.

## Consequence

Do not advance Kimi to the cognitive screen. Preserve V4 as a successful transport intervention and a failed capability-contract gate.

The cheapest next discriminating test, if authorized separately, is not another transport change. It is a compatibility-policy intervention that decides explicitly whether malformed/rejected capability requests consume the one successful-query allowance, while keeping the model-visible query contract unchanged. Alternatively, stop adapting the ladder and record Kimi as ineligible under the exact-query protocol. That choice is experimental intent, not a technical ambiguity.
