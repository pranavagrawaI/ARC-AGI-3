# Evidence-conservation decision — 2026-08-13

## Failure boundary

The closed v24 run made three contradicted commitments about a static 3x3 region while every successor visibly changed a separate 5x5 block. The model did not have to say whether it retained, revised, or abandoned its prior hypothesis, and the host preserved the full frames without making the omitted residual mechanically salient at the claim boundary. A fresh artifact could therefore narrow the explanandum without accounting for the evidence it removed.

This is the general failure: improving local fit by shrinking scope, substituting a proxy, or silently discarding residual observations. It is not specific to LS20, movement, or the Bed of Procrustes metaphor.

## Delegation

The host now computes only coordinate-free coverage facts for each successor: total changed cells, counts inside and outside the prior primary mask, and the corresponding booleans. It does not name an object, select a better region, infer relevance, or assign causal meaning.

After a probe, the next model-authored claim must bind an update to that exact probe:

- disposition: `retain`, `revise`, or `abandon`;
- outside-change status: `explained`, `auxiliary`, or `unresolved` (`none` only when no outside change occurred);
- a short rationale.

The host validates presence, enum shape, residual completeness, and evidence binding. The model remains responsible for whether the residual matters, what persisted, what caused the change, and what hypothesis should replace the old one. It may retain the same region; no semantic answer is forced.

## Model principle

The prompt states one transferable rule: conserve the explanandum and evidence across revisions. A residual may be resolved, provisionally set aside, or remain open, but it may not disappear merely because a smaller hypothesis is easier to predict. Before narrowing scope, ask what observed fact is excluded and whether its exclusion changes the action model or decision.

## Qualification boundary

Offline regressions cover both prior shapes: a locally supported primary effect with simultaneous outside change, and a contradicted static proxy followed by change elsewhere. They show that the evidence and update obligation survive the artifact reset while semantic classifications remain model-authored. They do not establish that Qwen will identify the moving block, learn action directions, or complete a level. Only a fresh approved run can test that.
