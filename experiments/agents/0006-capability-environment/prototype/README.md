# Archived crude 0006 potency spike

> THROWAWAY. This is intentionally nonconformant with the proposed v0 Evidence Operator contract.

## Status

Issue #6 accepted this spike as evidence that the model can be given a sufficiently strong path to its own empirical history. It is no longer the implementation plan.

The current authority is the lean sequence in issues #9, #16, and #11: freeze minimum matched-control invariants, specify one recorded-trace or tiny-environment vertical loop, then build and inspect that loop. The #9 half of that sequence is built in `../p9/`; its frozen contract is `../p9/CONTRACT.md`. Do not run this directory's live-pair launcher as the next experiment. The source remains only as a design-search archive until the thin prototype supplies evidence for fine-grained deletion.

## Question

Does a crude package—exact last-frame delta, persistent model-authored hypotheses, prospective prediction, contradiction check, and explicit probe goal—produce an observable behavioral signal against unchanged 0002 on one paired `ls20` micro-run?

This pair can justify or kill further architecture work. It cannot establish a package effect, generalize across games, or identify which mechanism caused a difference.

## Intervention

The baseline is the existing 0002 runner, unchanged.

The treatment reuses that runner but substitutes `PrototypeCapabilityAgent`, which:

1. injects an exact summary of the latest frame change;
2. carries forward at most five model-authored hypotheses across context eviction;
3. requires a visible prediction, contradiction check, and probe goal before every action;
4. returns the previous prediction beside the observed transition on the next step.

There is no content-addressed evidence, formal ledger, canonical query surface, or polished decision protocol. Those are withheld until this package shows potency.

## Archived scope

The spike contains a live-pair launcher and mechanical review scaffold. They are retained to preserve the design search, not authorized for execution. Endpoint viability, live pairing, and RHAE evaluation are downstream of a working offline vertical loop.

## Decision rule

- **No signal:** no trace-cited improvement on any process dimension and equal-or-worse action behavior. Stop contract elaboration; delete the prototype or change the intervention premise.
- **Promising signal:** clear, trace-cited treatment improvement on at least two process dimensions without an obvious transport or execution confound. Build the cheapest replication or ablation; do not jump directly to the full contract.
- **Mixed:** choose the cheapest action that separates intervention weakness from interface burden.

After review, preserve only the verdict and cited evidence. Delete or replace this prototype.
