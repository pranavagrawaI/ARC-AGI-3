# Crude 0006 potency prototype

> THROWAWAY. This is intentionally nonconformant with the proposed v0 Evidence Operator contract.

## Status

Issue #6 accepts this mechanism as a sufficiently strong model-facing path to its own empirical history. Closing that specification ticket establishes neither interface readiness nor gameplay benefit. Those tests remain deferred to #12 and #14. Failed provider attempts are operational diagnostics only and are excluded from version control.

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

## One-command preflight

From the repository root:

```bash
OPENROUTER_PROVIDER_ONLY=<pinned-provider> OPENROUTER_REASONING=on \
uv run python3 experiments/agents/0006-capability-environment/prototype/run_micro_pair.py \
  --model google/gemma-4-31b-it \
  --model-profile openrouter-generic
```

Preflight writes a frozen `pair-plan.json` but performs no gameplay.

## Explicit paid execution

Repeat the printed command with:

```text
--execute-gameplay --confirm execute-ls20-crude-0006-pair
```

Defaults freeze both members to the same game, environment seed, requested model seed, model request, 30-action ceiling, 60-call ceiling, context policy, and provider environment. Pair order is deterministically randomized from `--pair-order-seed`.

Cost must be re-estimated from the pinned provider before execution; model calls and provider pricing determine actual spend.

## Outputs

```text
pair-plan.json
baseline/{manifest.json,process.jsonl,trace.jsonl,...}
treatment/{manifest.json,process.jsonl,trace.jsonl,...}
analysis.json
human-review.json
```

`analysis.json` contains mechanical diagnostics only. `human-review.json` requires trace-cited judgments for:

- preserved earlier evidence;
- contradiction reconciliation;
- informative probe choice;
- avoidable actions.

## Decision rule

- **No signal:** no trace-cited improvement on any process dimension and equal-or-worse action behavior. Stop contract elaboration; delete the prototype or change the intervention premise.
- **Promising signal:** clear, trace-cited treatment improvement on at least two process dimensions without an obvious transport or execution confound. Build the cheapest replication or ablation; do not jump directly to the full contract.
- **Mixed:** choose the cheapest action that separates intervention weakness from interface burden.

After review, preserve only the verdict and cited evidence. Delete or replace this prototype.
