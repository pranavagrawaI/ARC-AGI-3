# Cost-bounded shadow evaluation and builder-refresh inventory

## Outcome

**Complete.** Shadow execution now requires a cumulative observed-cost limit
and per-call reserve, preserves provider-route identity when the response
exposes it, and treats incomplete arms or route drift as inconclusive. Existing
Gemma 4 checkpoints were then mined without a model or game call.

## Delivered

- `solvers/tycho_nooa/src/tycho_nooa/shadow_eval.py`: cost reservation,
  post-response cost stops, route capture, route-stability assessment, and paid
  CLI refusal without monetary controls.
- `solvers/tycho_nooa/tests/test_shadow_eval.py`: cost, missing-limit, route
  preservation, and route-drift regressions.
- `tools/failure_atlas.py` and `tools/failure_atlas_core.py`: zero-call
  `builder-refreshes` inventory command.
- `tools/tests/test_failure_atlas.py`: report comparison, ranking, spend, action
  relation, and transport-wrapper parsing regression.
- `analysis/builder-refreshes/gemma4-existing-20260821.json`: inspectable result.

## Problems

The earlier live shadow spent $0.03613761 against a stated $0.03 cap because
the cap was only documentary. Client-side enforcement won over another
advisory field: a call starts only if observed spend plus its reserve fits. An
already-started provider response can still exceed its reserve, so the result
records `observed_cost_exceeded`, withholds all later calls, and becomes
inconclusive rather than claiming an absolute billing guarantee.

The earlier response evidence lacked provider-route identity. The evaluator can
now retain non-secret route fields exposed by the response and invalidates a
comparison if identities drift. If the provider exposes none, route identity
remains explicitly unverifiable; the client cannot reconstruct hidden routing.

One historical builder note contained a Python triple-quote wrapper tail. A
red regression reproduced the contaminated field, and the parser now removes
that transport suffix without changing report semantics.

## Decisions

No new game or paid model run was made. Offline ranking precedes another
counterfactual because it can eliminate duplicate or low-information cases at
zero marginal model cost. The miner keeps substantive field changes visible;
it does not label a changed report redundant merely because its recommendation
is absent or unchanged.

## Result

The miner inspected 102 checkpoint calls and found four builder batches costing
$0.16460826 total. It nominated two later reports:

1. Feasibility sequence 3 / actor call 60 is unshadowed. Seventeen builder calls
   cost $0.06014475 and produced a substantively changed collision report with
   no action recommendation; the actor next called `run_python`.
2. Reasoning-capture sequence 2 / actor call 36 cost $0.03765601 and retained
   `UP`. Its existing matched shadow is joined as `contradicted`, with a 0.0
   historical-control first-tool match rate.

There were no exact report repeats. Candidate 1 is therefore the next
economically interesting shadow, not evidence that the builder refresh was
useless or safe to remove.

## Verification

- The shadow tests exercised 19 cases, including red-then-green cost and route
  regressions.
- The failure-atlas tests exercised eight cases, including a red-then-green
  transport-wrapper regression.
- The inventory records `model_calls_made: 0` and `issues: []`.

These checks establish local enforcement, evidence capture, and deterministic
offline mining. They do not establish provider-side billing containment,
causal redundancy of actor call 60's report, gameplay benefit, or transfer.

## Residual

The provider may omit route fields, and any single response may exceed the
reserved amount. Only provider-enforced billing limits could close the latter
gap. A paid matched shadow at actor call 60 would settle immediate response
dependence at that checkpoint, subject to control validity and exposed-route
stability.

## Resume

Inspect `analysis/builder-refreshes/gemma4-existing-20260821.json`, then prepare
the exact actor-call-60 report-removal packet. Preflight it without `--execute`
using explicit call, cost, and reserve limits before deciding whether the paid
four-call shadow is worth buying.
