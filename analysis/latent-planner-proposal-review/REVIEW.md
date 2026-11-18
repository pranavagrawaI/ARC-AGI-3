# Review of Claude's latent-planner proposal

Verdict: useful as a bounded computation tool after correcting its contract;
not a replacement for the agreed epistemic changes. Reviewed the supplied
proposal against repaired runtime submission-runtime-20260910-v5. No solver
source changed and no model or game experiment ran.

## Blocking findings

1. **False exhaustion (proposal section 1).** The supplied loop drops nodes at
   max_depth and still returns no_plan_exhausted. It also skips predictor
   exceptions and can return the same verdict after every transition failed.
   The promised model_error status is never returned. Reproduction: a goal at
   depth 2 with max_depth=1 is labelled exhausted; a predictor that always
   raises is also labelled exhausted. Exhaustion requires a complete traversal
   with neither unresolved transitions nor depth/resource cutoffs. Distinguish
   incomplete search, callback failure, and closed-model exhaustion.

2. **Timeout contract is false (sections 1 and 3).** The deadline is checked
   between node expansions, not between all callbacks or before returning a
   found goal. goal(z0), goal(nz), model loading, and encode are unbounded in
   the proposed wrapper. The specified bare fallback for predict can hang.
   Even a caught per-call timeout becomes a skipped edge, potentially causing
   false exhaustion. Reproduction: goal sleeps 30ms and returns true; with a
   1ms allowance the function returns already_satisfied. The existing 60s
   cell supervisor kills the Actor, not a 5s planner returning timeout. A
   bounded planner must bound all callbacks and refuse an unavailable bound.

3. **Search success does not earn batching (section 3).** Current
   solver_agent.submit_actions requires a successfully checked frozen helper
   identity for batches, a valid predicted_path, and no failed retrodiction.
   A search alone supplies none of the external observations needed to earn
   that permission. Nor can an arbitrary latent difference derive the host
   expect classes move/no_impact/level_up/terminal. An unchanged latent may
   hide changed pixels, and reaching an arbitrary goal need not advance a
   level. The omitted _out leaves this essential conversion unspecified.
   Return modeled actions/latents separately from host outcome predictions.
   Integration tests must prove rejection before qualification and acceptance
   after real matching transitions, while preserving existing gates.

4. **State isolation and normalization are unspecified (sections 1–3).**
   The visible loop reuses z across sibling actions. With a direct _step,
   in-place updates can produce a one-action plan that only works after a
   different sibling action has mutated the start state. The reproduction
   reaches x=11 with reported action +10 from x=0. _step is omitted, so this
   finding is conditional on its implementation: it must copy inputs and
   detach outputs using the same JSON normalization as prepare(). Goal
   callbacks also need isolation. key(nz) is outside the exception handler;
   a non-JSON result escapes the claimed status-only contract. Reuse plain()
   to preserve the existing numpy, finite-value and latent-size rules.

5. **Expansion cap is not a memory/work cap (section 1).** max_nodes bounds
   dequeued nodes, but one expansion generates all successors before another
   budget check. Reproduction: max_nodes=1 admits 1001 seen states with 1000
   actions. Bound generated states/frontier or branching, check time inside
   the action loop, and validate caller limits. Returning a depth-limited
   result must not claim the whole graph was exhausted.

## Keep and revise

Keep a shared owner for selecting/loading the model, provided top-level helper
execution remains bounded. Return the exact helper SHA and starting state
identity with the result; sharing a selector alone does not bind a later
submission to the searched snapshot. Keep a discoverable method and bounded
BFS if its contract is corrected. Do not promise global shortestness when
edges were skipped after errors. Discovery text must distinguish available
model code, local search success, and externally earned batch eligibility.

Allow goal to mean an explicit local predicate, with no claim that it is the
winning objective. Keep exploratory single actions available without a goal
or an executable model. A planner can execute a conditional plan but cannot
validate objective choice or preserve uncertainty by itself.

## Evaluation

First qualify deterministic mechanics: error versus exhaustion, depth cutoff,
mutation isolation, total and callback timeout, generated-state limits, and
real submission gates before/after a frozen-model match. The proposed final
integration test needs those eligibility preconditions, not unconditional
acceptance of a found plan.

Only then consider a separately authorized experiment. Cell seconds and
watchdog kills measure computation safety. Adoption, model calls per executed
action, total decision time, and interpretation of incomplete-search statuses
are needed to assess usefulness. A faster search through a false objective
model is not evidence of better epistemics. Existing endurance time was
97.75% Actor-request time; CPU savings alone are not the principal outcome.

The proposal's in-flight contingent comparison reference is stale for this
thread. Build any future candidate from repaired v5 and preserve recorded
runs. Packaging must include and hash the new module; unchanged shell launch
arguments alone do not establish reachability.

## Reproduction and scope

Run `python3 analysis/latent-planner-proposal-review/reproduce.py`.
RESULTS.json records six bounded counterexamples. The supplied plan loop is
preserved in that script. Since _step and _out were omitted from the proposal,
the script uses explicitly documented simple adapters; it does not claim to
have tested a complete Claude implementation or its timeout backend.
