# M2 bounded-search candidate

An isolated successor of `../m2_b1_fixed`, with the original model, serving and context policy inherited. **Locally qualified, including actual sandbox-to-native-game execution; no live model uptake or competition efficacy claim.**

The Python sandbox exposes:

```python
bounded_search(initial_state, successors, state_key=..., is_goal=...,
               heuristic=None, max_depth=None, max_nodes=100000,
               max_expanded=50000, max_seconds=10.0)
```

The model supplies pure callbacks. `successors(state)` yields `(action, next_state)` pairs; `None` rejects an edge. `state_key` is a stable hashable key identifying equivalent future behavior. `is_goal` returns bool; an optional heuristic returns a finite nonnegative estimate. Unit-cost edges only. With no heuristic, traversal is breadth-first. An arbitrary heuristic supplies guidance without a shortest-route guarantee. The helper contains no game coordinates, object roles or transition rules.

The result includes `status`, `reason`, `path`, `state`, `depth`, expansion/edge/node counts, live depth-frontier count, and elapsed seconds:

| Status | Meaning |
|---|---|
| `FOUND` | Route found in the supplied model; verify predictions during real execution |
| `HORIZON_EXHAUSTED` | Depth cap left unexplored model states |
| `GRAPH_EXHAUSTED` | Supplied reachable model graph exhausted without a goal |
| `RESOURCE_LIMIT` | Time, stored-node or expansion cap reached; not a proof of impossibility |
| `CALLBACK_ERROR` | Model callback/key/heuristic failed; result identifies the callback and error |

Stored predecessor records, including reopens, count against `max_nodes`. Stable record indices preserve correct paths when states are reopened at shorter depths. Heap tie breakers do not compare model state objects. A reopened shallower state retires its old depth cutoff.

Time limits are cooperative between callback returns and yielded edges. The existing enclosing tool deadline kills a callback that does not return. During search, the sandbox blocks `action()` and nested `bounded_search`; normal actions remain available after search finishes or reports an error. When function retention is enabled by the inherited notebook configuration, pure model-written wrappers can call the helper on later turns. Saved plans use the existing JSON `save_data` API.

## Reproduce qualification

```bash
PYTHONDONTWRITEBYTECODE=1 /home/pranav/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12 solvers/m2_search/qualify.py
```

The candidate runs 47 behavioral tests, including 200 random small-graph comparisons with an independent BFS. It also checks real sandbox retention, saved plans, speculative-action prevention, guard recovery after errors, and hard cancellation of hung callbacks. The built installer is executed against a temporary donor tree, its search API exercised through the installed package, and the generated source patch actually applied with Git. Existing receipt, evidence, pin and parent-retention comparisons remain required. The complete bugfix parent hash inventory is verified unchanged.

Risk evidence in `evidence/search-risk.json` uses the previously explicit stdlib-trace/AST estimate under the fixed search tests. No dependency or quality threshold is introduced; this new utility has no preexisting function baseline. The full parent source delta is reviewed; existing search-independent behavior remains covered by inherited checks.

## Research outcome

The paired frozen screen in `analysis/bounded-search-20261014` uses six B1 models and their actual final states, all from one game, with the same 20-second/100,000-node/depth-60 caps. Guided search produces one native completion versus zero unguided completions: the reminder model's 29-action recovery in 8.48 seconds. All returned transitions match. The other five models reach the node cap in both arms.

A separate successful-run reference diagnostic finds and natively verifies 28–31-action recovery routes for all six starts. The five nonrecovering generated models disagree with those routes on growing-orange or shrinking-yellow consequences. That establishes concrete model defects along valid witnesses; it does not prove those defects alone caused every node-cap failure or that larger search could never find a different valid route. No reference rules or routes enter the helper, candidate prompt, or paired screen.

An analyst-authored program through the actual sandbox helper and action IPC executes the 29 native actions and clears L2. This proves the implemented capability path, not autonomous discovery or adoption by the LLM. Next empirical question: whether the frozen model uses this interface and gains equal-budget completion on a matched live comparison. Do not promote this candidate on retrospective recovery alone.
