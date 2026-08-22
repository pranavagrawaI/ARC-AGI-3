# Execution substrate — earned changes only

Tycho-versus-NOOA is the wrong unit of choice. The current solver deliberately mixes
properties from both. Change one execution property only when a measured failure is
blocked on it, and preserve the authority boundary that property currently carries.

## What is established

| Property | Current state | Judgment |
| --- | --- | --- |
| Durable workspace and `wmlib` | Tycho filesystem, used through `run_python` | **KEEP** — the actor does inspect durable frames even though it does not call `read_file` |
| Canonical publication and planner gate | strict verification required | **KEEP** — candidate retention must not grant action authority |
| Builder Python process lifetime | fresh container/process per call | repeated helper redefinition is **SUPPORTED** |
| NOOA CodeAct session locals | available but bypassed | benefit for this Builder is **INCONCLUSIVE** |
| Raw `self` in CodeAct globals | full live agent object | **AUTHORITY RISK**, not a free port |
| Full-grid prompt inlining | substantial repeated payload | inefficiency **SUPPORTED**; token/cost benefit from state-by-reference is untested |
| Goal lifecycle obligation | immediate proposal → review → resolution gate implemented | **KEEP** — selected-checkpoint contract replay passes |
| Builder job scope | selected timeout was explicitly dynamics-scoped; 28/28 calls were mechanics | **DO NOT ADD ALLOCATOR** — job mixing was not the cause |
| Retained-candidate continuation | pinned matched pass ended retained 8/16 vs canonical 0/16; both gained zero | **SELECT RETAINED START, DO NOT EXTEND GENERIC PASS** |

The Builder authored 328 Python cells, repeatedly recreated helpers, and never wrote a
helper module. That supports a persistence experiment. It does not make the existing
loop “71% CodeAct”: CodeAct semantics include persistent locals, callable
model-defined functions, typed completion, and live-object execution.

## Order of work

### 1. Do not change substrate for candidate survival

The first repair is host-side artifact retention and contemporaneous scoring. It can
be implemented with the existing blob store, final workspace, verifier, and Builder
interface. A substrate change would confound the correctness repair.

### 2. Do not change substrate for goal-closure replay

The existing obligation already uses typed host-interrogable state and the action
decision boundary. Replay it before redesigning it. It does not require rendering
`self` or `state`, moving evidence out of the workspace, or adopting CodeAct.

### 3. Reduce prompt duplication before inventing state-by-reference

The actor already uses preloaded `grid` and `wmlib.frames()`. First compare matched
requests with full-grid inlining versus an initial grid plus summary/diff and explicit
on-demand retrieval. Measure uncached input tokens, completion, action, and evidence
use. Only a residual need justifies a live state handle.

### 4. Test Builder persistence behind the existing authority boundary

Only after explicit continuation selection is in place, replay one captured Builder request under:

```text
A: current fresh-process run_python
B: persistent Python session with the same narrow preloads
```

Hold prompt, model, task, evidence, tool budget, timeout, filesystem, and publication
policy fixed. Measure:

- repeated helper definitions and setup cells;
- completed Builder result within deadline;
- contemporaneous candidate score gained per model call;
- semantic trace reconstruction;
- containment and reachable object graph.

Do not bind raw `self`. Expose immutable or capability-limited views of `grid`,
`wmlib`, candidate storage, verifier, and report completion. Execute environment
actions only through the host.

### 5. CodeAct actor remains a later mechanism test

Run it only if goal-to-discriminator closure works yet the native actor still cannot
compose the operations needed for its next decision. The first arm changes action
representation and its necessary facade only; it does not simultaneously change
Builder policy, prompt history, grid delivery, budgets, or publication authority.

## Stop rules

- If candidate retention and goal closure fail before execution semantics matter,
  continue repairing those local contracts.
- If prompt elision preserves decisions and evidence use, do not build
  state-by-reference.
- If persistent Builder execution does not improve completion or candidate score per
  call, retain the fresh container and write durable helper modules only when earned.
- If a CodeAct arm cannot preserve isolation and semantic child traces, reject it even
  if its token count is lower.

The substrate is infrastructure. It earns change by removing a demonstrated blocker,
not by possessing more powerful abstractions.
