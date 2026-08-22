# Tycho-NOOA Architecture: What Remains to Steal

**Repository snapshot:** 2026-08-21  
**Status:** Architectural findings and questions, not an implementation claim  
**Decision under examination:** Whether the hybrid actually preserves the strongest
properties of both Tycho and NOOA, especially in the actor loop

## Judgment

The present solver is a real hybrid, but an asymmetric one.

It reproduces Tycho's cognitive environment deeply: a durable workspace, native
workspace tools, executable world model, verifier, planner, trigger-driven builder,
scribe, ownership modes, and one-action turn boundary. It uses NOOA successfully as
the runtime chassis: persistent `Agent`, `GenerationStrategy`, `RuntimeServices`,
event history, typed events, model client, and strategy-decorated methods.

But it does **not** use NOOA's central agency mechanism for either the actor or the
builder. Both agents set `context={"self": None, "state": None}` and operate through
custom provider-native tool loops. The model does not act upon a live typed agent
object through CodeAct. It cannot compose agent methods in Python, preserve live
intermediate objects across cells, query its experience through NOOA's object APIs,
or finish arbitrary generation methods through typed `return_result`.

The most accurate description is therefore:

> Tycho's working method, hosted and instrumented by NOOA, with only a partial use
> of NOOA's object-oriented agency.

This is not necessarily the wrong baseline. The native loop preserves Tycho parity
and gives a strong action-control surface. It is, however, too early to conclude
that the native loop should permanently remain the actor implementation. A bounded
CodeAct actor can preserve the same host-authoritative action boundary while adding
NOOA's compositional power. That possibility should be implemented as a matched
treatment rather than dismissed or adopted on architectural taste.

---

## 1. What the current actor actually uses

`TychoNooaAgent._choose_action` is decorated with the custom
`NativeToolStrategy`, not NOOA's `CodeActStrategy`.

The live path is:

```text
ARC host sends GameTurn
    -> persistent TychoNooaAgent records the observation in GameWorkspace
    -> optional Tycho trigger invokes a fresh FocusedBuilder
    -> _choose_action(turn)
    -> NativeToolStrategy publishes a NOOA Task event
    -> NOOA RuntimeServices.generate exposes Tycho tools to the provider
    -> model emits JSON tool calls
    -> WorkspaceToolSurface validates and dispatches each call
    -> loop repeats until take_action returns ActionDecision
    -> host validates and executes that decision
    -> authoritative ActionResult arrives on the next worker message
```

The actor's tools are:

- `ls`
- `read_file`
- `write_file`
- `edit_file`
- `edit_function` where the mode permits it
- `run_python`
- `set_verbosity`
- `take_action`
- treatment- or mode-specific additions such as builder and typed-goal tools

The main loop calls `runtime.generate(tools=...)` repeatedly. Prose without a tool
call does not finish a turn. `take_action` is the unique normal completion operation.
After `max_tool_steps`, the implementation spends one final call exposing only the
action tool; failure to commit then raises `GenerationError`.

This is unquestionably an agent loop. It is simply not CodeAct.

### `run_python` is not NOOA `execute_python`

The similar names conceal different execution models.

Tycho's `run_python` is one workspace operation delegated to Tycho's
`ToolExecutor`. It executes game-analysis code in Tycho's prepared environment,
with resources such as the current grid, frames, `wmlib`, model files, and planner
facilities. It returns a textual tool result to the native-tool loop.

NOOA's `execute_python` is the action primitive of `CodeActStrategy`. CodeAct
constructs a Python session for a typed generation method, seeds its locals, exposes
permitted live objects and methods, records cell results, carries session locals
between cells, and repeats until the model calls `return_result(...)`. That final
value is validated against the generation method's annotated return type.

Thus both systems let the model run Python, but only CodeAct makes Python the
language in which the agent composes its own capabilities.

---

## 2. What CodeAct means

CodeAct is both a loop and an action representation:

```text
generation-method call
    -> prompt contains task, typed state, visible methods, and execution context
    -> model calls execute_python(code)
    -> code runs with persistent session locals
    -> result becomes an event and the model is called again
    -> model can inspect, branch, iterate, and call more capabilities
    -> model calls return_result(value)
    -> NOOA validates value against the method return type
```

The distinction from the current native loop is not "loop versus no loop." It is
what the model is permitted to iterate over and the language in which it acts.

| Dimension | Current hybrid actor | NOOA CodeAct |
|---|---|---|
| Strategy | Custom `NativeToolStrategy` | Built-in `CodeActStrategy` |
| Model output | Provider JSON tool calls | Python through `execute_python` |
| Principal interface | Several Tycho tool schemas | Live typed objects and methods |
| Intermediate state | Event text and workspace files | Python locals, `Out[n]`, objects, events |
| Completion | `take_action(...)` | Typed `return_result(...)` |
| Return type | Specifically `ActionDecision` | Any annotated method return type |
| Authority boundary | Encoded by distinct tool | Must be preserved by facade and host |
| Trace grain | One named operation per tool call | One cell may compose many operations |
| Inference economy | Usually one inference per tool batch | Many deterministic calls can occur in one cell |

CodeAct does not imply that generated code must execute the environment action.
It can return an `ActionDecision`, leaving execution solely to the host. Conflating
"CodeAct chooses" with "CodeAct directly acts on the game" creates a false choice.

---

## 3. What Tycho contributes, and what is already well used

### 3.1 Durable epistemic workspace

Tycho gives the agent a persistent external medium rather than asking the model to
carry everything in conversational text. Observations, notes, executable model,
verification code, planner artifacts, reports, and terminal evidence can survive
turns and be inspected independently.

This is one of the hybrid's strongest inherited properties and is substantially
present. It supports causal forensics because host-authored evidence can remain
separate from model-authored interpretation.

### 3.2 Executable theory, not only verbal belief

`world_model.py`, `verify.py`, and `plan.py` make transition hypotheses executable.
The system can distinguish:

- a theory that fits a recorded trajectory;
- a theory that predicts legal counterfactual branches;
- a satisfiable terminal predicate;
- a planner whose operators and keys agree with the model;
- an action selected by the actor.

These are different claims. The architecture makes it possible to test them
separately, although current behavior still often fails between candidate objective
formation and selection of a discriminating experiment.

### 3.3 Actor-builder separation and explicit ownership modes

Tycho's `trigger`, `orchestrator`, `single`, and `no_world_model` modes represent
meaningfully different allocations of authority. The hybrid mechanically enforces
who may edit the executable world model and keeps builders unable to act on the
environment.

That separation provides an independent construction/repair perspective and makes
world-model ownership experimentally manipulable. It can also become harmful:
fresh builders can repeat work, builder reports can displace the actor's ownership
of purpose, and automatic construction can consume the decision deadline. These are
reasons to measure when the separation earns its cost, not reasons to erase it.

### 3.4 Event-triggered slow cognition

Tycho does not require expensive model construction on every actor step. A
transition mismatch or genuine level boundary can invoke focused builder work.
The hybrid preserves this trigger path and adds useful deadline and evidence
instrumentation.

The underexploited possibility is to treat triggers as requests for named cognitive
capabilities, not necessarily as requests for a fresh general builder every time.

### 3.5 A narrow, legible action interface

Tycho's native loop cleanly separates investigation from the final action
commitment. Provider tool schemas validate arguments and leave an operation-level
trace. For ARC, where an environment action is scarce and cannot be silently
retracted, this is valuable.

### 3.6 Sandbox and environmental separation

Speculative game-model code runs in the Tycho workspace environment rather than
through unrestricted mutation of the live agent runtime. This limits the blast
radius of broken analysis and makes the game model an inspectable artifact.

### 3.7 What Tycho still offers that is not fully realized

- Reflection/meta-reflection is deliberately disabled for the focused builder.
- The full reflection cadence and context-compaction policy are not yet reproduced.
- Builder value is not yet routed by phase: construction, divergence repair,
  objective discovery, planning, and terminal repair currently have very different
  expected value but share too much machinery.
- Workspace artifacts are durable, but their contents are not yet consistently
  surfaced as typed, queryable objects to the actor.
- A correct executable model can exist without causal uptake by the actor.
- Goal hypotheses can remain prose without creating an obligation to run a
  discriminating probe.

---

## 4. What NOOA contributes, and what is underused

### 4.1 The agent as a typed live object

NOOA's central proposition is not merely that an `Agent` owns an event history.
State lives in typed fields, capabilities live in methods, docstrings specify their
meaning, and generation-method signatures establish contracts. CodeAct lets the
model operate through that object.

The hybrid currently disables the `self` and `state` context blocks for both the
persistent actor and fresh builder. Most fields are hidden. Consequently, the model
does not encounter the agent as the semantic owner of its knowledge and
capabilities. It encounters a Tycho tool menu and filesystem.

This is the largest unstolen NOOA strength.

### 4.2 Capability composition through CodeAct

CodeAct can execute control flow over typed capabilities:

```python
history = self.workbench.trajectory(level=self.level)
candidates = self.goals.active()
for candidate in candidates:
    prediction = self.model.predict_probe(candidate)
    candidate.attach(prediction)
probe = self.experiments.best_discriminator(candidates)
return_result(self.actions.from_probe(probe))
```

The important feature is not the syntax. It is that intermediate values remain
objects, loops happen deterministically inside one cell, and one capability's
result can become another's input without being serialized into prose and
reconstructed by another inference.

The current native loop pays an inference and serialization boundary for much of
this composition.

### 4.3 Typed completion and automatic validation

CodeAct's `return_result` validates the final value against the generation method's
return annotation. The same mechanism can support more than actions:

- `EpistemicUpdate`
- `ProbeProposal`
- `BuilderRequest`
- `WorldModelDiagnosis`
- `ActionDecision`
- `LevelSummary`

The current hybrid uses typed Python models internally but presents most cognition
through native tools and prose reports. The `typed_candidate_goals` treatment is an
important step toward NOOA's paradigm, but it remains a native-tool adapter rather
than live object-oriented agency.

### 4.4 Dynamic context and queryable event history

NOOA supports dynamic context blocks and `EventQuery`. The current hybrid mainly
retains a growing event history, evicts old grids, and constructs custom observation
text. It underuses the ability to select context by semantic relevance and to let
the agent query prior events as structured experience.

A durable history is not automatically usable memory. If the actor must reread or
reconstruct long rendered transcripts, persistence can become burden rather than
capability.

### 4.5 Storage and snapshots

NOOA can place its event backend and snapshot state behind a `StorageManager` rather
than the default in-memory backend. The hybrid has excellent custom decision
checkpoints and a durable Tycho workspace, but does not yet use NOOA persistence as
the unified persistence mechanism for agent state, event queries, or resumption.

The custom evidence checkpointing should not simply be replaced: it preserves
credential-free exact requests and content-addressed workspace bytes for scientific
comparison. The opportunity is to connect NOOA snapshots to this evidence system,
not to collapse authoritative evidence into opaque framework storage.

### 4.6 Summarization and context management

NOOA contains summarization, truncation, context-block, and event-archive machinery.
The hybrid has old-grid eviction and a Tycho-style level scribe, but the README still
identifies emergency compaction and the full Tycho reflection/context cadence as
unfinished. A long-lived actor should use deliberate, inspectable context policy
rather than rely on one growing history until the window becomes consequential.

### 4.7 Skills and capability libraries

NOOA can attach and document skills as capabilities visible from the agent object.
Tycho's verified world-model operations, evidence queries, symmetry operations,
planner diagnostics, and replay facilities could become narrow skills rather than
an ever-growing flat provider tool surface or a collection of files the model must
remember to open.

This also creates a path for capabilities to be earned: a candidate operation can
remain an artifact until repeated evidence justifies promoting it into a callable
skill.

### 4.8 Strategy composition

NOOA supplies multiple generation strategies and strategy-local contracts. The
hybrid uses the abstraction, but its main actor and builder both recreate custom
native loops. It has not yet exploited the possibility that different cognitive
jobs deserve different strategies:

- CodeAct for multi-step typed investigation;
- native tools for a parity control or constrained repair surface;
- deterministic Python for evidence validation;
- a smaller prediction strategy for bounded classification;
- scribe/summarization only at meaningful lifecycle boundaries.

---

## 5. Why the native Tycho loop is presently attractive for the actor

The native loop has real strengths. They should be preserved as requirements, not
mistaken for proof that its exact implementation is optimal.

### Explicit action authority

`take_action` is visibly different from investigative tools. The model proposes,
the tool surface validates, the strategy recognizes commitment, and the host alone
executes. Analysis cannot accidentally masquerade as an environment result.

### Small visible action vocabulary

Concrete provider schemas make legal operations obvious and validate their
arguments. The model need not inspect an object or remember which Python methods
have side effects.

### Tycho parity

Keeping the original tool names, descriptions, workspace, and prompt relationship
preserves comparability with pinned Tycho. This matters while establishing whether
the port reproduced Tycho's behavior.

### Operation-level traces

Each tool call has a stable name, argument object, result, source, and call ID. It is
easy to reconstruct what was read, changed, tested, or committed. A CodeAct cell can
hide several semantic operations unless the called capabilities emit equally
granular events.

### Containment of speculative game code

`run_python` targets Tycho's game-analysis environment, not the live NOOA runtime.
This protects the event manager, lifecycle, and agent state from ordinary analysis
mistakes.

### A forced action deadline

The loop explicitly guarantees bounded investigation followed by an action demand.
ARC does not generally reward a turn that merely reports incomplete reasoning.

These are good reasons to keep `NativeToolStrategy` as the fidelity baseline. They
are not yet evidence that it produces better actions than a carefully bounded
CodeAct actor.

---

## 6. Why CodeAct may be the better actor loop

The actor's job is not merely to select a legal action. It must relate present
evidence, accumulated hypotheses, executable theory, available experiments, action
cost, and prospective consequence. This is exactly where programmatic composition
can matter.

CodeAct may offer:

1. **Computational density.** Several deterministic evidence queries and comparisons
   can occur in one Python cell instead of requiring several model round-trips.
2. **Typed intermediate state.** Candidate goals, predictions, contradictions, and
   probes need not be repeatedly converted between objects, JSON, files, and prose.
3. **Explicit control flow.** The actor can branch, iterate, rank, and test with
   ordinary Python rather than hoping a serial tool dialogue reconstructs the same
   procedure.
4. **Capability composition.** World-model queries, event search, builder advice,
   planner checks, and risk estimates can be combined as methods.
5. **Typed final action.** `return_result(ActionDecision(...))` can end the loop while
   leaving actual game execution to the host.
6. **Greater use of persistent identity.** The persistent NOOA actor can become the
   owner of typed epistemic state rather than merely the owner of an event manager.

No current evidence establishes that the native actor is superior on these
dimensions. Conversely, no evidence yet establishes that CodeAct will improve
gameplay. The architectural question remains open and experimentally tractable.

---

## 7. The strongest candidate architecture

### Preserve the authority boundary, not necessarily the native loop

Introduce a narrow model-visible facade rather than exposing all internals:

```python
class ActorWorkbench:
    evidence: EvidenceView
    goals: GoalRegistry
    model: WorldModelWorkbench
    experiments: ProbeWorkbench
    planner: PlannerWorkbench
    builders: BuilderWorkbench

    def legal_actions(self) -> list[LegalAction]: ...
    def propose_action(self, ...) -> ActionDecision: ...
```

Then make the actor method a bounded CodeAct generation method:

```python
@strategy(CodeActStrategy(config=...))
async def choose_action(self, turn: GameTurn) -> ActionDecision:
    """Investigate the authoritative evidence and return one legal proposal."""
    ...
```

The CodeAct environment may inspect and compose through `ActorWorkbench`, but it
must not directly:

- send an action to the ARC environment;
- mutate the worker lifecycle;
- forge host-authored observations or terminal evidence;
- rewrite event history;
- gain unrestricted filesystem or network authority;
- silently alter the canonical world model in modes where the actor does not own it.

The host remains the only action executor:

```text
CodeAct returns typed ActionDecision
    -> strategy validates the return type
    -> host validates legality and coordinates
    -> host executes exactly once
    -> environment result becomes authoritative evidence
```

### Keep Tycho's workspace as the scientific substrate

CodeAct should not replace Tycho's executable world model or workspace with
ephemeral Python beliefs. The workbench should provide typed access to the durable,
versioned artifacts and should cause meaningful operations to emit inspectable
events.

### Keep a native parity arm

The existing `NativeToolStrategy` remains valuable as:

- the exact Tycho-interaction baseline;
- a safe fallback;
- a discriminator for whether CodeAct composition changes behavior;
- a simpler surface for models that perform poorly with generated code.

### Alternative: separate epistemic and action loops

A more conservative intermediate design is:

```text
bounded CodeAct investigate(...) -> EpistemicUpdate
    -> native Tycho actor reads update
    -> native loop commits take_action
```

This lets CodeAct own structured scientific reasoning while native tools retain the
final action conversation. It is easier to isolate, but it pays for two model loops
and can create a new handoff where correct analysis is ignored. It should therefore
be treated as an intermediate treatment, not assumed to be the final hybrid.

The more ambitious and potentially cleaner design is one bounded CodeAct actor that
returns an action through a narrow facade.

---

## 8. What the hybrid enables that neither parent provides alone

### Executable scientific agency with grounded consequences

NOOA supplies a live typed computational agent. Tycho supplies an executable theory
of an external environment, immutable observation evidence, and a guarded action
boundary. Together, the agent can manipulate hypotheses as objects while testing
them against a durable external model and real consequences.

### Two kinds of memory with explicit provenance

- NOOA state/events can preserve semantic, queryable experience.
- Tycho workspace/versioning can preserve executable artifacts and raw evidence.

The combination can distinguish what the agent remembers from what the host
observed and what executable theory currently predicts.

### Earned capabilities

A hypothesis may progress through:

```text
prose conjecture
    -> typed CandidateGoal or transition claim
    -> discriminating probe
    -> verified workspace artifact
    -> repeated cross-state success
    -> promoted callable NOOA skill
```

Tycho alone has strong artifacts but a weaker general capability-composition model.
NOOA alone can compose methods but does not supply Tycho's game-specific process for
earning them against authoritative evidence.

### Strategy choice by cognitive job

The hybrid need not have one universal agent loop. It can choose among native tool
calling, CodeAct, deterministic verification, prediction, and summarization based on
the operation's authority, required control flow, and cost.

### Controlled self-modification

Tycho can revise executable world-model code under explicit ownership modes. NOOA
can expose the verified result as a typed capability. This allows model improvement
without granting unbounded self-mutation.

### Causal evaluation at the actual decision point

NOOA events and model-call boundaries combined with Tycho workspace commitments
allow matched interventions that ask whether an artifact actually changed the
actor's choice. This separates artifact correctness from causal uptake and gameplay
success.

---

## 9. Obvious improvements, ordered by leverage and dependency

### P0. Preserve correctness and authority before changing the loop

Maintain these invariants in every treatment:

- host observations and terminal results are authoritative;
- model notes and hypotheses cannot masquerade as observations;
- one returned decision causes at most one environment action;
- action execution remains outside generated code;
- mode-specific world-model ownership remains mechanically enforced;
- exact decision evidence remains restorable and comparable;
- planner search follows a satisfiable goal, usable dynamics, congruent forward
  operators, and sound state keys rather than compensating for their absence.

### P1. Add a typed `ActorWorkbench`

Make the smallest useful model-visible surface. Begin read-mostly. Every method
should have:

- a precise return type;
- explicit authority classification;
- structured evidence references;
- deterministic validation where possible;
- an event emitted at the semantic operation boundary.

Do not expose the entire current agent merely because CodeAct can inspect `self`.

### P2. Implement CodeAct as a matched actor treatment

Replace only `_choose_action` in the treatment. Hold the workspace, prompts'
substantive information, model, action budget, builder policy, and host protocol
constant. Return the existing `ActionDecision` model.

### P3. Promote the epistemic ledger from tools to live objects

`CandidateGoal`, evidence references, falsifiers, status, and revision history are
already a promising local treatment. Under CodeAct they should be queryable objects,
not merely additional provider tools. Add a typed `ProbeProposal` so a provisional
goal creates an explicit experimental obligation.

### P4. Make events queryable experience

Use NOOA `EventQuery`/events APIs or a narrow facade to answer questions such as:

- Which actions changed the grid?
- Which hypothesis predicted this transition?
- Which builder edits were later contradicted?
- When was this goal last tested?
- Which evidence is host-authored?

Do not render the complete event corpus into every prompt.

### P5. Add deliberate context and persistence policy

Combine:

- exact decision checkpoints for scientific replay;
- Tycho workspace for executable/evidence artifacts;
- NOOA storage/snapshots for typed agent state;
- inspectable summarization and event archival for context control.

Each layer must retain its own claim. A summary cannot replace evidence, and a
snapshot cannot replace an exact provider request commitment.

### P6. Convert proven operations into skills

Begin with narrow deterministic capabilities: evidence lookup, branch replay,
planner contract audit, symmetry transforms, and candidate-goal validation. Promotion
should require independent repeated use rather than mere availability.

### P7. Route builder work by reason

Instead of invoking one general fresh builder for every trigger, distinguish:

- cold-start construction;
- prediction-divergence repair;
- objective investigation;
- planner repair;
- terminal consolidation.

CodeAct can call the appropriate narrow capability, while an independent builder can
still be reserved for work where separation genuinely improves verification.

---

## 10. Risks of a CodeAct actor

### Safety and containment

NOOA itself warns that AST validation and module deny-lists are defense in depth,
not a containment boundary. Generated Python requires OS-level isolation. The
current separate worker process is useful but is not automatically sufficient if it
can access sensitive files or the network. The actor treatment must run with a
constrained execution backend and narrow object graph.

### Authority leakage

A facade method that appears read-only may expose a mutable object by reference.
Return immutable views or copies where mutation is not authorized. Audit transitive
object reachability, not just method names.

### Loss of semantic trace grain

One Python cell may perform several operations. Each workbench capability should
emit typed events, and the trace should preserve both the source cell and the
semantic child operations.

### Confounded comparison

Changing strategy, prompt, tools, context, budgets, and builder policy together
would make any result uninterpretable. The first comparison must change only the
action representation and its necessary facade.

### Capability without uptake

Providing a typed object does not prove that the model will use it. Positive
controls must show that the model can discover and call the facade before a null
result is interpreted as absent value.

### Python can encourage procedural overreach

CodeAct may produce elaborate computation when a single grounded action is needed.
Iteration, tokens, cell time, and action deadline all remain bounded. A final action
reserve is still required.

### Loss of Tycho parity

A CodeAct actor is a new treatment, not a faithful Tycho port. Its provenance and
claims must remain separate from the native baseline.

---

## 11. Discriminating experiment

### Question

Does a bounded CodeAct actor preserve the current control and evidence properties
while improving capability composition, inference efficiency, or action quality?

### Arms

**N — native control**

```text
NativeToolStrategy
    -> Tycho provider tools
    -> take_action
```

**C — CodeAct treatment**

```text
CodeActStrategy
    -> narrow ActorWorkbench
    -> return_result(ActionDecision)
```

An optional intermediate arm may call CodeAct only for `investigate()` and retain a
native final actor, but it should not replace the direct N-versus-C comparison.

### Hold constant

- model and provider route;
- substantive task and world-model instructions;
- exact observation and prior evidence;
- workspace bytes and mode;
- builder availability and authority;
- maximum wall-clock and action budget;
- host validation and action execution;
- response capture and evidence audit.

Token and inference budgets cannot be literally identical because CodeAct changes
the action representation. Record both and compare resource-normalized behavior.

### Measures

- valid decisions returned;
- levels completed and actions wasted;
- model calls, tokens, cost, and time per environment action;
- number and type of evidence queries;
- candidate goal -> discriminator -> probe transitions;
- planner and builder uptake;
- malformed calls and failed typed returns;
- violation or attempted violation of authority boundaries;
- ability to reconstruct why the final action was chosen;
- persistence and reuse of correct discoveries;
- branch-model accuracy separately from gameplay success.

### Positive controls

- A simple task whose solution requires reading two evidence objects and combining
  them, verifying that the CodeAct actor can discover and call the facade.
- A typed-return task verifying that malformed `ActionDecision` values are rejected
  and repaired.
- An authority test verifying that generated code cannot execute an environment
  action or mutate host evidence.
- A trace test verifying that multiple facade calls inside one cell remain visible.

### Invalidity

The comparison is invalid if:

- prompts differ in substantive evidence or advice;
- workspace or history commitments differ;
- one arm receives builder/model information the other lacks;
- provider route drifts without being detected;
- a control arm cannot use its own interface;
- action or wall-clock budgets are not enforced;
- a returned action is executed during a frozen decision-point comparison;
- a timeout or provider failure is interpreted as a capability result.

### Decision rule

Promote CodeAct to the default actor only if it:

1. preserves host authority and evidence integrity;
2. has usable, reconstructable semantic traces;
3. passes its interface positive controls;
4. improves a consequential measure or achieves equivalent behavior with materially
   lower inference cost;
5. survives more than one state, seed, or decision class.

Retain native tools as the default if CodeAct fails containment, cannot reliably
finish typed actions, or its compositional gains do not affect decisions. Revise the
facade rather than rejecting CodeAct if failures arise from missing or misleading
capability exposure.

---

## 12. Open architectural questions

1. **Who owns epistemic state?** Should candidate goals and hypotheses live on the
   persistent NOOA actor, in the Tycho workspace, or in both with one authoritative
   representation and explicit projections?
2. **What must be durable across CodeAct calls?** Per-turn session locals are useful,
   but important discoveries should become typed agent state or committed workspace
   evidence rather than accidental REPL variables.
3. **Which capabilities are safe to expose by reference?** Can evidence views and
   models be immutable while still permitting efficient analysis?
4. **Should the builder remain fresh?** Freshness supplies independence and bounded
   history; persistence could prevent repeated rediscovery. Does a persistent
   specialist with scoped memory dominate both?
5. **Should the actor call builders from CodeAct?** This improves composition but may
   turn an expensive inference loop into an invisible ordinary method. Deadline,
   cost, and trace events must remain explicit.
6. **Where should game-analysis Python run?** Can CodeAct call a separate Tycho
   sandbox as a skill, preserving isolation, rather than executing world-model code
   in the live agent session?
7. **What is the minimal facade?** Too little exposure recreates a tool menu under
   Python syntax; too much exposure destroys modularity and containment.
8. **How should action proposals express uncertainty?** `ActionDecision` currently
   captures the command. Would an internal `ActionProposal` containing expected
   effects, evidence references, falsifier, and risk improve judgment, with the host
   stripping it to the legal action?
9. **How are model-authored state and host evidence kept distinct in NOOA storage?**
   Persistence must not flatten provenance.
10. **Does CodeAct help the actual bottleneck?** If objective formation and probe
    selection remain weak, greater computational composition may only automate the
    wrong work more efficiently.
11. **Which model benefits from which loop?** Native tools and CodeAct may have
    model-specific affordances. A result for Gemma cannot be assumed for Qwen, and a
    zero-level failed run is not valid evidence about either architecture.
12. **Can one strategy adaptively hand off to another without creating bureaucracy?**
    Routing must be driven by a decision-relevant condition, not by a ritualized
    multi-agent protocol.

---

## 13. Claims and evidence boundaries

### Observed in source

- The persistent actor uses `NativeToolStrategy`, not `CodeActStrategy`.
- The native strategy repeatedly calls NOOA `RuntimeServices.generate` with Tycho
  tool schemas and completes through `take_action`.
- `run_python` delegates to Tycho `ToolExecutor`.
- Both `TychoNooaAgent` and `FocusedBuilder` suppress NOOA's `self` and `state`
  context blocks.
- The hybrid uses pinned NOOA agent/runtime/event interfaces and pinned Tycho
  workspace/tool/prompt mechanisms.
- The typed candidate-goal treatment is explicitly described as inspired by NOOA's
  typed paradigm, not as CodeAct or NOOA memory reuse.
- NOOA supports CodeAct sessions, typed returns, agent fields/methods, dynamic
  context, event queries, storage, skills, and tracing.

### Inference

- The hybrid presently embodies Tycho more deeply than NOOA.
- The NOOA actor often functions as runtime ownership around a Tycho semantic core,
  rather than as the model-visible object that owns cognition.
- A narrow CodeAct facade could preserve Tycho's safety and action boundaries while
  increasing composition and reducing serialization/inference boundaries.
- The hybrid could support earned typed capabilities that neither parent supplies
  alone in the same form.

### Not established

- CodeAct produces better ARC gameplay.
- Native tools produce better ARC gameplay.
- CodeAct is cheaper after accounting for longer code generation and execution.
- A correct world model will be used by either loop.
- NOOA memory or skills improve transfer merely because they are available.
- The builder should be removed, made persistent, or invoked from CodeAct.
- Success on ar25 seed 0 transfers to other seeds or games.

---

## 14. Source map

### Current hybrid

- `solvers/tycho_nooa/src/tycho_nooa/agent.py` — persistent actor, hidden fields,
  `_choose_action`, observation/action lifecycle
- `solvers/tycho_nooa/src/tycho_nooa/strategy.py` — custom native actor loop
- `solvers/tycho_nooa/src/tycho_nooa/tools.py` — Tycho tool surface and dispatch
- `solvers/tycho_nooa/src/tycho_nooa/builder.py` — fresh builder and construction loop
- `solvers/tycho_nooa/src/tycho_nooa/epistemic.py` — typed candidate-goal treatment
- `solvers/tycho_nooa/src/tycho_nooa/trigger_builder.py` — trigger/orchestrator routing
- `solvers/tycho_nooa/src/tycho_nooa/decision_checkpoints.py` — exact decision evidence
- `solvers/tycho_nooa/README.md` — implemented slice and explicit omissions
- `solvers/tycho_nooa/PROVENANCE.md` — pinned reuse versus local adaptations

### NOOA

- `src/nooa/agent.py` — object state, context, event query, storage, runtime
- `src/nooa/strategies/codeact.py` — `execute_python`/`return_result` loop
- `src/nooa/runtime/event_query.py` — structured event selection
- `src/nooa/context_blocks/` — static and dynamic context construction
- `src/nooa/storage/` — event backends and snapshots
- `src/nooa/skill.py`, `skill_registry.py`, `library_manager.py` — callable skills
- `src/nooa/agents/summarization.py` — summarization support
- `src/nooa/tracing/` and `trace_explorer/` — hierarchical execution traces

### Tycho

- `tycho/workspace/workspace.py` — durable game workspace and lifecycle
- `tycho/workspace/agent_tools.py` — native tool contract
- `tycho/workspace/sandbox.py` — game-analysis execution boundary
- `tycho/workspace/version_store.py` — artifact versioning
- `tycho/agent/agent.py` — original actor loop
- `tycho/agent/builder.py` — world-model construction
- `tycho/agent/wm_signal.py` — model-divergence trigger
- `tycho/agent/modes.py` — ownership/control variants
- `tycho/prompts/` — actor, builder, scribe, and reflection semantics

---

## Final orientation

The question is no longer whether CodeAct is a loop. It is whether the hybrid should
let the actor think and compose through NOOA's executable object model while keeping
Tycho's evidence, workspace, verification, and action authority intact.

The current native loop has earned its place as a faithful and controllable baseline.
It has not earned permanent architectural monopoly. The next obvious improvement is
not to delete it, nor to expose an omnipotent `self`, but to build the smallest typed
`ActorWorkbench` capable of a fair CodeAct actor treatment. Reality, not aesthetic
preference, should then decide which loop owns action selection.
