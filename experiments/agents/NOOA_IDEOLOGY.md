# The NOOA implementation compass

Status: shared direction document for `0002-NOOA` and the planned `0006-NOOA`. This is not a semantic contract, source manifest, or claim that every feature in the NVIDIA ARC example belongs in either arm.

Read this when the implementation begins to feel like a collection of prompts, tools, callbacks, or special cases.

## The governing idea

An agent should be built as ordinary software, with the model inserted only where open-ended judgment is needed.

Everything else follows from that.

NOOA rejects the idea that agents need a parallel universe of prompt templates, workflow graphs, callback chains, tool ceremonies, and orchestration languages. Python already has most of the right abstractions:

- classes for agents;
- fields for durable state;
- methods for capabilities;
- type annotations for contracts;
- ordinary method bodies for deterministic work;
- model-backed method bodies for open-ended work;
- normal control flow for orchestration;
- exceptions for failure;
- `async` for concurrency.

The agent is not primarily a conversation. It is a program whose selected methods happen to be implemented by an LLM loop.

A deeper formulation is:

> **The model is not the whole agent. The whole program is the agent.**

That is the break from conversation-first agent design.

In a conversation-first architecture, the model sits in the middle while tools, memory, and routing orbit it:

```text
LLM
  + tools
  + memory
  + prompts
  + callbacks
```

In NOOA, the software system is primary:

```text
Python program
  containing
  selected model-backed methods
```

The model remains powerful, but it is no longer sovereign. It is an implementation mechanism behind a method contract.

## Why PyTorch is the right analogy

PyTorch did not make neural networks useful by surrounding them with a new ceremony. Its great move was to make differentiable programs feel like ordinary Python.

The developer writes normal objects, methods, control flow, and state. Beneath that familiar surface, PyTorch records a computation and gives it a deeper execution meaning.

NOOA makes the analogous move for agents:

> Use normal Python objects, methods, state, types, and control flow—and let the framework make selected methods agentic.

| PyTorch | NOOA |
|---|---|
| `nn.Module` is a model | `Agent` subclass is an agent |
| Fields hold parameters and submodules | Fields hold durable state and capabilities |
| `forward()` defines behavior | Agentic methods define behavior |
| Ordinary Python composes the computation | Ordinary Python orchestrates the agent |
| Autograd interprets executed operations | The strategy runtime interprets model-backed methods |
| Tensors are live program objects | Method arguments and state are live program objects |
| Framework machinery stays beneath Python | LLM-loop machinery stays beneath the method call |

The shared design instinct is simple:

> Do not force the developer to think in the framework's machinery. Let them think in the host language.

This matters twice for agents. Humans know Python, and models have learned a great deal of Python. A narrow agent DSL discards competence both already possess.

There is one important difference. Autograd is mathematically exact once the operations are defined. An LLM loop is stochastic, semantically fallible, and capable of changing strategy mid-execution. NOOA can inherit PyTorch's programming model, but not its reliability guarantees. Agentic methods still need containment, traces, type validation, state ownership, and explicit authority boundaries.

The shortest comparison is:

> **PyTorch made differentiable programs look like Python. NOOA tries to make agentic programs look like Python.**

## The governing principles

### 1. Reuse programming abstractions before inventing agent abstractions

If normal Python can express the orchestration, use normal Python.

Instead of declaring a graph:

```text
planner → researcher → critic → executor
```

write the program:

```python
plan = await self.plan(task)
result = await self.execute(plan)

if not self.verify(result):
    result = await self.revise(result)
```

The point is not that Python is convenient. The point is that agent behavior remains in the same conceptual world as ordinary software. It can be inspected, tested, refactored, versioned, and composed without translating between a program and a second orchestration language.

### 2. Treat agentic behavior as a method call

The application should be able to say:

```python
result = await agent.solve(task)
```

The caller should not need to know how many model turns, Python executions, retries, or nested calls occurred inside the method.

The method signature declares what enters and what may return. The object's fields declare durable state. The method instruction declares the open-ended task. The return type defines what must validate before ordinary software resumes.

The loop may be fluid. Its boundary should not be.

### 3. Move exact work out of the model

The division of labor is:

```text
deterministic computation → normal Python
open-ended semantic judgment → model-backed method
```

Counting, parsing, filtering, arithmetic, exact comparison, validation, and state transitions should not be performed through prose when code can do them reliably.

Reserve the model for work such as:

- interpretation;
- synthesis;
- ambiguity;
- hypothesis generation;
- semantic compression;
- open-ended planning.

The aim is not to make the model do everything. It is to place the model exactly where ordinary computation stops being enough.

### 4. Let the model compute, not merely select tools

A tool menu gives the model a fixed set of verbs:

```text
search()
filter()
retrieve()
submit()
```

A programming language over allowed objects gives it loops, conditions, helper functions, aggregation, simulation, search, and composition. Several exact operations can occur inside one computational step without paying one conversational turn per primitive.

Models already know Python. Use that competence directly, inside a bounded execution environment, instead of teaching them a cramped agent DSL.

### 5. Treat the context window as a viewport

A large object does not need to be serialized into the prompt to remain available.

The model may see:

```text
records = list(len=1_000_000, head=[...], tail=[...])
```

and inspect the complete live object through Python.

This separates two ideas that conversation-first systems often collapse:

```text
accessed information
    what is currently rendered in context

accessible information
    what the model may inspect through computation
```

The context window is a viewport, not the entire world.

This power creates an authority question: which objects are allowed to be accessible? NOOA's mechanism makes live objects possible. The experiment contract decides which live objects the model may actually reach.

### 6. Make state explicit

State should live in named program structures, not only in the residue of a transcript.

```python
self.hypotheses
self.todo
self.memory
self.world_model
self.current_task
```

Explicit state can be inspected, tested, reset, persisted, and traced. Invariants can be asserted over it. Mutations can be attributed.

An agent should not have to reconstruct its own condition from conversation every turn.

This does not imply that the host should author the agent's beliefs. It means that when durable state exists, its owner and mutation rules should be explicit.

### 7. Replace prompt conventions with executable contracts

Do not rely on prose where code can enforce the boundary.

Instead of asking for an answer with evidence and confidence, define the return type:

```python
class Result(BaseModel):
    answer: str
    evidence: list[str]
    confidence: float
```

Validation does not make the answer true. It prevents malformed completion from masquerading as success.

> Tell the model what to do through language. Enforce what can be enforced through code.

### 8. Make the computational surroundings inspectable—when authorized

NOOA can expose context, events, memory, and live state through explicit APIs. This lets the model reason about its own computational surroundings rather than treating the harness as invisible preprocessing.

That is a powerful default for building capable agents. It is not an automatic experimental entitlement.

For `0002-NOOA`, the model may inspect only the generic Python/files world already permitted by the repaired 0002 contract. NOOA's agent objects, hidden context, event history, traces, IPC, provider client, and memory machinery remain outside the generated-code environment. The substrate may retain them for execution and audit; the model may not inherit them as a richer history or introspection API.

This is not a rejection of NOOA's ideology. It is the ideology applied under a controlled authority boundary:

> Build the mechanism natively in NOOA. Expose only the function the experimental arm has earned.

## What this means for ARC

In a strong ARC agent, the harness should own exact work:

- the real environment;
- state transitions;
- action validation;
- observation recording;
- process isolation;
- terminal detection;
- exact grids;
- execution ordering.

The model should own judgment:

- interpretation;
- hypothesis generation;
- deciding what to inspect;
- writing and testing world-model code;
- choosing experiments;
- selecting actions;
- revising its own beliefs.

Within one action point, a model-backed method may perform several computations before returning:

```text
receive state
→ inspect public data
→ compute exact measurements
→ test helper code
→ revise a hypothesis
→ return one action
```

The social decomposition can disappear without losing the functional decomposition. Observer, simulator, critic, explorer, and planner do not need to become five agents. Their useful functions can become deterministic methods, explicit state, executable helpers, and one model-backed method that decides when to use them.

NOOA removes multi-agent theater. It does not remove structure.

## The controlled-experiment tension

The existing NOOA ARC example contains more than a neutral runtime. It includes memory, retrieval, semantic skills, event history, summarization, helper conventions, and other cognitive affordances.

Those features may be good NOOA. They are not automatically good `0002-NOOA`.

When implementation becomes confusing, use both tests:

```text
Does the implementation look NOOA-like?
Does the function look 0002-R-like?
```

Only if both are true does the mechanism belong in `0002-NOOA`.

```text
NOOA-like implementation AND 0002-R-like function
→ keep in 0002-NOOA

otherwise
→ delete from 0002-NOOA
→ record in 0006-nooa/NOOA_FEATURES.md
```

Examples:

- NOOA's native event dispatch carrying the same current public observation: **keep**.
- NOOA's typed return validation enforcing the same final action: **keep**.
- NOOA's hidden full event history becoming queryable from generated Python: **delete and log**.
- Automatic semantic memory retrieval: **delete and log**.
- ARC strategy skills: **delete and log; likely reject permanently**.
- Model-invisible tracing that changes no model-visible surface: **keep as shared substrate**.

This rule prevents two symmetrical failures:

1. copying the NVIDIA example so faithfully that the control inherits treatment cognition;
2. pruning so aggressively that the result stops being a real NOOA agent and becomes the old harness wearing a NOOA import.

## Direction checks

When the design feels uncertain, ask:

1. **Could exact Python own this?** If yes, why is a model call involved?
2. **Is the model here because judgment is open-ended, or because implementation was easier in prose?**
3. **Does this look like an ordinary class, field, method, return type, exception, or control-flow decision?** If not, what genuinely requires a new abstraction?
4. **Is durable state explicit, owned, and inspectable?**
5. **Does the method boundary remain stable while the internal loop stays agentic?**
6. **Is generated Python operating on authorized live objects, or quietly inheriting the framework itself?**
7. **Does validation enforce structure without pretending to establish truth?**
8. **Does the implementation preserve the paired arm's information boundary?**
9. **Would deleting this mechanism break an exercised 0002-R-like function?** If not, delete it.
10. **If deleted, is it in the maintained deletion log with its source and possible future role?**

## What NOOA does not solve

A better software body does not guarantee a better mind.

NOOA does not by itself solve:

- hypothesis quality;
- goal discovery;
- abstraction selection;
- experiment ranking;
- premature convergence;
- metacognition;
- knowing what is unknown;
- deciding which cognitive operation is needed next.

It improves the conditions under which those abilities operate. Memory failure can become less likely. Exact comparison becomes easier. State becomes inspectable. Malformed completion can be blocked. Predictions can become executable.

Once those failures are removed, the remaining weakness becomes clearer: the model may still think badly, ask the wrong question, or settle on the wrong abstraction.

That is not a failure of the ideology. It is what the ideology helps us see.

## One paragraph

NOOA treats agent engineering as software engineering rather than prompt choreography. The complete program—not the language model—is the agent. Deterministic work remains deterministic code; open-ended judgment becomes a typed model-backed method. The model acts through ordinary Python over authorized live objects, while explicit state, validation, event ordering, and containment provide the durable structure around it. The goal is not to extend a model with miscellaneous tools. It is to build an intelligent software system whose selected components use a model where semantic flexibility is necessary.

The shortest formulation is:

> **Build the agent as a program. Use the model as a programmable source of judgment inside that program.**
