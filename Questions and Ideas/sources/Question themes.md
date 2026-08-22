# Proposed problem themes

### 1. Observation and state construction
Does the agent extract the relevant objects, relations, changes, and latent state from the raw observation?

Failures include bad segmentation, missed changes, confusion between static and movable objects, and treating hidden state as visible state.

### 2. Representation and abstraction
Has the agent chosen a representation in which the problem becomes tractable?

Examples: pixels versus objects, absolute coordinates versus displacement, colours versus shape equivalence, individual moves versus transformations.

### 3. Context creation
What information enters the active context, in what form, and with what emphasis?

This covers prompts, observations, tool descriptions, verifier feedback, summaries, and the ordering that determines what becomes salient.

### 4. Context management
What is retained, compressed, evicted, repeated, or allowed to dominate as the run grows?

Failures include stale beliefs, unbounded reasoning, repeated full grids, buried instructions, and compression that removes the decisive distinction.

### 5. Memory and consolidation
Does experience become durable knowledge that is available when needed?

This includes actor beliefs, world-model notes, level summaries, stable object names, and whether conclusions survive lane or level transitions.

### 6. Goal acquisition and terminal recognition
Does the system discover what success means, distinguish candidate goals from verified outcomes, and recognize success after it occurs?

The constant `outcome() == "ongoing"` belongs here.

### 7. World-model fidelity
Does the constructed model reproduce the observed dynamics and relevant terminal conditions?

Separate:

- transition accuracy
- hidden-state modeling
- boundary conditions
- terminal predicates
- uncertainty masks
- action availability

### 8. Causal grounding
Does the agent know which observations support which mechanism, or has it merely written a compatible story?

This is where interventions, counterexamples, and discriminating probes matter.

### 9. Uncertainty and calibration
Does the agent distinguish confirmed, provisional, contradicted, and unknown claims?

Failures include premature certainty, permanent indecision, and silently treating missing evidence as negative evidence.

### 10. Exploration and probe design
Does each exploratory action discriminate among live hypotheses?

A stronger standard is **dual-purpose probing**: when equally informative actions exist, choose the one that also advances the most plausible plan. In this run, DOWN and LEFT would have been better probes than UP and RIGHT.

### 11. Planning and search
Once the model is sufficient, can the agent derive and maintain a route?

Failures include arithmetic errors, wrong coordinate frames, invalid operators, failure to replan, and searching before the goal or dynamics are grounded.

### 12. Action discipline
Does reasoning terminate in exactly one timely, legal, intentional action?

This includes premature action, delayed commitment, redundant confirmation, illegal parameters, and continuing exploration after a route is known.

### 13. Metacognition
Can the agent notice that it is stuck, repeating itself, operating with a broken model, or spending effort without information gain?

Metacognition should govern strategy changes—not merely produce prose such as “I’m stuck.”

### 14. Tool discoverability and selection
Does the agent know which tools exist, what distinctions they encode, and when each is useful?

The symmetry library’s existence and quotient choices belong here.

### 15. Tool semantics and composability
Can tools be called naturally and combined without accidental incompatibilities?

Examples include `point='full'`, inconsistent parameter conventions, ambiguous success statuses, and output formats that require brittle parsing.

### 16. Interface friction
How much unnecessary translation is required between the agent’s intention and a valid operation?

This includes schemas, coordinate conventions, tool call syntax, overly large outputs, poor error messages, and advisory prose masquerading as structured data.

### 17. Ownership and lane handoffs
Which component owns each decision, artifact, and repair—and can evidence cross that boundary?

Actor, builder, scribe, verifier, host, and planner can all possess different truths. The important question is not merely whether a file can be read, but who is required to act on it.

### 18. Triggering and control flow
Does the right process run at the right event?

Failures include:

- builder fires too early or too often
- completion repair waits for a turn that never occurs
- divergence triggers repeat without new evidence
- critical work depends on an optional successor event

### 19. Verification and promotion
What must be checked before a hypothesis, model change, or capability is trusted?

This includes verifier coverage, false-positive prevention, false-negative terminal models, promotion gates, and whether verification tests the claimed mechanism rather than a proxy.

### 20. Evidence integrity and observability
Can we reconstruct what the model saw, thought, called, received, changed, and executed?

This includes correlation IDs, reasoning capture, attempted-versus-executed calls, tool status semantics, durable terminal evidence, and trace completeness.

### 21. Credit assignment
Can we tell what caused success or failure?

The builder was present in this run, but its final recommendations were ignored. Presence is not causation. Credit assignment separates actor reasoning, tool uptake, builder influence, environmental luck, and instrumentation artifacts.

### 22. Speed and latency
How long does an action or builder pass take?

This is operational time: provider latency, sandbox startup, repeated calls, serialization, and blocking handoffs.

### 23. Efficiency and resource economics
How much useful information or progress is obtained per action, token, tool call, dollar, and context slot?

A system can be fast but wasteful, or slow but efficient. The final two 43-tool builder passes are primarily an efficiency failure.

### 24. Stopping, backoff, and abandonment
Does the system stop an unproductive procedure without abandoning the whole task?

Needed mechanisms include:

- no-progress detection
- repeated-probe suppression
- builder backoff
- hypothesis abandonment
- plan commitment
- graceful budget exhaustion

### 25. Robustness and recovery
Can local failures remain local?

A malformed builder recommendation should not terminate the actor. A builder outage after a confirmed win should not erase the win. Recovery includes retries, fallbacks, rollback, and preserved pending work.

### 26. Provider and runtime reliability
Can external failures be distinguished from agent failures?

Examples: credit limits, context-window admission, network errors, sandbox death, subprocess protocol failure, and Python-version boundaries.

### 27. Correctness bugs
Where does implementation behavior  violate its intended contract?

This deserves separation from bloat. A three-line parser bug can be catastrophic without the system being generally overcomplicated.

### 28. Complexity and bloat
Which structures increase the number of states, paths, concepts, and failure surfaces without earning corresponding capability?

Look for duplicated dispatch logic, multiple competing truth sources, excessive tool surfaces, long prompts, redundant reports, and policies whose removal changes nothing.

### 29. Modularity and changeability
Can one mechanism be repaired or ablated without silently changing unrelated behavior?

This matters for both engineering and causal experiments. Deep modules should hide complexity rather than spread protocol knowledge across lanes.

### 30. Reproducibility and variance
Does behavior survive reruns, seeds, games, provider fluctuations, and minor timing differences?

A same-seed replication addresses stochastic fragility only partially. Transfer is a separate question.

### 31. Generalization and transfer
Which learned abstractions survive changes in layout, colours, transformations, mechanics, and goals?

The important distinction is between:

- memorizing ar25
- transferring symmetry reasoning
- transferring investigative practice
- transferring the architecture’s capability

### 32. Evaluation and metric validity
Does the measurement mean what its name claims?

This includes denominator choice, exception handling, post-hoc edits, success definitions, action-cap censoring, and whether a metric measures reach, execution, causal use, or outcome.

### 33. Provenance and contamination
Can we distinguish contestant evidence from oracle knowledge, inherited mechanisms, developer knowledge, and post-run interpretation?

This protects both scientific claims and future code from accidental game-specific priors.

### 34. Human inspectability
Can we understand and judge the system without reconstructing it manually from four partially aligned logs?

Good inspectability should reveal:

```text
observation → belief change → tool evidence → plan → action → consequence
```

### 35. Capability governance
Are capabilities introduced only when earned, bounded by evidence, and removable through clean ablations?

This covers earned autonomy, tool admission, verifier gates, and preventing an experimental convenience from becoming an unexamined permanent dependency.

## The higher-level map

These themes can be grouped into six layers:

1. **Seeing** — observation, representation, context  
2. **Knowing** — memory, world models, uncertainty, goals  
3. **Thinking** — exploration, planning, metacognition, tools  
4. **Acting** — interfaces, ownership, triggers, action discipline  
5. **Surviving** — efficiency, stopping, robustness, runtime reliability  
6. **Learning about the system** — observability, attribution, evaluation, reproducibility, provenance