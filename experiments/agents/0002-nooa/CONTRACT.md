# 0002-NOOA pre-freeze contract

Status: build authority for the first working candidate. Not frozen. Revise this contract when executable evidence shows it is wrong or unnecessarily strong.

## 1. Identity and question

`0002-NOOA` is the repaired 0002 minimum Python REPL capability implemented on NOOA.

It answers:

> Can NOOA provide a cleaner, more reliable, and more extensible implementation base for the same semantically neutral model-directed Python/files agent represented by `0002-R`?

It does **not** answer whether the planned 0006 capability environment improves ARC play. Neither `0006` nor `0006-NOOA` is currently implemented.

The intended families are:

```text
0002-R    → planned 0006
0002-NOOA → planned 0006-NOOA
```

The first comparison is base against base. A later `0006-NOOA` must be constructed as a declared treatment delta over a frozen `0002-NOOA`, not by comparing an enriched NOOA agent directly with `0002-R`.

## 2. Authority

The model owns:

- interpretation of frames and transitions;
- action semantics learned from evidence;
- representations, notes, programs, hypotheses, predictions, and plans it chooses to create;
- experiment and action choice;
- revision.

The host may:

- transport exact public observations and legal input-event IDs;
- execute model-authored Python under explicit restrictions;
- persist model-authored files for the current run;
- validate the final action syntax and public legality;
- enforce declared resource budgets;
- report errors, truncation, omission, restart, and transport behavior;
- preserve a reconstructable, credential-free record of every model-visible crossing,
  including canonical pre-serialization content and the actual provider request/response
  payload where available.

The host must not choose:

- semantic frame interpretation;
- relevant evidence or history;
- hypotheses, objectives, predictions, experiments, plans, or actions;
- belief revision or mismatch diagnosis;
- game-specific tool routing.

Event labels describe input events only. A label must never be represented as evidence that the event moves, clicks, interacts, confirms, resets, or has any other game effect.

## 3. Reference boundary

The reference behavior is the frozen `0002-R` contract at:

- [`../0006-capability-environment/p9/CONTRACT.md`](../0006-capability-environment/p9/CONTRACT.md)
- [`../0006-capability-environment/p9/README.md`](../0006-capability-environment/p9/README.md)

`0002-NOOA` targets **capability and information parity**, not byte-identical requests or identical internal architecture. NOOA is the independent substrate under examination, so its CodeAct/runtime protocol may differ where the difference is explicit, measured, and does not add cognitive assistance.

Historical `0002-R`/P9 files are read-only references for this work. Building `0002-NOOA` must not modify or re-freeze them.

## 4. Required model-facing capability

At each action point the model receives only:

1. the current rendered public observation;
2. public status fields explicitly emitted by the environment and listed in the shared
   paired-run specification, with no derived progress, reward interpretation, or
   host-authored status summary;
3. currently legal input-event IDs and any public coordinate requirements;
4. an explicit description of the available observation/file representations;
5. bounded recent model/tool/action continuity;
6. one generic persistent Python/files capability for this independent run.

The model may repeatedly:

```text
inspect current public observation
→ execute Python
→ inspect the result
→ write or read model-authored files
→ execute more Python
→ return one legal environment action
```

Python and files are generic capability. The model may invent notes, ledgers, world models, planners, transition indexes, or hypothesis stores through them, but the host does not prescribe or maintain those structures.

Declared system packages use normal imports. Model-authored workspace files remain available as relative-path data and persistent storage, but the writable workspace is not a module search path. Reusable names and functions persist directly between validated Python cells until an explicitly reported interpreter restart.

The paired-run specification must declare:

- the number of complete recent model turns retained;
- the number of recent environment transitions retained;
- whether before and after observations are both shown;
- whether action results include changed-pixel counts or only raw observations;
- image and text-grid representations;
- coordinate origin, axis order, and bounds;
- permitted Python modules and distributions;
- workspace files created by the host;
- the exact omission-notice schema;
- the current public status-field catalogue.

These fields are exactly as declared by the shared paired-run specification. Their values
are experiment settings, not part of this identity contract.

### Final action contract

The model-visible final answer has the same semantic shape as 0002-R:

```json
{"action": 1, "data": null, "reason": "brief model-authored string"}
```

- `action` is one currently legal integer input-event ID.
- The model is instructed to use `null` unless the public event contract requires
  an object. To match frozen `0002-R` exactly, the parser also accepts and preserves
  an object for a simple action, and permits additional keys for a coordinate action;
  required integer `x` and `y` values and bounds are still enforced.
- `reason` is non-authoritative prose. It is not a required prediction, claim, or reconciliation record.
- Exactly one environment action is admitted per successor observation.
- The action is validated against the current public legal-action set and coordinate bounds.
- At each arm's shared final-action boundary, the exact lowercase string `"null"` is canonicalized to JSON null before validation. Every other string remains invalid. This repairs one observed provider serialization defect without relaxing action legality, coordinate requirements, bounds, or paired parser behavior.
- Other invalid final output returns a bounded, explicit repair opportunity rather than being silently coerced.

NOOA may use an internal return/yield mechanism to deliver this object, but it must not expose an unmatched batched `submit_actions()` interface.

## 5. Observation and history contract

`0002-NOOA` must match the information available to `0002-R` without silently strengthening its epistemic record.

Required:

- exact current public pixels and metadata;
- the current-layer/render convention declared by the shared paired-run specification;
- the coordinate convention declared by the shared paired-run specification;
- deterministic run-local observation filenames and documented construction rule;
- persistent model-writable observation projections for the run;
- bounded recent action/transition continuity exactly as declared by the shared paired-run
  specification;
- explicit pointers to older observation files after conversational eviction;
- fresh state for every independent run.

Forbidden in the base:

- a host-owned immutable episode archive exposed to the model;
- a special `trajectory()` history API;
- automatic relevant-history selection;
- semantic transition summaries;
- an executed-transition log with unique long-horizon action→result information beyond the reference contract;
- host-restored observation files after model mutation.

This deliberately preserves the 0002-R limitation: historical workspace observations are available but model-writable, and beyond bounded recent continuity the base does not guarantee an authoritative action→successor causal record. A future immutable evidence archive or exact transition operator is a declared `0006-NOOA` candidate, not hidden substrate.

Historical workspace projections are convenience artifacts, not host-certified evidence.
Once created, their contents may be modified or deleted by the model, and subsequent reads
do not prove fidelity to the original observation. An omission pointer states only where an
earlier projection was written; it does not certify that the file still exists or remains
unmodified.

The game identifier, benchmark name, source implementation, hidden state, non-public reward information, and prior-run files remain absent from the model conversation and workspace.

## 6. Python and workspace contract

The generated-code environment must provide:

- one fresh writable workspace per independent run;
- persistent in-memory state and files across model rounds and environment actions until an explicit interpreter restart;
- the Python modules and distributions declared by the shared paired-run specification;
- bounded textual results with truncation metadata;
- explicit timeout and interpreter-restart semantics;
- no repository, game implementation, host home, credentials, prior runs, unrestricted process control, or network access;
- one concrete documented import/path convention requiring no guesswork.

The workspace may contain neutral interface documentation and public observation projections. It must not contain:

- ARC strategy or ontology;
- game names or source code;
- pre-authored hypotheses or plans;
- NOOA memory stores;
- prior-episode knowledge;
- semantic helper APIs.

Generic files are enough. Dedicated `write_helper`, `load_helpers`, hypothesis, memory, recall, or reflection APIs are not part of `0002-NOOA`.

Generated Python must not receive references to the NOOA agent object, context manager,
event manager, provider client, host controller, IPC transport, trace store, or process
supervisor.

The generated-code environment must not permit importing, locating, reading, or
introspecting the NOOA runtime, ARC host package, agent implementation, prompt source,
hidden context state, IPC files, trace artifacts, process arguments, or host environment
variables. Only the explicitly documented generic Python/files surface is model-visible.

## 7. Context contract

The active context is bounded and omission is explicit.

Both bases derive context and resource budgets from one shared paired-run specification so
they cannot drift independently.

Required context behavior:

- current public observation is always present;
- recent turns remain ordered;
- tool calls and results remain paired;
- eviction removes whole eligible history units;
- every omission is represented by a pointer-only notice;
- model-authored workspace files survive conversation eviction;
- no host-authored semantic summary is generated;
- no embedding, keyword, graph, or recency system selects memories for injection;
- no background LLM call rewrites or consolidates context.

NOOA context blocks may implement transport and ordering. They may not add hidden relevance or semantic selection.

NOOA may internally retain complete events for model-invisible tracing and context
construction. The model may not query, enumerate, search, dereference, or otherwise recover
evicted events through Python, agent methods, framework objects, files, or hidden APIs. After
an event is evicted from active context, it remains model-accessible only through a
representation explicitly permitted by this contract.

## 8. Final-action lifecycle

Before a valid final action, the model may use multiple bounded model/Python rounds. A valid
final action closes that action point. Acceptance binds the action to the current observation
and prevents any further model call, Python execution, file mutation, action replacement, or
rationale revision before the environment executes it.

The next action point begins only after the exact successor public observation or an explicit
host-side execution failure is recorded. This seam is part of the base contract; it does not
require a prediction or reconciliation protocol.

## 9. Failures and host recovery authority

The model or run record must distinguish at least:

- Python syntax and runtime failures;
- output truncation;
- execution timeout and interpreter restart, including what state was lost or retained;
- malformed generated-code or final-action requests;
- illegal action IDs and invalid coordinate payloads;
- retryable and fatal transport failures;
- exhausted declared budgets;
- agent-runtime or transport-process failure.

Only demonstrably idempotent pre-response transport failures may be retried automatically.
Every attempt is recorded. Model-repairable failures return exact actionable evidence within
the declared budget. Host failures must not be rewritten as game evidence or model mistakes.
Silence, timeout, or malformed output never licenses a host-selected environment action.

## 10. Forbidden cognitive affordances

The following are excluded even when the NVIDIA NOOA ARC example supplies them:

- `MemorySkill`, SQLite model memory, spontaneous recall, hybrid retrieval, and memory-graph traversal;
- reflection, consolidation, or generic LLM summarization;
- Markdown hypothesis/knowledge variants supplied by the host;
- ARC solver skills, world-model coaching, and semantic action descriptions;
- `trajectory()` or any model-facing history convenience API beyond the paired-run declaration;
- dedicated persistent-helper coaching;
- cross-episode knowledge seeding;
- action batching or macros;
- host force-advance actions;
- adaptive effort ladders unless identically configured by the paired-run specification;
- route planning, automatic objective inference, specialists, or multiple agents.

Game-status helpers are forbidden unless they expose only the exact current public status
fields already required by the paired-run specification. The name is not disqualifying;
information expansion is.

The maintained deletion log and re-admission rules live at
[`../0006-nooa/NOOA_FEATURES.md`](../0006-nooa/NOOA_FEATURES.md). Presence there does not
authorize a feature in either planned treatment.

## 11. Allowed substrate differences

NOOA is the substrate under test. Native agent lifecycle, CodeAct execution, event dispatch,
provider serialization, context storage, internal persistence, process supervision, and
model-invisible tracing may differ from `0002-R`.

An allowed implementation difference is not automatically an allowed information difference.
Any new model-visible method, prompt instruction, context block, state field, file, retry,
generated-code builtin, or automatic LLM call must be classified as either:

1. a different transport for a function already present in `0002-R`; or
2. an unmatched affordance to remove and record in the deletion log.

The implementation rule is conjunctive: keep a mechanism in `0002-NOOA` only when its
implementation is NOOA-like **and** its function is 0002-R-like.

## 12. Claims not licensed

Satisfying this contract does not establish:

- behavioral equivalence with `0002-R`;
- superior gameplay or action efficiency;
- that NOOA is the chosen 0006 substrate;
- that 0006 or 0006-NOOA exists;
- long-horizon semantic continuity;
- benefit from any held-out 0006 mechanism.

It establishes only the intended semantic and authority boundary for a native-NOOA
implementation of the minimal 0002 Python/files agent.
