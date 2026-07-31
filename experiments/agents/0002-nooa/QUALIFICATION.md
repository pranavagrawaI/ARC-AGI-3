# Qualifying 0002-NOOA

Status: executable qualification plan for the pre-freeze candidate. This file may change as tests expose missing or unnecessary checks; it does not define the semantic identity of `0002-NOOA`.

The stable authority boundary is in [`CONTRACT.md`](./CONTRACT.md). Engineering choices are in [`IMPLEMENTATION.md`](./IMPLEMENTATION.md). Upstream provenance is in [`SOURCE_MANIFEST.json`](./SOURCE_MANIFEST.json).

## Current gate state

The tracer bullet is a qualification instrument, not a qualified base.

| Gate | State | Current evidence | Remaining load-bearing gap |
|---|---|---|---|
| G1 | partial | pinned upstream bytes, verbatim copies, notices, and selected forbidden imports are executable checks | complete import/file/introspection route matrix |
| G2 | pass for offline fixture | cumulative memory/file loop crosses a real successor through the host IPC | repeat through the live provider adapter |
| G3 | partial | complete-turn eviction and pointer-only omission are exercised | exhaustive hidden-event enumeration/dereference mutants |
| G4 | partial | runtime names, globals, NOOA/undeclared imports, and out-of-workspace reads fail while NumPy/Pillow work | process environment/arguments, package discovery, parent-process, cache, and IPC route matrix |
| G5 | partial | valid return stops later scripted model work and file mutation | explicit replacement-action and post-accept-hook mutants |
| G6 | partial | syntax/runtime errors, malformed/illegal actions, output bounds, budgets, timeout restart, response timeout, process death, and tree cleanup are exercised | retryable versus fatal/uncertain provider transport |
| G7 | partial | actual fixture messages/tools/observation/workspace are normalized against the frozen control constructor; unmatched fields fail | full instruction digests, errors, imports, omission, budgets, and actual provider payload |
| G8 | partial | hidden-history and runtime-term reintroduction mutants fail parity | complete required mutant matrix |
| G9 | blocked by design | none | provider adapter and G1–G8 closure |

The next implementation decision is therefore narrow: finish the provider and
transport/evidence seam, or finish the exhaustive offline negative matrices
first. The current recommendation is the provider seam because it can expose
request-shape and retry semantics that would otherwise force the matrices to be
rewritten.

### Reproduction

From the repository root:

```bash
uv run --python 3.14 python -m unittest discover \
  -s experiments/agents/0002-nooa/tests -v
```

From `experiments/agents/0002-nooa/nooa-runtime/`:

```bash
.venv/bin/python -m pytest -s tests -q
```

The timeout fixtures create nested workers; run them where the operating system
process and Landlock guards are available.

## Shared paired-run specification

A comparison constructs `0002-R` and `0002-NOOA` from one immutable paired-run specification. It must declare at least:

- model, provider endpoint, request profile, and model seed where supported;
- environment/game identifier and environment seed, host-only where required;
- total action, model-call, token, repair, timeout, and transport-retry budgets;
- number of complete recent model turns retained;
- number of recent environment transitions retained;
- whether before and after observations are both shown;
- whether action results include changed-pixel counts or only raw observations;
- image and text-grid representations;
- coordinate origin, axis order, and bounds;
- permitted Python standard-library modules and distributions;
- host-created workspace files and their initial contents or digests;
- exact omission-notice schema;
- current public status-field catalogue;
- legal-event and final-action schemas.

No paired field may be overridden independently for one base. A runner may use smaller limits than the qualification defaults below, but both bases receive the same declared values.

## Initial qualification settings

These are test and experiment defaults, not identity:

| Setting | Initial default |
|---|---:|
| Retained conversation turns | 6 |
| Model rounds per action | 10 |
| Decision repairs | 2 |
| Total model calls | 100 |
| Total provider-reported tokens | 250,000 |
| Python execution timeout | 20 seconds |
| Environment actions | 500 |
| Retryable transport retries | 2 |

## Required run evidence

Each fixture or live run must preserve enough model-invisible evidence to reconstruct:

- a credential-free canonical record of every model-visible message, image reference, tool definition, tool result, and error;
- the actual provider request/response payload where available, with credentials and image bodies safely redacted while retaining commitments to omitted bytes;
- generated Python and the exact bounded result returned to the model;
- workspace changes attributable to each model round and action point;
- context-retention decisions and omission notices;
- transport attempts and retry classification;
- the submitted final action and validation result;
- the exact successor public observation or host execution failure;
- budget configuration and aggregate usage;
- NOOA source commitment, runtime versions, participating source digests, and dirty-state marker;
- subprocess start, exit, and IPC sequence where applicable.

Logging and tracing are evidence transport only. They must not alter model-visible content, select key moments, infer relevance, or become an alternate model-accessible memory.

## G1 — Source and feature closure

- `SOURCE_MANIFEST.json` validates and every retained or copied source matches its recorded digest before modification.
- Copied files retain required license and modification notices.
- The generated-code runtime import graph and file visibility contain no NOOA, ARC host, agent implementation, `nooa_memory`, trace, IPC, provider, or controller surface.
- No skill, summarizer, reflection, seeding, batching, adaptive effort, or host-action fallback is active.
- Disabled features disappear from the model-visible interface rather than remaining as unavailable coaching stubs.
- Every removed NOOA feature is entered in `../0006-nooa/NOOA_FEATURES.md` with source, reason, disposition, and any re-admission condition.

## G2 — Cumulative Python/files loop

A deterministic fixture policy must:

1. inspect an exact current observation;
2. execute Python;
3. create a model-authored file;
4. receive the bounded Python result;
5. reuse in-memory state and the file on a later model round;
6. reuse the file after an environment successor;
7. return one legal action.

The host independently verifies the resulting file and admitted action.

## G3 — Context eviction and hidden-history closure

Under forced eviction:

- omission is explicit and matches the paired-run schema;
- current observation and declared recent continuity remain correct;
- model-authored files survive;
- no semantic summary or automatic retrieval appears;
- an evicted observation remains available only through a contracted model-writable projection;
- no authoritative or action-linked historical record appears;
- generated code cannot recover the evicted event by querying NOOA events, context blocks, agent objects, traces, IPC records, caches, or framework APIs.

A dedicated negative fixture retains the evicted event internally for host audit and proves the model cannot enumerate, search, dereference, import, or read it.

## G4 — Runtime non-introspection

From generated Python, attempt every plausible route to the host substrate:

- direct references such as `self`, agent, context, event manager, client, controller, or supervisor;
- imports and package discovery for NOOA and the ARC host;
- module-file and source inspection;
- process arguments and process environment;
- IPC directories and files;
- trace stores, run artifacts, prompt source, hidden context state, and provider configuration;
- parent-process or unrestricted process inspection.

Each route must be absent or fail with a classified interface error. The fixture must also prove that generic Python/files capability still works, so isolation cannot pass by disabling the REPL.

## G5 — Final-action temporal seam

Drive one action point through multiple Python rounds and a valid final action. Verify:

1. validation binds the final action to the current observation/action point;
2. acceptance terminates that action-point method;
3. no later model call, Python execution, file mutation, action replacement, or rationale revision occurs before environment execution;
4. the next model-visible turn begins only after the exact successor public observation or explicit execution failure is recorded.

Mutants that run a post-accept hook, permit a replacement action, or mutate files after acceptance must fail.

## G6 — Failure recovery

Inject and distinguish:

- Python syntax error;
- Python runtime error;
- oversized output;
- timeout and interpreter restart;
- malformed final action;
- illegal action ID;
- invalid coordinate payload;
- retryable pre-response transport failure;
- fatal or uncertain transport failure;
- exhausted budgets;
- runtime/IPC process death.

Assert each fault actually fired and produced the contracted routing and evidence. Silence, timeout, or malformed output must never produce a host-selected environment action.

## G7 — Model-visible parity report

Construct one `0002-R` and one `0002-NOOA` fixture run from the shared paired-run specification. Normalize and diff:

- semantic system-instruction sections and digests;
- current observation fields and representations;
- public status fields;
- model-visible methods, tools, builtins, and files;
- recent-turn and transition boundaries;
- omission behavior;
- package/import visibility;
- final-action contract and temporal seam;
- errors and recovery;
- budgets;
- exact canonical request content and actual provider payload where available.

Every difference is classified as:

- `same_capability_different_transport`;
- `measured_substrate_difference`; or
- `unmatched_affordance`.

Any `unmatched_affordance` fails qualification.

## G8 — Mutation evidence

At minimum, mutants that reintroduce any of the following must fail G1, G3, G4, or G7:

- NOOA runtime introspection;
- hidden event-history query;
- memory or automatic retrieval;
- generic summarization or reflection;
- semantic skill text;
- history selection;
- action batching;
- host force-advance.

Mutants that remove omission notices, Python/files persistence, timeout/restart reporting, action validation, or the post-accept temporal barrier must fail G2, G3, G5, or G6.

## G9 — One bounded live usability probe

After G1–G8 pass, run a small same-model, same-provider, same-game, same-budget pair. Inspect raw traces from both bases for:

- interface rediscovery;
- tool and import friction;
- context loss;
- accidental information expansion;
- provider/runtime incompatibility;
- action-boundary violations.

This licenses interface observations only. It does not establish gameplay superiority, causal treatment effect, or 0006 readiness.

## Freeze boundary

Freeze `0002-NOOA` only after:

1. the offline candidate passes G1–G8;
2. the bounded live probe exposes no unclassified model-visible crossing;
3. a delete-centric pass removes implementation that no independent check detects;
4. restoring the smallest removed function repairs a demonstrated failure;
5. the contract, source manifest, and executable qualification evidence agree.
