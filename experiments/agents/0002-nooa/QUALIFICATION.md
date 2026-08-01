# Qualifying 0002-NOOA

Status: executable qualification plan for the pre-freeze candidate. This file may change as tests expose missing or unnecessary checks; it does not define the semantic identity of `0002-NOOA`.

The stable authority boundary is in [`CONTRACT.md`](./CONTRACT.md). Engineering choices are in [`IMPLEMENTATION.md`](./IMPLEMENTATION.md). Upstream provenance is in [`SOURCE_MANIFEST.json`](./SOURCE_MANIFEST.json).

## Current gate state

The current minimal tracer is qualified through G8. A prior bounded live G9
passed on LS20 through OpenRouter with Gemma, but its model-visible workspace
surface predates the final helper deletion and does not qualify the current bytes.

| Gate | State | Current evidence | Remaining load-bearing gap |
|---|---|---|---|
| G1 | pass offline | pinned source bytes and notices; pruned provider source; full generated-code import/file/runtime route matrix; forbidden-affordance mutations | repeat source commitments in the live pair |
| G2 | pass for offline fixture | cumulative memory/file loop crosses a real successor through host IPC; provider mode uses the same long-lived control path | exercise a successor through the live provider adapter |
| G3 | pass offline | forced eviction matches the 0002-R boundary; host retains audit events while generated code cannot enumerate framework history; omission-removal mutant fails | exercise eviction only if the bounded live run reaches it |
| G4 | pass offline | namespace/helpers after initial start and timeout restart, garbage-collector enumeration, concurrency, processes, packages, source, caches, IPC/run artifacts, and outside-workspace reads are closed while generic Python, NumPy, and Pillow work | inspect live traces for an unclassified route |
| G5 | pass offline | valid return stops later model/Python work; delayed background mutation, replacement, and post-accept workspace mutation are rejected; environment-step failure closes the pending action into host evidence | inspect the live successor boundary |
| G6 | pass offline | existing recovery matrix plus retryable/fatal/uncertain provider transport injection; fatal, uncertain, and exhausted-retryable paths carry exact attempt evidence and make no host action | repeat the success path in G9; no offline failure class remains open |
| G7 | pass offline | a real constructed 0002-R fixture and the NOOA fixture run the same sequence; every crossing is value-diffed for instructions, observations/images, status, tools, context boundaries, decisions, and evidence; replay uses host-owned committed observations | inspect both live provider payloads in G9 |
| G8 | pass offline | eight behavioral crossing mutants are rejected independently of wording; deletion probes kill namespace closure, omission, persistence, timeout restart, validation, and both temporal barriers | none offline |
| G9 | requalification pending | historical run `g9-20260731T153745Z-d253eb4b` passed its frozen bytes; the separate CN04 smoke validates the unchanged coordinate-action route | rerun one bounded pair on the committed minimal workspace surface; gameplay competence and cross-game generality remain outside this gate |

The provider and transport/evidence seam, exhaustive negative/mutation matrices,
and complete offline parity catalogue are green. The historical G9 establishes
interface usability for its committed predecessor only.

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

Offline paired-runner regression from the repository root:

```bash
uv run --python 3.14 python experiments/agents/0002-nooa/run_paired_probe.py \
  --offline --max-actions 2 --max-model-calls 4 \
  --max-model-rounds-per-action 2 --run-root /tmp/0002-nooa-g9-offline
```

The timeout fixtures create nested workers; run them where the operating system
process and Landlock guards are available.

Current rerun (2026-08-01): 91 Python 3.12 runtime tests and 31 Python 3.14 host
tests pass. The offline G9 runner regression is included in the host count. The
host count includes end-to-end coordinate-action and multi-layer last-layer
closure tests.

### CN04 click-route smoke

The bounded online CN04 smoke at `provider-smoke-probes/cn04/20260801T141859Z`
advertised coordinate action `6` at all three requests. It validated and crossed
two actions (`{x: 15, y: 27}` and `{x: 39, y: 36}`), each with a successor.
The third request exhausted its explicit reported-token budget before a decision;
the worker still stopped with exit code 0. Compact commitments and redaction
results are in `artifacts/FREEZE_EVIDENCE.json`; raw evidence is local only.

### CN04 long-horizon click tracer

The separate online CN04 tracer at
`provider-smoke-probes/cn04/20260801T142326Z-50a-20r-4m` completed 50 actions:
all 50 were validated coordinate action `6` decisions and all 50 recorded a
successor. It stopped at the action budget (`NOT_FINISHED`, 0 levels completed)
with a clean worker exit. This is crash-resistance and route evidence, not proof
of CN04-solving or cross-game capability. The
subsequent removal of the optional frame helper does not alter this action route;
the exact current provider pair still requires bounded G9 requalification.

### Historical live G9 evidence

The accepted run is
[`g9-20260731T153745Z-d253eb4b`](./artifacts/paired-usability-probes/g9-20260731T153745Z-d253eb4b/summary.json),
with paired-spec digest
`c6d5ebb6639d10b7f5f486ce0e380459bb9ec33db04786470c18170a5e6ae0a5`.
It used LS20 seed 0, routing model `openai/google/gemma-4-31b-it`, wire model
`google/gemma-4-31b-it`, and `https://openrouter.ai/api/v1`. The control made
seven provider crossings and the candidate five; every projected static or
tool-conditional request field matched the frozen profile. A post-run scan
found neither the inherited credential nor unredacted data-image bodies in the
provider evidence.

Two prior nonzero runs were rejected and retained as diagnostic evidence:

- [`g9-20260731T152815Z-02b334c2`](./artifacts/paired-usability-probes/g9-20260731T152815Z-02b334c2/summary.json)
  exposed Chat-style multimodal blocks crossing directly into `/responses`;
- [`g9-20260731T153352Z-c9265982`](./artifacts/paired-usability-probes/g9-20260731T153352Z-c9265982/summary.json)
  reached one action and exposed an unflattened NOOA `_batch` successor-history
  item.

Prospective regressions now require exact `input_text`/`input_image` projection,
native successor-history flattening, numeric token-field preservation, and
tool controls that are present exactly when tools are present.

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
| Retained public transitions | 8 |
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

Current deletion-probe evidence (2026-07-31):

| Removed mechanism | Independent gate result |
|---|---|
| sandbox namespace pruning | G4 produced three failures and exposed `self`, `CodeActStrategy`, and `variables()` |
| pointer-only omission insertion | G3 failed because the eviction notice disappeared |
| episode-scoped sandbox retention | G2 failed on the successor because Python memory was gone |
| timeout `restart_empty` recovery | G6 failed because the killed worker could not accept the repair cell |
| runtime action postcondition and final validation | the illegal-action fixture admitted action `99` instead of repairing |
| accepted-workspace commitment | G5 admitted a post-accept file mutation |
| pending-action guard | G5 failed; the worker's duplicate action-point identity remained only as a later backstop |

Each source mutant was applied alone, its named test was observed failing, and
the production implementation was restored before the full rerun. Eight
wording-independent behavioral crossing mutants separately expose runtime
introspection, hidden-history query, automatic retrieval, summarization,
semantic skill text, history selection, action batching, and host force-advance.
Each changes an actual instruction, observation, context, or tool crossing and
G7 rejects it. Text sentinels remain only a secondary diagnostic.

## G9 — One bounded live usability probe

After G1–G8 pass, run a small same-model, same-provider, same-game, same-budget pair. Inspect raw traces from both bases for:

- interface rediscovery;
- tool and import friction;
- context loss;
- accidental information expansion;
- provider/runtime incompatibility;
- action-boundary violations.

This licenses interface observations only. It does not establish gameplay superiority, causal treatment effect, or 0006 readiness.

Both arms receive one neutral, spec-committed usability instruction in
`NOTES.md`: use Python to create a probe file before action one, then read and
update it after the first successor and before action two. A successful verdict
requires both action boundaries, Python use at both, the persisted file, clean
shutdown, complete traces, and no blocking stop. The instruction supplies no
game semantics and is not part of the frozen base outside G9.

The paired entry point is `run_paired_probe.py`. Its `--offline` mode is the
runner regression and uses no provider. Live mode requires inherited
`ARC_LLM_API_KEY`, explicit `--game`, `--model`, `--wire-model`, and
`--api-base`, and creates a fresh same-seed game for each arm. `--model` is the
NOOA/LiteLLM routing identifier; `--wire-model` is the exact model value both
actual `/responses` payloads must carry. It never searches `.env` files.

```bash
ARC_LLM_API_KEY=... uv run --python 3.14 python \
  experiments/agents/0002-nooa/run_paired_probe.py \
  --game <game-id> --model <routing-model-id> --wire-model <wire-model-id> \
  --api-base <provider-v1-base> \
  --max-actions 3 --max-model-calls 12
```

## Freeze boundary

Freeze `0002-NOOA` only after:

1. the offline candidate passes G1–G8;
2. the bounded live probe exposes no unclassified model-visible crossing;
3. a delete-centric pass removes implementation that no independent check detects;
4. restoring the smallest removed function repairs a demonstrated failure;
5. the contract, source manifest, and executable qualification evidence agree.
