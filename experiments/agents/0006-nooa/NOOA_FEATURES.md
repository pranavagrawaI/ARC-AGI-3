# NOOA features held out of 0002-NOOA

Status: maintained deletion log and candidate inventory. The first admitted
0006-NOOA vertical slice is specified in [`CONTRACT.md`](./CONTRACT.md); this
file remains the record of candidates that have not earned admission.

Direction comes from the shared [`NOOA implementation compass`](../NOOA_IDEOLOGY.md).
This file records where that ideology meets the stricter experimental authority boundary.

This file records useful mechanisms present in the NVIDIA NOOA ARC example that are intentionally removed from `0002-NOOA` to keep it comparable with `0002-R`. They may later be reused or adapted when building the planned `0006-NOOA`, but none is automatically part of that treatment.

## Source boundary

- Local source: `/home/hipra/projects/nooa`
- Repository: `https://github.com/pranavagrawaI/labs-OO-Agents.git`
- Inspected commit: `bd51d16e2594f0bc237b14fb31e9a880ec385b39`
- Primary example: `examples/arc_agi_3/`
- License: Apache-2.0; retain upstream SPDX and attribution when copying code.

`0002-NOOA` should expose only the repaired 0002 Python/files capability contract. A feature belongs here when it is useful implementation evidence but would give `0002-NOOA` unmatched memory, retrieval, semantic guidance, context management, or action authority.

## 0002-NOOA keep/delete rule

For every mechanism, ask:

1. Does its implementation look NOOA-like?
2. Does its function look 0002-R-like?

Only when both answers are yes does it belong in `0002-NOOA`. Otherwise delete it from the
base and maintain its source, exclusion reason, disposition, and re-admission evidence here.
Entries include permanent rejections as well as possible `0006-NOOA` material; logging a
feature does not promise to reintroduce it.

### 0002-NOOA tracer-bullet disposition

Exercised through 2026-07-31 against pinned commit `bd51d16e2594f0bc237b14fb31e9a880ec385b39`:

- **Kept/adapted:** ordinary `Agent` composition, `CodeActStrategy`, event middleware,
  typed method return, deterministic `FakeLLMClient` fixtures, and NOOA's guarded Python
  worker. The worker is retained for an episode so model-created Python variables and files
  survive one-action method boundaries; each new public action point is deterministically
  reseeded into that same worker.
- **Pruned:** ARC-specific observation/action methods, trajectory and grid conveniences,
  skills, memory APIs, retrieval/reflection, summarization, helper coaching, action batching,
  force-advance, effort ladders, viewers, fleet machinery, and cross-episode seeding.
- **Added as model-invisible substrate:** a Python 3.14 host adapter, long-lived Python 3.12
  worker, versioned append-only JSONL IPC, exact public transition forwarding, legal-action
  and coordinate validation, fail-closed process supervision, filesystem confinement, and
  network denial.
- **Still withheld/pending:** production provider wiring, paired real-game parity, provider
  transport classification/retries, full provider trace export, exhaustive negative and
  mutation matrices, and any 0006 treatment mechanism. Deterministic fixtures now cover
  timeout restart, response timeout, worker death, and process-tree cleanup on Linux/WSL.

## Re-admission rule

A held-out feature may enter `0006-NOOA` only when all of the following are recorded:

1. **Missing function:** which observed 0006 failure it repairs.
2. **Mechanism mapping:** evidence operator, epistemic ledger, decision/reconciliation protocol, context assembler, or model-invisible substrate.
3. **Authority check:** why the host still does not choose semantics, relevance, beliefs, experiments, or actions.
4. **Treatment visibility:** the exact model-visible difference from `0002-NOOA`.
5. **Determinism and audit:** what state it reads/writes and how the run reconstructs that crossing.
6. **Matched comparison:** how `0002-NOOA` remains the unchanged control.
7. **Deletion test:** the smallest version whose removal recreates the observed failure.

Availability in NOOA, convenience, or apparent sophistication is not enough.

## Candidate inventory

| Held-out NOOA feature | Upstream source | Why excluded from `0002-NOOA` | Possible `0006-NOOA` role | Current disposition |
|---|---|---|---|---|
| SQLite-backed model memory | `packages/nooa-memory/src/nooa_memory/manager.py`; `examples/arc_agi_3/launcher.py::build_memory_config` | Gives the base an unmatched durable memory and retrieval interface | Storage substrate behind a model-authored epistemic ledger | **Adapt**; do not expose generic `MemorySkill` unchanged |
| Typed memory records and graph relations | `packages/nooa-memory/src/nooa_memory/schema.py` | Adds claim-like types and `supports`/`contradicts`/`refines` relations absent from 0002-R | Possible backing records for claims, evidence links, revisions, and supersession | **Adapt**; define 0006-owned claim semantics first |
| Explicit memory API | `examples/arc_agi_3/solver_agent.py::MemArcSolverAgent.knowledge_api` | Tells the model how and what to remember | Candidate model-facing ledger operations | **Redesign**; generic memory operations are not an epistemic ledger |
| Automatic spontaneous retrieval | `nooa_memory.manager`; ARC `SpontaneousConfig(enabled=True, top_k=6)` | Host-selected retrieval changes the base context and may reinforce recent beliefs | Optional implementation aid for the context assembler | **Withhold by default**; deterministic required sections and model-requested evidence come first |
| Hybrid semantic/keyword/recency retrieval | `packages/nooa-memory/src/nooa_memory/retrieval.py` | Adds host ranking and representation choices absent from 0002-R | Possible optional search projection over ledger records | **Defer**; never use as authoritative relevance selection |
| Memory reflection and background consolidation | `nooa_memory`; ARC `ReflectionPolicy(trigger="manual", background=True)` | Adds extra LLM cognition, asynchronous mutation, and potentially unmatched calls | Could propose ledger cleanup or consolidation | **Do not re-add initially**; revisions should be synchronous and actor-authored |
| Generic token-budget summarizer | `src/nooa/agents/summarization.py`; `nooa.interactive.install_summarizer` | A second LLM semantically rewrites history and may silently preserve or omit facts | Infrastructure reference for bounded context | **Replace, not reuse unchanged**; 0006 needs an auditable context assembler |
| Dynamic context blocks | `nooa.context_blocks`; ARC `self.context.set_dynamic("arc_game", ...)` | Richer context assembly surface than 0002-R | Natural substrate for current observation, active claims, unresolved mismatches, recent continuity, and requested evidence | **Reuse/adapt** with fixed ownership and ordering |
| Exact `trajectory()` access | `examples/arc_agi_3/solver_agent.py::ArcSolverBase.trajectory` | Special-purpose history interface exceeds the generic Python/files base contract | Starting point for exact evidence retrieval | **Adapt** into neutral evidence operators with stable occurrence identities |
| `latest_state()` and `game_status()` | `solver_agent.py::ArcSolverBase` | Extra model-facing convenience methods not present in 0002-R | Current-observation and public-status projections | **Likely adapt**; current observation should remain authoritative and singular |
| Grid parsing and rendering helpers | `solver_agent.py::grid_array`, `render_grid`, `grid_image` | Adds dedicated representational conveniences to the base | Candidate neutral frame projection operators | **Evaluate** against the common visual-interface contract before re-admission |
| Diff summaries and animation-frame capture | `examples/arc_agi_3/harness.py::diff_summary`; state/action records | Gives precomputed transition evidence absent from the minimal base | Candidate exact transition comparison operator and raw evidence source | **Adapt**; preserve exact cells and avoid semantic summaries as authority |
| Persistent Python helper modules | `solver_agent.py::write_helper`, `read_helper`, `load_helpers` | Explicitly coaches a world-model architecture that 0002-R leaves for the model to invent | Possible ergonomic interface for durable model-authored computations | **Defer**; generic Python/files may already supply the function |
| Model variable snapshot state | NOOA agent state and storage machinery | Adds durable structured state beyond generic files | Possible durable controller/ledger state and restart support | **Reuse as model-invisible substrate** if exact resume semantics are verified |
| Event-sourced runtime and serialized dispatch | NOOA runtime/event infrastructure | Different orchestration path, though not necessarily cognitive assistance | Strong substrate for ordering observation, commitment, action, result, mismatch, and reconciliation | **Reuse** beneath both 0002-NOOA and 0006-NOOA where model-visible parity is preserved |
| Tracing and trace explorer | NOOA tracing modules and ARC launcher | Additional operational machinery, mostly model-invisible | Audit every model-visible crossing and treatment-state transition | **Reuse as substrate**; tracing must not select semantic relevance |
| Headless dispatcher | `examples/arc_agi_3/launcher.py::_headless_dispatch` | Implementation detail, not a cognitive feature | Stable autonomous runtime for both base and treatment | **Reuse in both arms** |
| Cross-episode knowledge seeding | `examples/arc_agi_3/launcher.py::seed_knowledge` | Gives the base prior knowledge and breaks independent-run parity | Later transfer experiment after within-run continuity works | **Defer to a separate treatment/evaluation**, not default 0006-NOOA |
| Memory-vs-Markdown experiment variants | `MemArcSolverAgent`, `MdArcSolverAgent`, `experiment.py` | Answer a different experimental question from 0002 versus 0006 | Reference designs for storage ablations | **Reference only** |
| ARC solver skill | `examples/arc_agi_3/skills/grid-game-solver/SKILL.md` | Injects movement, interaction, object, wall, target, and goal-search priors | Examples of discoverability and workflow guidance | **Do not re-add**; violates the neutral input-event and semantic-authority boundary |
| Interactive world-model skill | `skills/interactive-game-solver/SKILL.md` | Prescribes `encode`/`predict` and experiment strategy | Design reference for what the model may invent | **Do not ship as treatment guidance** unless separately justified and ablated |
| Free-text action rationale | `solver_agent.py::submit_actions` | Not a typed falsifiable commitment; encourages ceremony without enforcement | Input lesson for the decision protocol | **Replace** with typed model-authored assertions and non-authoritative explanation |
| Action batching | `solver_agent.py::submit_actions`; harness `--max-actions-per-turn` | Prevents observation and reconciliation between individual actions | Possible earned macro after reliable one-action closure | **Do not re-add initially**; only after a model-authored bounded macro contract exists |
| Host force-advance on silence | `examples/arc_agi_3/harness.py` | Host chooses an environment action, contaminating model action authority | None for the core treatment | **Reject**; fail closed instead |
| Effort ladder | `solver_agent.py::_install_effort_ladder`; `launcher.py::parse_effort_ladder` | Changes inference policy and model budget across time | Operational latency/cost policy | **Defer**; must be matched across arms if ever used |
| Fleet runner, production sandbox orchestration, and scorecards | `run_multi.py`, `run_solver.py`, `sandbox.py`, `scorecard.py` | Adds deployment machinery unrelated to the minimal base comparison | Later evaluation and operations support | **Defer** until the treatment stabilizes |
| Viewer, recorder, and run artifacts | `viewer.py`, `recorder.py`, trace/result files | Duplicates existing experiment viewers and may impose another artifact schema | Source of useful audit and replay patterns | **Borrow patterns selectively**; retain one authoritative run format |

## Mapping to the planned 0006-NOOA mechanisms

### 1. Evidence operators

Most relevant NOOA sources:

- exact `trajectory()`;
- latest-state access;
- grid parsing/rendering;
- diff summaries and animation frames;
- event-sourced run records.

Likely adaptation:

```text
NOOA records and methods
→ immutable public occurrence archive
→ small semantically neutral operator catalogue
→ exact result and provenance
```

Do not inherit semantic action labels, automatic relevance, or mutable workspace files as evidence authority.

### 2. Epistemic ledger

Most relevant NOOA sources:

- SQLite memory persistence;
- typed memory records;
- graph relations;
- agent snapshot state.

Likely adaptation:

```text
generic NOOA storage
→ 0006-owned claim/revision interface
→ model-authored content and dispositions
→ host-enforced identity and history only
```

Do not equate generic memories, Markdown hypotheses, or background reflection with an epistemic ledger.

### 3. Decision protocol and reconciliation gate

Most relevant NOOA sources:

- serialized event dispatch;
- `handle()`/`submit_actions()` interception seam;
- exact action/result records;
- tracing.

Likely adaptation:

```text
model commitment
→ host admission
→ one environment action
→ successor observation
→ assertion evaluation
→ unresolved mismatch state
→ model disposition
→ restored action authority
```

Free-text rationale, action batching, and host force-advance do not satisfy this mechanism.

### 4. Context assembler

Most relevant NOOA sources:

- dynamic context blocks;
- external durable memory;
- exact trajectory retrieval;
- token-budget machinery;
- tracing.

Likely adaptation:

```text
current observation
+ active model-authored claims
+ unresolved mismatches
+ recent continuity
+ explicitly requested evidence
→ deterministic bounded packet
```

The generic LLM summarizer and automatic top-k recall are references, not the default assembler.

## Features that should remain available to both arms

If a mechanism is model-invisible and necessary for reliable execution, prefer placing it beneath both `0002-NOOA` and `0006-NOOA` rather than counting it as treatment:

- headless dispatch;
- process supervision;
- event ordering;
- exact transport logs;
- sandbox restrictions;
- provider normalization;
- tracing that does not alter context;
- deterministic failure reporting.

## Features that are not 0006 candidates

Unless the research question changes, do not promote these merely because NOOA supplies them:

- host-selected environment actions;
- source-code or game-name knowledge;
- semantic object/player/wall/goal labels;
- route planning;
- automatic objective inference;
- multiple agents or specialist models;
- background belief revision outside the actor's explicit turn;
- untracked LLM summarization or reflection calls.

## Update discipline

When pruning `0002-NOOA`, add any newly removed NOOA feature here with:

```text
source symbol
reason for base exclusion
candidate 0006 function
current disposition
re-admission evidence required
```

When a feature is admitted into `0006-NOOA`, move it from the candidate table into an implementation record that cites the exercised prototype and treatment-only diff. Do not silently delete the rejected alternatives or their reasons from this file.
