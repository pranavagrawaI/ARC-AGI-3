# NOOA ARC-AGI-3 with the project Gemma route

## Question

What does NVIDIA's object-oriented-agent harness actually make the model see and do, and what happens when the published ARC implementation is run with the same Gemma route used by this project?

The model is only the reasoning engine. The harness determines the effective environment: context, callable actions, execution, persistence, recall timing, success, and invalid-action handling.

## Provenance

- Canonical source: `https://github.com/NVIDIA-NeMo/labs-OO-Agents.git`
- Inspected revision: `bd51d16e2594f0bc237b14fb31e9a880ec385b39`
- Clone used for execution: `/tmp/labs-OO-Agents`
- Python environment: `uv sync --extra arc`
- Model: `google/gemma-4-31b-it`
- Endpoint: OpenRouter's OpenAI-compatible API
- Provider routing: ordered `deepinfra`, `chutes`, `together`; fallbacks disabled
- Reasoning: enabled and included
- Credential values were neither copied into this directory nor printed.

Runs used the upstream `memory` agent, `interactive-game-solver` skill, additive grid text plus image, seed 0, offline ARC environments, and the in-process CodeAct sandbox. The ARC game itself still ran in a separate process through append-only state/action files. These are bounded mechanism probes, not leaderboard evaluations.

## Compatibility changes required before Gemma could act

The untouched clone could reach Gemma, but it could not execute Gemma's valid tool calls. Gemma returned reasoning plus a tool call with an empty text-content field. `CompletionClient` treated any empty text accompanied by reasoning as `EmptyContentError`, retried the same model turn eight times, and discarded every `execute_python` call. The harness eventually force-advanced the game without a model action.

The preserved patch, `upstream-compatibility.patch`, makes two narrow changes:

1. carry the project's OpenRouter provider pin and reasoning request through NOOA's ARC LLM builder;
2. classify a response as empty only when it has neither text nor tool calls.

Regression tests cover synchronous and asynchronous reasoning-plus-tool-call responses. Verification:

```text
uv run pytest tests/unifiedllm/test_empty_content_retry.py \
  examples/arc_agi_3/tests -m 'not integration' -q

79 passed, 2 skipped, 1 deselected
```

This is already a harness result: the same reasoning model was unusable under the untouched response-normalization contract and usable after a two-condition compatibility fix. Model capability did not change.

## Actual ARC call path

```text
run_solver.py
  creates state/action JSONL paths and starts the game harness

launcher.py
  builds CompletionClient, the memory skill, and MemArcSolverAgent
  watches state and user-message queues
  invokes ArcSolverBase.handle(notification)

ArcSolverBase.handle()
  is an ordinary typed method whose implementation is CodeActStrategy
  prefills a color image
  exposes the current object, method contract, skill, dynamic state, and event history

CodeActStrategy
  renders the current context
  calls Gemma with execute_python and return_result tools
  runs generated Python in the stateful REPL
  appends the response, tool call, stdout/image, or exception to event history
  calls Gemma again until the method returns or submit_actions yields

submit_actions()
  validates legal action names and CLICK coordinates
  rejects invalid input as an ordinary string, allowing same-turn repair
  caps oversized batches and records the omitted suffix loudly
  appends an accepted batch to actions.jsonl
  raises _ReturnResultSignal(RespondReason.WAIT)

harness.py
  reads the accepted batch
  executes actions in the separate ARC environment process
  stops early on level completion, WIN, or GAME_OVER
  appends the resulting public state to states.jsonl
  wakes the agent on the new state
```

## What Gemma actually saw

The initial model request contained roughly 19,799 characters of static system material before live history. It included:

- the CodeAct protocol and REPL rules;
- the full `handle()` method contract;
- the interactive-game-solver skill;
- memory and helper APIs;
- the live state, legal actions, turn/level status, and 64×64 hex grid;
- a rendered color image;
- Python names including `self`, `np`, `show`, and `return_result`.

Later calls also contained prior model code and REPL results. Prompt size therefore grew with analysis: in the longitudinal `cd82` run, requests grew from 8,979 to 25,210 input tokens.

The model did not receive live Python references over the API. It received names and previews, emitted source code, and the REPL dereferenced live objects bound to those names.

## Persistence boundaries observed

| State | Persistence | Observed use |
|---|---|---|
| REPL variables and imports | Across model cells within an agentic turn | Grid arrays, diffs, coordinates, and trajectory slices were reused during analysis. |
| Agent object fields and dynamic context | Across game turns in the process | Current state, turn clock, helper namespace, and context were refreshed without transcript reconstruction. |
| Append-only state/action history | Across the whole run | `latest_state()` and `trajectory()` associated accepted actions with successor grids. |
| Typed memory | Durable within the run and independently retrievable | Gemma wrote one causal hypothesis on the fourth `cd82` turn. |
| Helper modules | Files plus imported Python under `self.h` | Available, but Gemma never wrote or loaded one in these traces. |
| Raw model conversation | Event history subject to context policy | Code, outputs, tool errors, and model calls remained visible; prompt cost grew accordingly. |

The advertised executable-world-model mechanism was therefore available but not realized in these runs. Gemma formed a narrative causal model and used direct NumPy analysis; it did not implement `encode`/`predict` helpers.

## Executed runs

Machine-readable aggregation is in `run-summary.json`; raw prompts, outputs, event logs, IPC, grids, and results are under `results/`.

| Game/run | Environment actions | Model calls | Prompt tokens | Completion tokens | REPL executions | Wall time | Levels |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ls20`, initial probe | 1 | 6 | 79,325 | 8,517 | 10 | 352.233 s | 0 |
| `cd82`, initial probe | 1 | 3 | 38,688 | 2,525 | 7 | 109.689 s | 0 |
| `cd82`, longitudinal probe | 5 | 12 | 199,349 | 15,057 | 22 | 608.359 s | 0 |
| **Total** | **7** | **21** | **317,362** | **26,099** | **39** | **1,070.281 s** | **0** |

No REPL execution raised an error after the compatibility patch.

### `ls20`

Gemma spent five CodeAct cells locating regions, rendering the grid, and testing visual hypotheses before submitting `UP`. The accepted action was legal, but the rationale referred to a supposed player coordinate that its own earlier analysis had not established. The probe stopped after the successor state, so it establishes the first-turn mechanism but not learning quality.

### `cd82`: causal learning across turns

The longer trace exercised the useful loop:

1. Gemma inspected the grid and clicked `(31, 28)`.
2. `trajectory()` plus exact NumPy diffs exposed one changed pixel at the bottom row.
3. A nearby click at `(32, 28)` produced no change.
4. Repeating `(31, 28)` changed the next bottom-row pixel.
5. Gemma revised the hypothesis to a coordinate-specific repeatable trigger and wrote it to typed memory.
6. It requested ten repeated clicks; the host-enforced cap admitted only two and recorded `truncated_from: 10`.

This is genuine harness-supported causal work. The model supplied the ontology (`button`, `trigger`) and hypothesis. The harness supplied action-conditioned evidence, exact computation, persistence, validation, and bounded execution.

It was not yet good play. Five actions, 12 model calls, 199,349 prompt tokens, and 608 seconds completed no level. The model also claimed it would proceed “one by one” and then batched ten actions, which would have hidden intermediate evidence; the deterministic cap limited that mistake but did not understand its epistemic consequence.

### `ls20`: attempted game-over run (incomplete)

The later uncapped-turn attempt is preserved at
`results/full-game/20260729_133928_ls20_memory_gemma-pinned/`. It did **not**
reach `GAME_OVER` and is not included in `run-summary.json`'s completed-probe
totals. Exact status is recorded in that directory's `INCOMPLETE.json`.

Gemma submitted 290 actions in 20 batches: 281 `UP`, six `LEFT`, and one each
of `DOWN`, `RIGHT`, and `RESET`. After an initial four-action directional probe,
it formed an unverified “consume the target” theory and repeatedly submitted
15- or 20-action `UP` batches. At environment step 290 it still had completed
zero levels.

The provider path then degraded. The last usage-bearing response followed an
87,965-input-token request; subsequent calls produced repeated LiteLLM/OpenRouter
BadRequest or timeout errors and zero-token empty/error responses. The harness
eventually force-advanced two additional `UP` actions while Gemma was silent,
bringing the environment to step 292. At that point the trace no longer measured
model behavior, so it was manually stopped and preserved rather than spending
more budget on fallback behavior.

Final incomplete-attempt accounting: 73 model calls, 1,439,498 recorded prompt
tokens, 20,043 output tokens, 82 successful REPL executions, zero REPL errors,
zero levels, and terminal state `NOT_FINISHED`. The artifacts are durable local
copies rather than symlinks into `/tmp`.

## Comparison with this project's agents

### Versus `0001`

`0001` already owns neutral mechanics that Gemma rebuilt manually in CodeAct:

- `ObservationParser` proposes same-valued connected regions without assigning semantic roles;
- tracking and event code bind observations to action-conditioned changes;
- `ControlledEntityBelief` earns control from repeated motion evidence;
- `InteractionDetector` computes blocked motion, contact candidates, and other-object changes.

NOOA's trace shows where those mechanisms can help: Gemma spent expensive calls discovering coordinates, slicing arrays, and writing ad hoc diffs. But `0001` cannot supply the semantic conclusion that a specific region is a repeatable trigger or decide which discrepancy changes the plan. That remained model work.

### Versus `0002-R` / the `0006` learning loop

The existing seed-0 `ls20` 0002-R trace at `p9/artifacts/runs/0002-R-20260726T160029Z-25b27c93` used the same Gemma model and provider request profile. In about the same elapsed time, it executed five actions with 14 model calls, 129,357 prompt tokens, 5,252 completion tokens, and eight Python calls; NOOA executed one action with six calls, 79,325 prompt tokens, 8,517 completion tokens, and ten REPL executions. Both completed zero levels.

That is suggestive, not a controlled architecture comparison: action surfaces, prompts, images, stop budgets, and environment adapters differ. It does show that NOOA's repeated intra-turn analysis did not buy more environment contact on this model in this trace.

`0006` supplies something NOOA's free-form rationale does not: a host-validated pre-action prediction, a pending-decision gate, public post-action checking, and mandatory reconciliation after mismatch. NOOA asks for a rationale and exposes later evidence, but it does not make the prediction mechanically checkable. Conversely, NOOA supplies a richer ordinary-Python composition surface and model-owned executable helpers that `0006` does not currently offer.

## Observation, interpretation, conclusion

### Observations

- Untouched NOOA rejected valid Gemma tool calls because of a response-normalization assumption.
- After the fix, generated Python executed and its outputs reached subsequent model calls.
- Gemma used live grid/state methods, NumPy, full trajectory, action submission, and eventually typed memory.
- It did not create the executable `encode`/`predict` world model promoted by the skill.
- Deterministic action validation and batch caps prevented illegal or excessively large direct execution.
- Seven actions across the probes completed no level and consumed 317,362 prompt tokens.
- The attempted full `ls20` run also completed no level: it collapsed into repeated `UP` batches, then provider failure and harness fallback at step 290.

### Interpretation

NOOA's ARC behavior derives from a package, not object orientation alone:

```text
rich skill and method contract
+ repeated stateful CodeAct calls
+ full action-conditioned trajectory
+ live Python computation
+ durable memory/helper surfaces
+ deterministic action validation and IPC containment
```

In the successful `cd82` sequence, the highest-value mechanisms were full trajectory access, exact local computation, persistent causal state, and deterministic submission gates. The executable-helper mechanism remains only a hypothesis for Gemma because it was not used.

### Conclusion

Do not port NOOA wholesale and do not dismiss it. The traces support a narrower synthesis:

- deterministic host code owns public evidence, accepted actions, caps, prediction checks, and irreversible state changes;
- neutral symbolic machinery proposes measurements and action-conditioned structure;
- the model owns semantic hypotheses, experiment choice, and revisions;
- model-authored executable helpers remain candidate state until tested against host-owned observations.

This is “hard-code the lab, not the theory” and “authority follows ownership” expressed through NOOA's strongest software-first mechanism.

## Next decision

**Recommended next experiment:** one matched five-action `cd82` probe, not a fleet and not a hybrid build.

Freeze Gemma, provider order, seed, visual representation, environment/action budget, and score extraction. Compare:

1. **Current 0002-R control** — generic workspace/code harness.
2. **NOOA control** — the patched upstream ARC harness already recorded here.
3. **One declared treatment** — NOOA's ordinary object surface backed by existing neutral `0001` measurements plus `0006`'s pre-action prediction/host-check contract. No semantic labels, game rules, routes, or goals enter host code.

The treatment earns continuation only if it produces more falsifiable action-conditioned learning per environment action without increasing invalid actions or hiding evidence. Measure at minimum:

```text
levels and RHAE diagnostics
model calls and tokens per environment action
time to first accepted action
predictions made / matched / mismatched / reconciled
hypotheses surviving a repeat test
helper code written, invoked, and falsified
invalid or truncated submissions
raw trace readability
```

Why this move: the raw NOOA trace already proves expressivity and persistence. It does not prove that executable model-owned helpers add value, and a full Gemma fleet would mostly pay to repeat that uncertainty. The matched treatment directly tests the proposed complement while keeping semantic theory out of deterministic host code.

Alternative that could dominate: if reproducing NVIDIA's published score is the objective rather than finding transferable mechanisms, run the paper's exact model/configuration and full 25-game fleet instead. That is substantially more expensive and answers a different question—total-system reproduction, not Gemma harness synthesis.

## Artifacts

- `upstream-compatibility.patch` — exact local changes to the pinned upstream clone
- `run-gemma.sh` — credential-free reproduction wrapper; reads the project's existing `.env`
- `analyze_runs.py` — deterministic run summarizer
- `run-summary.json` — aggregate metrics
- `results/representative/` — one-step `ls20` and `cd82` traces
- `results/longitudinal/` — five-action `cd82` causal-learning trace
- `results/smoke/` — untouched-client failure evidence
- `results/patched-smoke/` — shortened intermediate run; not used for outcome claims
