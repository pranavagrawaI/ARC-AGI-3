# Tycho flow, NOOA architecture

This is a stable solver module, not an experiment number.

## Target

Recreate the interaction form that makes Tycho's machinery usable while keeping
NOOA's explicit `Agent` / `GenerationStrategy` architecture:

```text
ARC host -> GameTurn -> TychoNooaAgent
                           |
                           v
                    NativeToolStrategy
                           |
             +-------------+-------------+
             |             |             |
        workspace       run Python   take_action
       (Tycho exact)    (Tycho exact)  (turn boundary)
             |             |
             +------> validated planner
                       (guarded hint)

After a recorded transition only:

  exact Tycho wm_signal -> fresh NOOA FocusedBuilder -> fresh report -> actor
                                  |
                         same Tycho workspace
                         no take_action tool
                         independent history
```

There is one persistent NOOA agent per game and one growing NOOA event history.
The workspace and tool executor are imported from pinned Tycho rather than
adapted into an experiment folder. The model sees ordinary native tools named
`ls`, `read_file`, `write_file`, `edit_file`, `run_python`,
`set_verbosity`, and `take_action`. `take_action` returns the sole decision for
the turn; the next observation is passed to the same agent instance.

Tycho seeds `world_model.py`, `verify.py`, and `plan.py` in the workspace. The
actor may inspect or revise them with the same file and Python tools it uses for
notes. The system prompt includes Tycho's pinned single-agent planner
instructions. A successful planner call writes `notes/validated_plan.json`;
subsequent observations expose Tycho's guarded continuation only while the
model hash, observed trajectory, and action availability still agree. The actor
continues to commit one action at a time.

Run evidence is separate from the ordinary tool trace:

- `tool-trace.jsonl`: every attempted native tool call, including validation,
  policy, and execution failures, with the provider `tool_call_id` used to join
  it to `llm-trace.jsonl`;
- `planner-trace.jsonl`: planner invocations, validated artifacts, continuation
  hints, and model/frame/action invalidations.
- `trigger-builder-trace.jsonl`: the mechanical trigger verdict, artifact
  before/after hashes, fresh-report status, and syntactic actor follow/test
  evidence;
- `builder-tool-trace.jsonl`: only the focused builder's workspace tool calls.
- `llm-trace.jsonl`: one line per model round-trip, across the actor, the scribe
  and every per-fire builder — model, token counts (prompt, completion, cached,
  reasoning), cost, and the tool names the call produced. Set
  `TYCHO_NOOA_CAPTURE_REASONING=1` to add the response's own prose and reasoning
  text. Capture is not why the flag exists on the wire: this provider returns
  reasoning unasked. The flag raises reasoning effort, so it changes the run.

Planner recognition covers Tycho's direct `python plan.py ...` form and bounded
standard-library Python wrappers such as `subprocess.run`. The planner's own
`PLANNER_COMMAND` marker supplies a post-execution fallback for an indirect form
the source detector does not recognize. A recognized planner call and a game
action cannot share one model tool batch, so planner feedback is visible before
the actor commits an action.

## Current slice

Implemented and hermetically tested:

- an independent Python 3.14 [`arc-host`](../../arc-host/README.md) with a
  persistent, versioned process boundary;
- a Python 3.12 `tycho-nooa-worker` entry point implementing that protocol;
- persistent NOOA actor identity;
- native tool calls rather than `self.*` calls inside generated CodeAct;
- exact Tycho workspace and tool-executor reuse;
- exact Tycho single-agent planner instructions at the host action-vocabulary
  boundary;
- canonical-replay planner feedback and guarded cross-turn continuation;
- durable planner invocation, artifact, continuation, and invalidation tracing;
- Tycho's exact transition-trigger path (`TriggerDispatcher` -> `wm_signal`);
- a fresh bounded NOOA builder agent per trigger, using Tycho's exact builder
  system/kickoff prompts and auto-attached verifier against the actor's existing
  workspace;
- a builder tool surface with `edit_function` but without `take_action` or
  actor verbosity controls;
- mandatory fresh `notes/world_model_report.md` delivery before the actor's
  next action;
- a goal-satisfiability gate ahead of every automatic planner probe, plus a
  reported operator/key contract audit (see `PROVENANCE.md`);
- separate evidence for harness invocation, builder construction, artifact
  hashes, and subsequent actor follow/test behavior;
- durable files readable on a later turn;
- one legal action as the end-of-turn result;
- one growing tool/event history across turns;
- successor acknowledgement, including solved-level and terminal evidence that
  arrives after the final decision turn;
- bounded old-grid eviction with durable frame breadcrumbs;
- host-authored `level_<N>/level_complete.json` evidence and a mandatory
  action-result outcome-repair checkpoint, with a level-start retry while it
  remains unresolved;
- two live ar25 seed-0 runs that completed level 0 (19 and 20 actions), each
  executing two successful symmetry calls. This is a same-configuration local
  replication, not evidence of general capability across seeds or games.

Not yet implemented:

- image attachment, bounded reasoning retention, and emergency context
  compaction (old grid payload eviction is implemented);
- Tycho's reflection cadence, context compaction, and broader orchestrator
  modes (the scribe and the LEVEL_START trigger landed in `712f892`);
- replication across another seed or game.

Those omissions are explicit. The two live runs replicate this one seed's
gameplay success and symmetry uptake; they do not establish a general capability.

## Test

The nested runtime targets Python 3.12 because NOOA is `<3.14`:

```bash
(cd solvers/tycho_nooa && uv run --frozen pytest -s)
```

`-s` avoids a WSL/Windows temporary-file capture failure in this workspace; it
does not change the tests.

### CRAP analysis

Run the solver tests under Coverage.py and print every function's CRAP score,
highest first:

```bash
(cd solvers/tycho_nooa && uv run --frozen python tools/crap.py)
```

The default source is only `src/tycho_nooa`; the host, experiments, and other
repository code are not measured. Coverage data and JSON are written to a
temporary directory and removed after the report. Use JSON for automation or a
strict intervention gate with:

```bash
(cd solvers/tycho_nooa && uv run --frozen python tools/crap.py --format json)
(cd solvers/tycho_nooa && uv run --frozen python tools/crap.py --fail-above 30)
```

The gate fails only for a raw score strictly greater than the threshold. Exit
code `1` means a CRAP violation, `2` means the analysis could not be trusted,
and `3` means pytest failed. Arguments after `--` replace the default
`pytest -s` selection.

CRAP identifies pressure rather than prescribing a mechanical refactor. When a
function exceeds 30, compare its complexity with the threshold first: if the
complexity alone is already above 30, additional coverage cannot make it pass.
Delete or simplify before refactoring, and add tests only when they constrain
meaningful behavior.

During local source development, the same tests can be run against the pinned
checkouts without resolving dependencies again:

```bash
PYTHONPATH=/home/hipra/projects/nooa/src:/home/hipra/projects/tycho \
  /home/hipra/projects/nooa/.venv/bin/pytest solvers/tycho_nooa/tests
```

The cross-interpreter smoke launches the installed Python 3.12 worker from the
Python 3.14 host, executes one fake ARC action, and checks a durable file written
through the real native tool loop:

```bash
(cd arc-host && uv run --frozen python \
  ../solvers/tycho_nooa/integration/smoke_host_worker.py)
```

For a live run, the worker reads `ARC_LLM_MODEL` and either standard provider
credentials or the explicit OpenAI-compatible gateway pair
`ARC_LLM_BASE_URL`/`ARC_LLM_API_KEY`. `ARC_LLM_USE_RESPONSES=1` selects NOOA's
Responses client.

Set `ARC_LLM_CONTEXT_WINDOW` for any model litellm cannot resolve. Without it
`client.context_window` is `None`, and NOOA's entire context-limit defence is
gated on that value: the proactive block budget in `ActorRuntime._build_messages`,
`_archive_on_context_error` (`if not stats or not ctx_window: return`) and the
reduced-`max_tokens` retry all no-op, so the first overflowing request ends the
run instead of shedding history. It is env-set rather than hardcoded so it
cannot drift out of step with `ARC_LLM_MODEL`.

`ARC_LLM_MAX_TOKENS` sets one validated positive output ceiling for the actor,
builder, and scribe. The default remains 16000. This is useful for gateways that
reserve credit against the requested maximum rather than actual completion size;
record it as part of the run configuration because changing it changes the arm.

The host invocation is:

```bash
(cd arc-host && uv run --frozen arc-host \
  --game ls20 --seed 0 --max-actions 44 --stop-after-levels 1 \
  --env-file ../.env \
  --run-dir /absolute/path/to/run -- \
  uv run --frozen --project ../solvers/tycho_nooa tycho-nooa-worker)
```
