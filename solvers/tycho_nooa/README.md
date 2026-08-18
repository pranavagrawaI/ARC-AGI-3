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

Planner lifecycle evidence is separate from the ordinary tool trace:

- `tool-trace.jsonl`: every executed native tool call;
- `planner-trace.jsonl`: planner invocations, validated artifacts, continuation
  hints, and model/frame/action invalidations.
- `trigger-builder-trace.jsonl`: the mechanical trigger verdict, artifact
  before/after hashes, fresh-report status, and syntactic actor follow/test
  evidence;
- `builder-tool-trace.jsonl`: only the focused builder's workspace tool calls.

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
- separate evidence for harness invocation, builder construction, artifact
  hashes, and subsequent actor follow/test behavior;
- durable files readable on a later turn;
- one legal action as the end-of-turn result;
- one growing tool/event history across turns;
- successor acknowledgement, including solved-level and terminal evidence that
  arrives after the final decision turn.

Not yet implemented:

- image attachment and Tycho's context-eviction policy;
- Tycho's reflection cadence, context compaction, and broader orchestrator
  modes (the scribe and the LEVEL_START trigger landed in `712f892`);
- a live model run proving spontaneous note or world-model use.

Those omissions are explicit. This tracer establishes the architecture and its
test surface; it does not claim gameplay or model-uptake success.

## Test

The nested runtime targets Python 3.12 because NOOA is `<3.14`:

```bash
(cd solvers/tycho_nooa && uv run --frozen pytest -s)
```

`-s` avoids a WSL/Windows temporary-file capture failure in this workspace; it
does not change the tests.

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
Responses client. The host invocation is:

```bash
(cd arc-host && uv run --frozen arc-host \
  --game ls20 --seed 0 --max-actions 44 --stop-after-levels 1 \
  --env-file ../.env \
  --run-dir /absolute/path/to/run -- \
  uv run --frozen --project ../solvers/tycho_nooa tycho-nooa-worker)
```
