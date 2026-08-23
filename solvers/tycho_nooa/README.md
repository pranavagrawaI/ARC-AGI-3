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

Builder authority depends on the frozen Tycho mode:

  trigger:       exact Tycho wm_signal -> fresh FocusedBuilder -> actor
  orchestrator:  actor invoke_builder  -> fresh FocusedBuilder -> actor
  single:        actor owns world_model.py; no builder
  no_world_model: no executable model and no builder

Every builder receives the same Tycho workspace, has independent bounded
history, and cannot call take_action.
```

There is one persistent NOOA agent per game and one growing NOOA event history.
The workspace and tool executor are imported from pinned Tycho rather than
adapted into an experiment folder. The model sees ordinary native tools named
`ls`, `read_file`, `write_file`, `edit_file`, `run_python`,
`set_verbosity`, and `take_action`. `take_action` returns the sole decision for
the turn; the next observation is passed to the same agent instance.

Except in `no_world_model`, Tycho seeds `world_model.py`, `verify.py`, and
`plan.py` in the workspace. The actor can revise the executable model only in
`single`; writes are mechanically rejected in the builder-owned modes. Each
mode receives its pinned Tycho world-model paragraph. A successful planner call
writes `notes/validated_plan.json`; subsequent observations expose Tycho's
guarded continuation only while the model hash, observed trajectory, and action
availability still agree. The actor continues to commit one action at a time.

### Sterile CodeAct execution vertical slice

`tycho_nooa.sterile_execution` is an offline capability build for a future
CodeAct treatment. It does **not** replace `NativeToolStrategy` or participate
in live decisions. One `SterileExecutor` starts a sanitized `python -I -S`
process for one prospective CodeAct invocation, keeps REPL variables only in
that child across cells, and validates `return_result(...)` against T1's
Pydantic return type. The child receives JSON values rather than a live Agent:
there is no `self`, canonical workspace descriptor, credential environment, or
Python-object transport across the boundary.

The fixed OS policy gives the child one disposable writable scratch directory,
read-only standard-library access, no sockets or subprocesses, bounded output,
and CPU/memory/file/descriptor limits. Landlock, seccomp, and rlimit setup must
all succeed before the worker reports ready.

The first read-only capability is `evidence.query`. T1 translates the
authoritative current-attempt frames or ordinary transitions into paginated,
frontier-bound JSON DTOs; workspace paths and NumPy objects never cross. T2
caches the serialized page for one invocation and returns a fresh value copy on
each identical query, so generated-code mutation cannot poison the cache or the
workspace. Archived attempts, death/terminal/animation channels and CodeAct
Actor integration remain later slices.

Frozen-workspace parity is executable through
`python -m tycho_nooa.evidence_parity`. The retained 2026-08-23 treatment in
`analysis/codeact-evidence-parity/20260823/` compares legacy workspace-local
`wmlib.py` values with values retrieved and analyzed inside a real sterile
worker. It covers the no-transition edge, an 18-transition CLICK trajectory,
and multi-page synthetic retrieval; it does not claim parity for evidence
channels that `evidence.query` does not yet expose.

T1 now records each offline CodeAct invocation in an append-only provenance
log. Core-assigned invocation, cell, event, and capability-request identities
describe dispatch, capability mediation, execution outcome, terminal proposal,
and invocation completion or failure. Payloads remain exact in that raw JSONL
record and carry deterministic hashes; the disposable worker cannot choose
event types, ancestry, or accepted-return status. Timeout and worker loss
explicitly terminate the invocation with `locals_lost=true` rather than
silently starting a new REPL.

`NooaExecutionProjection` derives a separate model-visible `EventManager` from
the raw record. Cell source, stdout, stderr, exceptions, and capability data are
escaped before NOOA's XML-like renderer sees them, so text resembling a system
or tool boundary stays data inside its projected event. NOOA collapse may
replace a visible span with a summary without modifying the raw log, and the
full projection can be reconstructed from that log. This remains offline
infrastructure: no live Actor/provider context uses the projection yet, so the
tests establish deterministic reconstruction of the context renderer's output,
not parity with an already-deployed CodeAct Actor.

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
  reasoning), cost, tool names, returned prose, and returned reasoning. Live
  research runs always request provider reasoning and retain the response text.
  `TYCHO_NOOA_REQUEST_REASONING=0` and
  `TYCHO_NOOA_CAPTURE_RESPONSE_TEXT=0` are rejected rather than silently
  producing an evidence-poor run. A provider may still return no exposed
  reasoning; `reasoning_tokens=0` plus an empty `reasoning` field records that
  provider limitation rather than manufacturing an explanation.
- `decision-checkpoints/`: one ordered actor/builder directory per model call,
  containing the exact credential-free request at NOOA's `llm_call` boundary,
  its SHA-256 commitment, the standardized response, and a pre-call workspace
  manifest. Workspace bytes live in a shared content-addressed blob store, so
  unchanged files are not copied hundreds of times. Each request also records
  its host sequence, level, turn, and whether the call occurred while deciding
  or after the action result; offline packet mining only admits decision-phase
  calls into a pre-decision packet.

Decision checkpoints support matched shadow requests and historical workspace
restoration. They deliberately claim identity only at NOOA's model-call
boundary: provider formatters subsequently translate that request into Chat
Completions or Responses wire syntax. Restore into a new directory with:

```python
from pathlib import Path

from tycho_nooa.decision_checkpoints import restore_workspace

restore_workspace(
    Path("run/decision-checkpoints"),
    call_index=1,
    destination=Path("restored/call-000001"),
)
```

### Frozen decision-point shadows

`tycho_nooa.shadow_eval` compares an exact checkpoint request with one
precommitted message-content perturbation. It never executes a returned tool or
contacts the game, so it measures only immediate response dependence.

First inspect the checkpoint without spending model budget:

```bash
(cd solvers/tycho_nooa && uv run --frozen python -m tycho_nooa.shadow_eval \
  inspect ../../runs/<run-id>/decision-checkpoints --call-index 1)
```

The output gives the source request commitment and JSON pointers, exact values,
and commitments for every string under `messages`. Use them to write a
perturbation before sampling either arm:

```json
{
  "schema": "tycho-nooa-shadow-perturbation-v1",
  "id": "remove-builder-report",
  "question": "Does the immediate actor choice depend on this builder report?",
  "hypothesis": "Removing the report changes the first take_action call.",
  "expected_effect": "take_action",
  "invalid_if": "Any request leaf besides the declared message changes.",
  "source_request_payload_commitment": "sha256:<from inspect>",
  "operation": {
    "type": "replace_exact_text",
    "pointer": "/messages/17/content",
    "target_value_commitment": "sha256:<from inspect>",
    "old": "<exact report text including delimiters>",
    "new": ""
  }
}
```

Prepare a self-contained packet. This rechecks the source request/workspace
commitments, copies referenced workspace blobs, applies the treatment, and
proves that exactly one message leaf changed:

```bash
(cd solvers/tycho_nooa && uv run --frozen python -m tycho_nooa.shadow_eval \
  prepare ../../runs/<run-id>/decision-checkpoints \
  ../../analysis/shadows/<shadow-id>/packet \
  --call-index 1 --perturbation ../../analysis/shadows/<shadow-id>/perturbation.json)
```

`run` without `--execute` prints the plan and refuses model calls. Paid
execution requires both a call cap and a cumulative observed-cost limit. The
per-call reserve must fit before a request starts; two repeats per arm require
exactly four logical calls, with randomized arm order and provider retries
disabled:

```bash
# Zero-cost preflight; exits 2 because --execute is absent.
(cd solvers/tycho_nooa && uv run --frozen python -m tycho_nooa.shadow_eval \
  run ../../analysis/shadows/<shadow-id>/packet \
  ../../analysis/shadows/<shadow-id>/result --repeats 2 --max-calls 4 \
  --max-cost-usd 0.05 --cost-reserve-per-call-usd 0.0125)

# Model-only execution after reviewing the packet and plan.
(cd solvers/tycho_nooa && uv run --frozen python -m tycho_nooa.shadow_eval \
  run ../../analysis/shadows/<shadow-id>/packet \
  ../../analysis/shadows/<shadow-id>/result --repeats 2 --max-calls 4 \
  --max-cost-usd 0.05 --cost-reserve-per-call-usd 0.0125 --execute)
```

The evaluator verifies the model, client class, credential-free provider
configuration, cache settings, and original request bytes before the first
call. The reserve is a client-side stop rule, not a provider-side billing
guarantee: one already-started response can cost more than its reserve. In that
case the evaluator records `observed_cost_exceeded` and withholds every later
call; missing response cost records also stop execution. Every exposed
non-secret provider-route field is retained. Route drift makes the comparison
`inconclusive`; absent route metadata is reported as unverifiable rather than
invented. The evaluator journals every call immediately and reports
`supported` only when both arms are complete and internally stable and their
precommitted effect signatures differ; stable equality is `contradicted`, while
single pairs, provider errors, route drift, cost stops, or within-arm
variability are `inconclusive`. Even `supported` permits only a short
closed-loop promotion. It is not evidence of gameplay benefit or transfer.

### P1 matched warm restart

`tycho_nooa.matched_replay` freezes one legacy run's durable workspace and last
public transition, checks content commitments before every arm, and executes
exactly one fresh-actor decision under three named policies:

```text
trigger / control
orchestrator / control
orchestrator / typed_candidate_goals
```

For the executable-theory publication mechanism test, the same packet and
executor accept the explicitly frozen `THEORY_TRANSACTION_ARMS` contrast:

```text
builder_control       / trigger / control
builder_transactional / trigger / transactional_builder
```

Pass that arm tuple to both `prepare_legacy_run_packet(..., arms=...)` and
`execute_matched_replay(..., arms=...)`. The per-arm result records the final
theory publication artifact and ledger events in addition to the shared Builder,
actor, planner, verifier, and workspace measurements. Provider clients and
enforce-before-call cost limits remain caller-owned.

The third arm is explicitly a local treatment. It gives the actor typed,
durable `CandidateGoal` records with evidence references, a falsifier, and a
status, then lets `request_builder(focus="objective", ...)` carry that hypothesis
to the same actor-pulled builder. A proposal mechanically blocks `take_action`
until that request returns a durable structured review and the actor explicitly
selects, rejects, or retains the candidate with `update_goal`. A Builder request
and its resolution cannot share one tool batch, so the actor must first receive
the review. Objective review reports preserve the assessment, reason, proposed
route, recommended action, cited evidence, and the full Builder report under
`epistemic/`. The ledger never promotes a hypothesis into canonical `outcome()`;
objective-review passes are mechanically prevented from editing `world_model.py`.
In this treatment the untyped `invoke_builder` tool is replaced, not supplemented,
and an objective request requires a candidate-goal ID so evidence validation
cannot be bypassed.

A packet prepared from a legacy run is a **matched fresh-actor warm restart**,
not an exact continuation: runs that predate decision checkpoints do not retain
the original NOOA event history or model-call request. The packet manifest
records this claim ceiling. `execute_matched_replay` accepts injected LLM clients
but intentionally owns no provider or spending policy; callers must authorize
and cap live calls separately.

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
- a mechanically enforced Tycho `ModePolicy`: `trigger` and `no_world_model`
  actors cannot edit `world_model.py`, while `single` receives the exact source
  prompt, function-editing tool, workspace seeding, verifier, and write ownership;
- Tycho's exact `orchestrator` actor prompt and `invoke_builder(reason)` schema,
  with every request creating a fresh bounded builder and returning its advisory
  report before the actor can commit an action;
- a separately named `typed_candidate_goals` treatment with durable typed goal
  hypotheses, authoritative evidence-reference validation, mandatory
  objective-review acknowledgement before action, and focused builder requests
  that do not silently alter the exact orchestrator control; the treatment uses
  a ten-step objective-review loop, structured assessment/route results, and a
  fresh protected Builder slice while the exact control keeps its 40-step and
  ordinary deadline policy;
- a content-committed three-arm legacy warm-restart harness that measures goal
  events, builder exposure/result/uptake, verifier output, planner invocation,
  world-model change, and the unexecuted action at one matched decision point;
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
- content-addressed, byte-restorable decision checkpoints for every actor and
  builder model call;
- capped, one-factor frozen decision-point shadow packets and immediate-effect
  evaluation;
- durable files readable on a later turn;
- one legal action as the end-of-turn result;
- protocol-visible decision deadlines: the pre-actor builder is capped at 240
  seconds and yields early enough to reserve 300 seconds for the actor plus 5
  seconds for returning the decision to the host; a typed objective request gets
  a fresh slice of up to 240 seconds while preserving 120 seconds for post-review
  Actor resolution plus the same 5-second host-return reserve;
- an opt-in `BuilderConfig(show_budget_telemetry=True)` treatment derived from that
  host-authoritative monotonic deadline before every repair call: whole seconds,
  remaining repair calls, finishing-call reserve, and investigate/commit mode. It
  is intentionally off by default: the first matched reasoning-on replay induced
  an edit but degraded the retained model from 8/16 to 0/16 after the Builder
  ignored negative verifier feedback instead of rolling back;
- one growing tool/event history across turns;
- protocol-v3 successor acknowledgement carrying the settled decision frame,
  exact terminal frame, typed transient frames, and selection provenance;
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
- Tycho's reflection cadence and context compaction (the scribe and the
  LEVEL_START trigger landed in `712f892`);
- live use of the now-mandatory `request_builder` review loop and promotion of a
  selected candidate into canonical `outcome()` remain unobserved. In the earlier
  ten-action Gemma 4 31B run ending
  `20260821T135858Z`, the legacy builder tool was absent at every model boundary
  and the actor proposed one falsifiable candidate from six authoritative frames,
  but did not select it or request construction before the cap; the current
  treatment would reject that premature environment action rather than silently
  leave the proposal inert;
- replication across another seed or game.

Those omissions are explicit. The two live runs replicate this one seed's
gameplay success and symmetry uptake; they do not establish a general capability.

## Test

The nested runtime targets Python 3.12 because NOOA is `<3.14`:

```bash
(cd solvers/tycho_nooa && TYCHO_SANDBOX_RUNTIME=host uv run --frozen pytest -s)
```

`-s` avoids a WSL/Windows temporary-file capture failure in this workspace; it
does not change the tests.

`TYCHO_SANDBOX_RUNTIME=host` is required wherever no container runtime is
reachable. Without it the sandbox refuses to start, the Builder never runs, and
the deadline tests that stall a scripted Builder call instead stall the Actor --
which is deliberately unbounded, because it owns the protected remainder of the
decision and only the host wall clock ends it. The suite then hangs rather than
failing, so the missing runtime does not name itself.

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
(cd solvers/tycho_nooa && uv run --frozen python tools/crap.py --fail-above 9.999999)
```

The gate fails only for a raw score strictly greater than the threshold. Exit
code `1` means a CRAP violation, `2` means the analysis could not be trusted,
and `3` means pytest failed. Arguments after `--` replace the default
`pytest -s` selection.

The solver standard is **strictly below CRAP 10**. The gate uses `9.999999`
because `--fail-above` is intentionally strict and would otherwise allow a raw
score equal to 10. CRAP identifies pressure rather than prescribing a mechanical
refactor: compare complexity with the threshold first, delete or split mixed
decisions when complexity alone makes the target impossible, and add coverage
only when the test constrains meaningful behavior.

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

`--worker-timeout` is transmitted to the worker on every turn; it is not a
duplicated solver-side guess. With the 600-second setting used in the diagnosed
Gemma run, the builder receives at most 240 seconds, leaving 360 seconds before
the host timeout. If the configured timeout cannot cover the 300-second actor
reserve and 5-second return reserve, the advisory builder is skipped. A builder
timeout is recorded in `trigger-builder-trace.jsonl`, every partial workspace
change is rolled back, and an actor-requested tool receives an explicit error
rather than a false “skipped” result. Partial calls and bytes remain recoverable
from the tool trace and content-addressed decision checkpoints.
The effective value is also retained as `worker_timeout_seconds` in the run's
`summary.json`, including failed runs.
