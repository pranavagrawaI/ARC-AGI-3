# Provenance

## NOOA

- repository: `https://github.com/pranavagrawaI/labs-OO-Agents.git`
- revision: `55a4b57872b49bfe9698cda6fc8e1f3b0bdd723e`
- reused interface: `Agent`, `GenerationStrategy`, `RuntimeServices`, event
  history, native `Tool` schema, `get_llm_client`, and `FakeLLMClient`
- adaptation: `NativeToolStrategy` is new integration code. It uses NOOA's
  public strategy/runtime interfaces but replaces CodeAct's two-tool surface
  with Tycho's workspace tools.
- port: `_sterile_guards.py` takes NOOA's pinned raw-`ctypes` Landlock,
  seccomp, and rlimit mechanisms from `nooa/runtime/sandbox/guards.py`, reduced
  to one fixed exec-worker policy. The adaptation adds file-size/open-file/
  process limits and deliberately omits NOOA's live-agent `self` broker.
- adaptation: `evidence.py` reads Tycho's pinned `wmlib.frames()`,
  `wmlib.transitions()`, and `wmlib.current_frame_key()` in T1, then replaces
  their filesystem roots and NumPy arrays with bounded frozen DTOs before any
  value crosses into the sterile worker.
- adaptation: `execution_events.py` keeps the exact T1-authored sterile
  execution lifecycle in append-only JSONL, then derives a smaller NOOA
  `EventManager` projection for model context. It reuses pinned NOOA's event
  registration, `EventManager.collapse`, context phase, XML block formatter,
  and OpenAI provider formatter. Escaping all T2-controlled text before that
  rendering is local boundary code; raw payloads remain unescaped in the
  provenance log. The projection is not yet wired into a live Actor strategy.
- adaptation: `sterile_codeact.py` preserves NOOA CodeAct's model-visible
  `execute_python`/`return_result` choice while rebuilding every provider
  context from the T1-authored projection and sending cell execution through
  `SterileExecutor`. It is an offline cognitive-parity controller, not the live
  `TychoNooaAgent` strategy and not a port of NOOA's ambient `self` namespace.
- adaptation: `FocusedBuilder` is a separate NOOA `Agent` created fresh for
  each fired trigger. It deliberately shares the actor's LLM client and Tycho
  executor/workspace while retaining an independent bounded NOOA event history.

## Tycho

- repository: `https://github.com/NIMI-research/Tycho.git`
- revision: `f68912a764372ead0a610db2e1c011d41ce5197e`
- reused directly: `tycho.workspace.workspace.GameWorkspace`,
  `tycho.workspace.agent_tools.ToolExecutor`, `tool_specs`, `canonical`,
  `GameWorkspace.validated_plan_hint`, and the actor's world-model paragraph
  rendered from `tycho/prompts/partials/wm_trigger.j2` plus the
  `wm_variant == 'trigger'` `notes/actor_beliefs.md` block of
  `tycho/prompts/actor.system.j2` — matching real Tycho's `trigger` mode
  (`tycho.agent.modes`), where the actor never owns `world_model.py` and is
  never taught Planner mechanics/`plan.py`; only the builder is (its own
  `builder.system.j2` render already carries that). The actor still passively
  receives `validated_plan_hint` output whenever the builder produces a
  validated plan artifact, unchanged.
- documented deviation: `wm_trigger.j2`'s builder-fires sentence names three
  triggers (divergence, level start, GAME_OVER reset). `TriggerBuilderLoop
  .before_actor` (`trigger_builder.py`) implements two of the three:
  divergence (`TychoTransitionTrigger.evaluate`, unchanged) and level start
  (`TychoTransitionTrigger.evaluate_level_start`, new — fires unconditionally
  on a genuine level advance, mirroring Tycho's own
  `elif ... turn_in_level == 0 and level > 0` branch, which never calls
  `should_fire()` for this cause). GAME_OVER-reset is excluded, and not merely
  deferred: arc-host's episode loop (`arc-host/src/arc_host/runner.py`)
  treats `GAME_OVER` as a member of `TERMINAL_STATES` and breaks the loop the
  instant it is observed — `solver.decide()` is never called again in that
  episode, so there is no live turn on which a GAME_OVER-reset trigger could
  ever fire against this deployment. `tycho_nooa.planner
  .trigger_actor_world_model_paragraph` rewrites the builder-fires sentence to
  name only the two implemented triggers and raises loudly if the upstream
  sentence changes. Everything else in the paragraph stays byte-exact.
- scribe reuse: pinned `tycho/agent/agent.py::_build_agent_graph` registers
  the scribe as `AgentSpec(id="scribe", trigger=on_events(Event
  .LEVEL_COMPLETE), sees_full_history=True, run=lambda ctx:
  self._summarize_level())` — unlike the builder (`sees_full_history=False`,
  a fresh transcript per fire), the scribe reuses the actor's own growing
  conversation. NOOA's `Agent` owns exactly one `event_manager` for its whole
  lifetime, so `tycho_nooa.scribe.ScribeStrategy` is a second
  `@strategy`-decorated method directly on `TychoNooaAgent`
  (`TychoNooaAgent._summarize_level`), not a separate `FocusedBuilder`-style
  fresh `Agent`. It is invoked from `TychoNooaAgent.observe`
  (`agent.py`), on the same two conditions pinned `Event.LEVEL_COMPLETE`
  covers — `levels_completed` advancing, and terminal `WIN` — deliberately
  before `GameWorkspace.new_level()` runs, so the scribe still sees the
  just-finished level's own directory as current; this is a separate call
  site from the LEVEL_START trigger above, which fires from
  `TriggerBuilderLoop.before_actor` on the following turn once `turn.level`
  itself has advanced. `take_action`/`submit_action` are
  withheld from the scribe's offered tools rather than offered-then-
  intercepted (Tycho's own approach). `_is_world_model_edit` and
  `_write_level_insight_fallback` are byte-exact ports of
  `agent.py`'s equivalents, adapted only for tycho_nooa's always-
  `actor_writes_wm=False` mode (so the builder-owned check needs no mode
  branch) and for a pre-parsed argument dict rather than Tycho's
  `{"input": {...}}` call shape. See `tests/test_scribe.py`.
- prompt-assembly decision: Tycho's own actor.system.j2 states the
  `world_model.py` ownership boundary, the `notes/actor_beliefs.md` channel,
  and "report is advisory" exactly once. To stay comparable — this slice is
  specifically testing whether the model writes `actor_beliefs.md` — the NOOA
  framing in `agent.py`'s docstring and `strategy.py`'s `strategy_prompt`
  preamble deliberately says nothing about world-model ownership, beliefs, or
  report semantics; that framing is generic (persistence, tools, the
  one-action boundary) and the Tycho paragraph is the single source of that
  instruction.
- trigger reuse: `tycho.agent.dispatcher.TriggerDispatcher.should_fire` invokes
  the pinned `tycho.agent.wm_signal.wm_signal` unchanged after each recorded
  same-level transition. Level-start firing is added separately (see
  "documented deviation" above); GAME_OVER-reset firing is not, for the
  structural-unreachability reason given there.
- builder reuse: the system and kickoff messages are rendered directly from
  pinned `tycho/prompts/builder.system.j2` and `builder.user.j2` — the system
  message with `runtime_notice.prompt_brief(...)` appended, because that template
  enumerates run_python's preloads as a closed list written before `symmetry`
  existed, so leaving it alone would have told the builder the module is not
  there; the
  auto-attached verification text comes directly from pinned
  `tycho.agent.builder._verify_state`; and the mandatory report path is pinned
  `tycho.agent.builder.REPORT_FILE`.
- semantic parity: the tool names and descriptions come from Tycho's
  `tool_specs("generic")`, with `runtime_notice.tool_note(...)` appended to
  `run_python` only; the actor receives Tycho's default generic surface,
  while the builder uses Tycho's builder flags (`include_edit_func=True`,
  `include_take_action=False`, `include_set_verbosity=False`). Non-action
  execution is delegated to Tycho's `ToolExecutor` without copying it.
- adaptation: the NOOA strategy owns tool-call/event orchestration and converts
  `take_action` into the solver module's typed `ActionDecision`.
- adaptation: `tycho_nooa.planner.PlannerIntegration` injects Tycho's guarded
  plan continuation into the NOOA observation and records planner invocations,
  newly written validated artifacts, continuations, and invalidations in
  `planner-trace.jsonl`. It does not reinterpret or execute plan actions.
- instrumentation adaptation: planner-call recognition is local integration
  code. It recognizes Tycho's direct command plus bounded Python standard-library
  wrappers and uses Tycho's emitted `PLANNER_COMMAND` marker as a trace fallback;
  it does not alter `plan.py` or its validation result.
- action-vocabulary adaptation: the pinned prompt's `ACTION6` is rendered as
  host-facing `CLICK`, and guarded CLICK hints recover `row`/`col` from the
  validated artifact because Tycho's formatter only expands `ACTION6`.
- instrumentation adaptation: `trigger-builder-trace.jsonl` separates the
  mechanical trigger, builder artifact hashes, and observable actor uptake.
  "followed" means the report's structured `recommended_action` exactly equals
  the committed action; "tested" means the actor read a model/report artifact
  or invoked model verification/planning. These are syntactic observations,
  not claims about hidden reasoning.
- bugfix: `BuilderResult.report_fresh` previously read `forced or
  hashes[REPORT_FILE]["changed"]`, so any pass that reached the forced
  tool-free final-report request was marked fresh even when the forced text
  was byte-identical to the report already on disk. `runs/tycho-nooa-tb-
  first-live-20260818T103742Z/DIAGNOSIS.md` observed exactly this on turns 11
  and 19: 12 consecutive forced passes over the same stale turn-1 scratchpad,
  two of which relabeled an unchanged report as fresh. `report_fresh` now
  reflects only `hashes[REPORT_FILE]["changed"]`; `report_forced` (which code
  path ran) is traced separately and no longer feeds the freshness claim. See
  `tests/test_trigger_builder.py::test_forced_report_identical_to_prior_report_is_not_fresh`.
- bugfix: `BuilderStrategy.execute`'s forced final-report call previously
  passed `tools=[], tool_choice="none"`, structurally disallowing the model
  from attempting a tool call. Pinned `tycho/agent/builder.py`'s equivalent
  fallback call (`builder_chat()` at the end of `WorldModelBuilder.build()`)
  still passes `self.tools_spec` on that call — it is never hard-blocked at
  the API level — but its handling of the reply only reads `reply["text"]`
  and never inspects `reply["tool_calls"]`, so an attempted tool call there
  goes unexecuted rather than disallowed. The forced call now passes
  `tools=tools, tool_choice="auto"` to match, while continuing to read only
  `response.content`, so a model that ignores the "do NOT call any more
  tools" instruction and attempts one anyway still falls through to
  `FALLBACK_REPORT` instead of having that tool call run. See
  `tests/test_trigger_builder.py::test_forced_report_call_offers_tools_but_ignores_attempted_tool_calls`.
- bugfix: the same `DIAGNOSIS.md` run found `notes/world_model.md` read on all
  12 divergence-triggered builder passes after turn 8 but never once advanced
  — every later pass re-read the same stale turn-1 scratchpad, so the durable
  handoff between builder passes was silently broken even though the report
  hand-off to the actor kept working. `BuilderStrategy.execute` now runs
  `_enforce_world_model_checkpoint` after the main tool loop and before the
  forced-report call: it prompts the model once to checkpoint
  `notes/world_model.md` via `write_file`/`edit_file`, and — unlike the
  forced-report call — actually executes any tool call the model attempts
  there, since a real edit is the point. Only if the file is still
  byte-identical afterward does it force a deterministic breadcrumb
  (`_fallback_world_model_checkpoint`) so the file changes unconditionally.
  `BuilderResult.world_model_checkpoint_forced` traces which path ran. See
  `tests/test_trigger_builder.py::test_world_model_checkpoint_is_forced_when_pass_leaves_it_stale`
  and
  `tests/test_trigger_builder.py::test_world_model_checkpoint_executes_a_genuine_model_edit`.
- adaptation: pinned Tycho has no forcing mechanism for `notes/actor_beliefs.md`
  — it is a voluntary actor scratchpad that the builder treats as a disposable
  starting hypothesis, not ground truth (`tycho/agent/builder.py`'s
  `WorldModelBuilder` reads it once as a prompt seed and never checks whether
  it advanced), so a stale copy costs Tycho's own design nothing. This is a
  deliberate customization beyond Tycho's fidelity target, not a bugfix:
  `DIAGNOSIS.md`'s first live run found `notes/actor_beliefs.md` written once
  at turn 1 and never refreshed despite two later probing bursts, and our goal
  — reliable level solving — benefits from the builder seeing the actor's
  current thinking rather than its turn-1 draft. The model can update the file
  during its normal pre-action tool loop. After an action is committed,
  `NativeToolStrategy.execute` performs no provider call: if the file remains
  byte-identical it appends a deterministic, provenance-marked breadcrumb that
  records the committed action (`_fallback_actor_beliefs_checkpoint`). This
  replaces the earlier mandatory model-authored post-action checkpoint. In
  `runs/tycho-nooa-gemma4-31b-reasoning-capture-20260821T081012Z`, that extra
  call withheld an accepted `DOWN` for 29.9 seconds and crossed the host's
  600-second deadline by 2.3 seconds. See
  `tests/test_native_flow.py::test_actor_beliefs_checkpoint_is_forced_when_turn_leaves_it_stale`,
  `tests/test_native_flow.py::test_actor_beliefs_written_before_action_is_preserved_without_fallback`,
  and `tests/test_native_flow.py::test_accepted_action_does_not_wait_for_post_action_llm_call`.

- adaptation: `TriggerBuilderLoop.before_actor` catches any exception raised by
  the divergence probe (`TychoTransitionTrigger.evaluate` ->
  `TriggerDispatcher.should_fire` -> `wm_signal`), traces it as `trigger_error`,
  and treats the turn as unsignaled instead of letting it escape `decide()`.
  Pinned Tycho does not guard this call. The asymmetry that motivates the guard
  is deployment-specific: `wm_signal` executes `verify.py` in Tycho's sandbox,
  and when no container runtime exists the resulting `SandboxError` is only a
  tool *result* on the actor's path (the actor reads it and works around it)
  while being fatal on the trigger's path, where it surfaces as an arc-host
  `ProtocolError` that ends the episode. `runs/tycho-nooa-level-start-ar25-
  20260818T174559Z/ABORT.md` records a live run lost this way at action 1,
  before any trigger, builder, or planner trace existed. The builder's own
  `except Exception` still re-raises, deliberately: a probe that cannot answer
  is absence of signal, but a builder that cannot run Python would produce a
  world model that only looks like evidence. See
  `tests/test_trigger_builder.py::test_unavailable_sandbox_degrades_the_trigger_instead_of_ending_the_episode`.

- adaptation: `tycho_nooa.wm_probe.GatedToolExecutor` splices a goal
  satisfiability gate into pinned Tycho's `wm_feedback_probe.py.tmpl` before
  running it, and `tycho_nooa.probe_gate` is the spliced code. Upstream's probe
  guards the planner with four preconditions — the model imports, `plan.py`
  exists, `actions()` returns something, `heuristic()` is finite — and none of
  them asks whether a goal state exists, so a `world_model.py` whose `outcome()`
  is still the seed's unconditional `return "ongoing"` reaches full A*/BFS.
  `runs/tycho-nooa-level-start-ar25-20260818T175557Z` records six such searches
  across 60 actions (5x `no_plan_within_budget`, 1x `planner_timeout`), each
  costing ~1.8s and ~8500 expanded states to produce upstream's unranked
  six-way `interpretation` disjunction — "wrong outcome(), incomplete
  actions(), unseen or unencoded target/route/off-screen state, game_over
  current state, insufficient budget, or planner/heuristic weakness" — whose
  first disjunct was the answer and was statically decidable in ~40ms. The splice asserts both anchors
  (`_PLAN_METHOD` and the unconditional `if planner_ready:` entry) and raises if
  either moves, refuses a gate whose top-level names would shadow a template
  helper, and re-parses the result; the gate itself fails open, so any error in
  it costs the search nothing. Two things are printed: `PLANNER_STATUS:
  not_run_goal_unsatisfiable` in place of a search that provably cannot reach a
  goal, and a `PLANNER_CONTRACT:` audit that classifies each `actions()` entry
  by what it does to `wmlib._planner_key` — `no_op`, `restart`, `retrograde`,
  `retrograde_partial`, `forward` — plus the bounded closure statistic for all
  declared operators and for the forward ones alone. The audit is reported, never
  enforced: an unsound key can make the planner fast while making it wrong, so
  naming the state field that blocks closure is the whole diagnosis and the
  builder decides.

  The goal verdict is now tri-valued:
  `proven_unsatisfiable | unknown | candidate_present`. Only a syntactically
  restricted unconditional non-goal return is permitted to block planning.
  A goal literal, a sampled goal result, or a constant goal return establishes
  only `candidate_present`; literal absence and non-goal samples remain
  diagnostics and produce `unknown`. This deliberately narrows Gate 0 from a
  heuristic rejection into a sound special-case proof. See
  `tests/test_probe_gate.py`, including the dynamically constructed
  `"level_" + "complete"` regression.

- adaptation: protocol v3 carries a complete public `EvidenceEnvelope` rather
  than only `frames[-1]`: the settled `decision_frame`, the exact terminal
  frame when the transition completes or ends a game, typed transient event
  frames, and explicit selection provenance. The actor still chooses from the
  settled frame, while Tycho's evidence store receives terminal and animation
  evidence before lifecycle mutation or scribing.

- adaptation: `NativeToolStrategy.execute` spends one final generate with
  `take_action` as the only offered tool before raising on an exhausted
  `max_tool_steps` budget. Pinned Tycho's loop has no equivalent because its
  turn boundary is not an ARC action. Exhausting the budget and refusing to act
  are different failures: `runs/tycho-nooa-goal-gate-ar25-20260819T120904Z`
  spent all 25 steps reading the grid with `run_python`, never called
  `take_action`, and lost the episode's other 19 actions to an arc-host
  `ProtocolError` at action 0. An ARC episode has no "no action" branch, so a
  silent turn throws the run either way; a model that still declines when
  `take_action` is the only tool left is a real refusal and still raises. Shape
  restricts the offered tools, then lets the failure stand. `tool_choice` stays
  `"auto"` because `"required"`
  draws 500s from some gateways — nooa's own `codeact.py` says so. See
  `tests/test_native_flow.py::test_a_spent_tool_budget_demands_the_action_instead_of_ending_the_episode`
  and
  `tests/test_native_flow.py::test_a_model_that_declines_the_demand_still_ends_the_episode`.

- port: `NativeToolStrategy` keeps the inlined grid on the most recent three
  observations and elides the rest to a breadcrumb, which is pinned Tycho's
  `agent.py::_truncate_old_grids` under its `TYCHO_GRID_KEEP=3` default
  (`agent/context_config.py`). Upstream's justification transfers unchanged —
  the frames stay on disk and the agent re-reads them by tool — and this solver
  has the same `read_file`/`run_python` affordance over the same
  `GameWorkspace`. The mechanism differs because the histories differ: upstream
  rewrites provider messages in place, so it must recognise a heavy payload by
  its rendered shape; here each observation carries its own breadcrumb form in
  event metadata (`history.py::ELIDED_OBSERVATION`), built at the turn it
  describes, so eviction is a swap and never a re-parse. The other two halves of upstream's
  context discipline — bounded reasoning (`TYCHO_REASONING_KEEP`) and prompt
  caching — are still absent; see the exclusion note below. See
  `tests/test_native_flow.py::test_only_the_last_three_observations_keep_their_inlined_grid`.

- correction, revised after the ar25 reasoning-channel failure: prose and
  reasoning take different retention paths. `history.py::retain_assistant_turn`
  keeps `LLMOutput` only when `response.content` is non-empty; a reasoning-only
  response removes that otherwise blank assistant message. The reasoning is
  replayed instead on the corresponding `ToolCallEvent` by
  `history.py::replayed_reasoning`, matching upstream's single assistant message
  with tool calls plus raw reasoning. This revision matters empirically:
  `runs/tycho-nooa-llm-trace-ar25-20260819T155638Z` returned all output on the
  reasoning channel. Retaining blank `LLMOutput` messages made the model read its
  own history as failed tool syntax and commit 0 actions. The current split keeps
  prose when present without manufacturing blank turns, while reasoning remains
  attached to the tool-call message. Old reasoning is still unbounded; upstream's
  `TYCHO_REASONING_KEEP=3` remains excluded below. See
  `history.py::retain_assistant_turn`, `history.py::replayed_reasoning`, and the
  history assertions in `tests/test_native_flow.py`.

- instrumentation: the model leg of a run is now recorded, and NOOA's
  context-limit defence is switched on. Four traces already existed and all four
  are written by our own code about our own actions, so a finished run could not
  say whether the model emitted any reasoning at all — the question the previous
  entry's bound depends on. NOOA already publishes the answer: `ActorRuntime`
  adds an `LLMComplete` (tokens incl. `reasoning_tokens`, cost, tool calls,
  `reasoning_content`) followed immediately by the `LLMOutput` carrying the same
  response's prose, and although `LLMComplete` is `Role.RUNTIME_EVENT` added with
  `record=False`, `EventManager.add` emits to handlers *before* the record check
  — so a subscriber sees every one without a client wrapper or a NOOA change.
  `llm_trace.py::LLMTrace` subscribes to both and pairs them (they are adjacent
  and unconditional in one return path), writing `llm-trace.jsonl`. It attaches
  to the actor's event manager and to each per-fire builder as the trigger loop
  creates it, since a builder is a separate agent with a separate stream.
  Counters and returned text are always written, and live calls always request
  provider reasoning. Explicit zero values for either control are rejected so a
  research run cannot silently discard its model-visible rationale. This is a
  protocol change from the historical runs below: asking can change the run
  (`unifiedllm.py`: `if reasoning := self.config.get("reasoning")`), not merely
  what we observe of it, so those older results remain evidence for their frozen
  configuration rather than reasoning-on controls. Separately,
  `ARC_LLM_CONTEXT_WINDOW` now reaches the
  client. Measured: `client.context_window` resolved to `None` for
  `openai/meta/muse-glimmer-30b`, and every layer of NOOA's defence is gated on
  it — the proactive budget in `_build_messages`, `_archive_on_context_error`
  (`if not stats or not ctx_window: return`) and `_compute_reduced_max_tokens`
  (returns `None`, so the retry re-raises). An overflow therefore propagated
  through `agent.act` to `worker.py`'s error message and ended the run. Note two
  things the knob does *not* buy: `render_context` evicts system blocks only
  (`events_dropped=0` is hardcoded), so the event pile is still never bounded
  proactively, and `TruncationConfig.max_event_tokens`/`min_preserved_events` are
  declared but consumed nowhere in this pin despite `interactive.py:204` claiming
  otherwise. See
  `tests/test_native_flow.py::test_the_llm_trace_records_the_counters_for_every_round_trip`
  and `tests/test_trigger_builder.py::test_the_llm_trace_covers_the_builder_as_well_as_the_actor`.

- mode reuse: Tycho defines four canonical actor modes in
  `tycho/configs/base.yaml` and enforces their behavior across harness
  construction, actor prompting, tool ownership, seeding, and builder
  activation. The hybrid imports that registry and centralizes those same
  dimensions in `ModePolicy`. All four pinned modes are executable. Actor
  writes to `world_model.py` are mechanically rejected in `trigger`,
  `orchestrator`, and `no_world_model`; `single` gives the actor the upstream
  single-mode prompt and the additional `edit_function` tool. The local audited
  mode names are frozen independently from the imported registry so a future
  upstream addition fails closed until its whole ownership contract is reviewed.
  `arc-host`'s worker allowlist carries `TYCHO_MODE` so a dropped setting cannot
  silently execute the wrong arm. See `tests/test_modes.py`.

- orchestrator reuse: the exact actor paragraph is rendered from pinned
  `tycho/prompts/partials/wm_orch.j2`, and the actor-facing
  `invoke_builder` schema preserves pinned `agent_tools.py`'s single optional
  `reason` string. The local native-tool strategy intercepts that call before
  ordinary synchronous dispatch, waits for the same fresh `FocusedBuilder`
  used by the trigger arm, publishes its report as the tool result, and prevents
  a builder request and `take_action` from sharing one tool-call batch. This is
  a runtime adaptation from Tycho's synchronous actor loop into NOOA's async
  strategy, not a new source policy. Harness-triggered firing remains disabled
  in orchestrator. Separate trace events record `actor_request`,
  `builder_result`, and subsequent `actor_uptake`. See
  `tests/test_trigger_builder.py::test_orchestrator_actor_pulls_a_fresh_builder_before_committing_action`.

- local treatment, not Tycho mode reuse: `TYCHO_NOOA_TREATMENT=
  typed_candidate_goals` is accepted only beside `TYCHO_MODE=orchestrator`.
  `epistemic.py` exposes Pydantic-validated native tools for proposing, updating,
  and listing `CandidateGoal` records and for requesting a builder with a typed
  focus. Each candidate carries a predicate, host-authoritative evidence
  references, a falsifier, status, revision metadata, and the objective-review ID
  the actor has acknowledged. The ledger validates references against the
  immutable evidence index and persists its own snapshot, candidate event stream,
  and structured Builder reviews under `epistemic/`; it has no method that writes
  `world_model.py`. The native strategy rejects `take_action` after a proposal
  until an objective request returns and the actor calls `update_goal`. It also
  rejects a request and resolution in the same tool batch, preserving a real
  model boundary at which the Builder result becomes visible. The treatment
  replaces the source control's untyped
  `invoke_builder` surface; objective-focused `request_builder` calls require a
  candidate ID and send that candidate as a hypothesis plus an explicit
  independent-inspection constraint. Objective passes must return assessment,
  reason, route, and recommended-action fields; missing fields force the compact
  report path and parse as `inconclusive`, never as support. They skip the
  mechanics scratchpad checkpoint and mechanically reject `world_model.py` edits,
  so a provisional hypothesis cannot contaminate canonical `outcome()`. Its
  ten-step Builder budget and fresh up-to-240-second slice (with 120 seconds
  preserved for post-review Actor work) are named live deadline adaptations; the
  exact orchestrator control retains 40 steps and its ordinary 300-second Actor
  reserve. Builder deadline or availability failures become durable `unavailable`
  reviews that the actor can acknowledge without falsely selecting the candidate.
  Canonical promotion after selection remains a separate, unimplemented edge.
  This is a model-visible epistemic adapter inspired by NOOA's typed object/method
  paradigm, not a claim of copying NOOA's memory or CodeAct system. See
  `tests/test_epistemic.py` and the treatment assertions in
  `tests/test_trigger_builder.py`.

- local evaluation harness: `matched_replay.py` packages the final public
  transition and durable workspace of a legacy run, commits the workspace tree,
  transition, and replay turn, and revalidates those bytes before every arm.
  The frozen arms are `trigger/control`, `orchestrator/control`, and
  `orchestrator/typed_candidate_goals`; every arm gets a fresh actor and exactly
  one decision, and no returned action is executed. This is deliberately named
  a matched fresh-actor warm restart. The legacy ar25 run predates model-call
  checkpoints and retained actor history, so exact continuation is impossible;
  the manifest records that lifetime limit as its claim ceiling. Injected LLM
  clients keep provider/cost authorization outside the harness. Scripted replay
  tests control flow and metric capture only, never objective-inference
  capability. See `tests/test_matched_replay.py`.

Added beside upstream, not into it: `symmetry_template.py` and
`dynamics_template.py` are ours while `wmlib_template.py` remains pinned Tycho.
`tycho_nooa.trusted_runtime` preserves those separate sources under one versioned runtime
contract. A fresh agent writes the exact helper
bytes plus `.trusted-runtime.json`; the manifest commits each module's provenance, SHA-256,
availability, read-only integrity, and exact-version restore policy. It also commits the
transaction evaluator's `host_planner_contract.py` and `theory_probe.py`, so the evaluator
constructs its fixed bundle by iterating the same manifest rather than maintaining a second
helper list. Runtime inspection has three explicit outcomes: a fresh or resumed current
declaration validates the complete byte-exact runtime including `dynamics.py`; an exact P0
declaration resumes its original byte-exact runtime without dynamics; and a historical
workspace without metadata remains `legacy/unknown`, makes no mathematics claim, and cannot
enter transactional publication. The historical P0 branch is declared and trusted for what
it contains, but is never silently upgraded to a different mathematical runtime.

The trusted files are denied at the Actor/Builder file gateway. `run_python` receives the
same declared files through read-only bind mounts (bubblewrap for the trusted-development
host runtime, per-file container mounts otherwise), closing the file-tool bypass without
making the rest of the workspace read-only. `GatedToolExecutor._PREAMBLE` appends the
symmetry and dynamics imports to upstream's; an exact declared runtime that predates dynamics
retains `dynamics = None` rather than being silently upgraded. Verified by
`tests/test_trusted_runtime.py` and the wiring and mathematical tests in
`tests/test_symmetry.py` and `tests/test_dynamics.py`.

Telling the model the mathematics exists is a separate problem from seeding it, and needed
three channels, all sourced from `runtime_notice.py` so they cannot drift: `tool_note(...)` on `run_python`,
and `prompt_brief(...)` appended to both system prompts — the actor's via
`TychoNooaAgent._resolve_system_prompt`, the builder's via `FocusedBuilder.__init__`. The
functions distinguish all three runtime outcomes: the complete current contract, a truthful
historical P0 notice with no dynamics preload, and an explicit legacy/unknown warning. The
builder channel is the load-bearing one: `builder.system.j2` lists run_python's preloads as
`grid`, `np`, `wmlib`, `frames`/`transitions` and nothing else, so an unamended prompt does not
merely omit `symmetry`, it denies it. A tool description is read when choosing a tool; a system
prompt is read when forming a hypothesis, which is where a comparison library is either reached
for or not.

The first live run then showed that being told is not the same as being able to call: two of its four reaches died on `canon(fig, point='full')`, because `point` took a group object while `colours`, in the same signature, took a string, and the brief printed `point=...` while printing `colours="fixed"` literally. `symmetry_template.group()` now coerces a name at every entry that takes a group, `colours` rejects anything outside `{'fixed', 'any'}` rather than silently permuting, and all three channels enumerate every admissible value. The rule this encodes: where the model's reasonable guess and the API disagree, the API is what changes.

The same run completed level 0 but exposed a second handoff defect. The builder
already read `notes/actor_beliefs.md`, `level_0/solved.json`, and verifier feedback
that named the missed terminal; the failure was not an absent read lane. A later
`REPAIR_FOCUS: current_level_dynamics` displaced the unresolved outcome mismatch,
and the builder spent its pass on level 1 without editing `outcome()`. The repair
records the host-observed levels-completed transition as
`level_<N>/level_complete.json`, injects that authoritative evidence into the
builder directly from the action-result callback, and keeps it as the latest
mandatory checkpoint until verification no longer reports the missed completion.
If that pass fails, the evidence remains pending for the level-start builder to
retry. The direct callback matters because a winning action can exhaust the action
budget, leaving no next actor turn on which a level-start repair could fire. Prose
alone cannot end that prefix. This prioritizes a known false negative without
letting the host invent the game's goal semantics.

The first credited replication attempt exposed a different ownership boundary:
`recommended_action: CLICK` is useful prose but not a valid `ActionDecision`
without coordinates. The advisory parser constructed it as if it were an actor
commitment, and Pydantic ended the episode before the actor could choose. The
parser now validates availability and coordinates without throwing, records a
`recommendation_parse_error`, and exposes no recommendation on malformed advice.
The actor lane, which still owns the environment action, continues normally.

The repaired ar25 seed-0 retry completed level 0 on its twentieth action and made
two successful `symmetry.canon` calls. Together with the earlier 19-action run,
this is a same-seed local replication of completion and symmetry uptake. The
routes were not identical and no second seed or game has yet replicated it.

Explicitly excluded from this integration slice: reflection cadence, emergency
context compaction, upstream's bounded reasoning (`TYCHO_REASONING_KEEP=3`) and
prompt caching (`TYCHO_PROMPT_CACHING=1`, `TYCHO_CACHE_ROLLING_EVERY=1`), the
GAME_OVER-reset trigger path (structurally unreachable under arc-host's
`TERMINAL_STATES` design — see "documented deviation" above), and any new
planner policy. The two context defaults are named here because they are
upstream defaults, not upstream options: omitting them is a divergence from the
pinned commit, not a smaller version of it.

No source is imported from a numbered ARC experiment.

## ARC host protocol

- protocol owner: `/arc-host`, version 2
- solver-side mirror: `tycho_nooa.protocol`, required because the host runtime
  is Python 3.14 while NOOA is Python 3.12
- version-2 addition: every `turn` carries the exact timeout enforced by
  `JsonlWorker`. `DecisionDeadlineConfig` caps the advisory pre-actor builder at
  240 seconds while reserving 300 seconds for the action-producing actor and 5
  seconds for host return. Exhaustion cancels that builder pass, records
  `builder_deadline_exceeded`, transactionally restores the pre-builder workspace,
  and continues to the actor; an actor-pulled request receives a typed tool error
  naming its completed partial calls and rollback, while a deadline with no
  builder allocation records `builder_deadline_skipped`. Decision checkpoints
  retain the cancelled call and partial workspace version. This is a local
  safety/cost adaptation, not a mechanism inherited from Tycho or NOOA. The
  allocation addresses the 453.2-second builder share preserved in
  `runs/tycho-nooa-gemma4-31b-reasoning-capture-20260821T081012Z/TIMEOUT-FORENSICS.md`.
  The host also preserves the effective value as `worker_timeout_seconds` in
  both successful and failed run summaries.
- parity evidence: strict payload tests plus
  `integration/smoke_host_worker.py`, which crosses both interpreters and the
  real subprocess transport
- test-only scripted model activation is explicit through
  `TYCHO_NOOA_SCRIPTED_TOOL_CALLS` and reports `client_kind=scripted`; it is not
  a live gameplay result.
