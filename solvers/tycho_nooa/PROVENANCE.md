# Provenance

## NOOA

- repository: `https://github.com/pranavagrawaI/labs-OO-Agents.git`
- revision: `55a4b57872b49bfe9698cda6fc8e1f3b0bdd723e`
- reused interface: `Agent`, `GenerationStrategy`, `RuntimeServices`, event
  history, native `Tool` schema, `get_llm_client`, and `FakeLLMClient`
- adaptation: `NativeToolStrategy` is new integration code. It uses NOOA's
  public strategy/runtime interfaces but replaces CodeAct's two-tool surface
  with Tycho's workspace tools.
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
  triggers (divergence, level start, GAME_OVER reset), but this slice's
  `TriggerBuilderLoop.before_actor` (`trigger_builder.py`) only implements
  divergence — no level-start or GAME_OVER-reset firing exists. Left
  byte-exact, that sentence would promise the actor builder refreshes that
  never happen. `tycho_nooa.planner.trigger_actor_world_model_paragraph`
  rewrites that one sentence to name only the implemented trigger and raises
  loudly if the upstream sentence changes. Everything else in the paragraph
  stays byte-exact. Implementing the missing lifecycle triggers is a separate,
  not-yet-scoped change.
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
  same-level transition. No level-start or GAME_OVER-reset trigger was added in
  this slice.
- builder reuse: the system and kickoff messages are rendered directly from
  pinned `tycho/prompts/builder.system.j2` and `builder.user.j2`; the
  auto-attached verification text comes directly from pinned
  `tycho.agent.builder._verify_state`; and the mandatory report path is pinned
  `tycho.agent.builder.REPORT_FILE`.
- semantic parity: the tool names and descriptions come from Tycho's
  `tool_specs("generic")`; the actor receives Tycho's default generic surface,
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

Explicitly excluded from this integration slice: Tycho's scribe, reflection
cadence, context compaction, level-start/GAME_OVER trigger paths, and any new
planner policy.

No source is imported from a numbered ARC experiment.

## ARC host protocol

- protocol owner: `/arc-host`, version 1
- solver-side mirror: `tycho_nooa.protocol`, required because the host runtime
  is Python 3.14 while NOOA is Python 3.12
- parity evidence: strict payload tests plus
  `integration/smoke_host_worker.py`, which crosses both interpreters and the
  real subprocess transport
- test-only scripted model activation is explicit through
  `TYCHO_NOOA_SCRIPTED_TOOL_CALLS` and reports `client_kind=scripted`; it is not
  a live gameplay result.
