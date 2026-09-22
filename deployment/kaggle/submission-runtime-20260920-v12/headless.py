#!/usr/bin/env python3
"""Headless q6 MdArcSolverAgent for the Kaggle join. No TUI, no viewer exporters."""

from __future__ import annotations

import argparse
import asyncio
import os
import json
import sys
import resource

from pathlib import Path

CHASSIS = Path(__file__).resolve().parent / "chassis"
sys.path.insert(0, str(CHASSIS))

from state_delivery import StateDelivery

from arc_llm import build_llm  # noqa: E402
from contingent_controller import arm as contingent_arm, enabled as contingent_enabled  # noqa: E402
from epistemic_critic import ACTOR_TEMPO_CONTEXT  # noqa: E402
from solver_agent import MdArcSolverAgent  # noqa: E402
from solver_agent import hidden, strategy, CodeActStrategy, _ARC_CELL_CONFIG, RespondResult


class ProceduralAgent(MdArcSolverAgent):
    """Perform two evidence-driven decisions. Follow decision_workflow and end
    each decision with self.submit_actions. Reuse the example procedure while
    adapting its measurements and conclusions to the current observation."""

    @hidden
    @strategy(CodeActStrategy(config=_ARC_CELL_CONFIG))
    async def handle(self, notification: dict[str, list]) -> RespondResult:
        """Read self.latest_state(), record observable alternatives using
        self.set_experiment, and submit one action within two model calls.
        On the next host observation, measure self.changed_region() or
        self.last_transition(), update only what the evidence supports, and
        submit the next discriminating action within two model calls."""
        ...

from nooa.config.truncation_config import (  # noqa: E402
    CaptureConfig,
    FormatConfig,
    TruncationConfig,
)
from nooa.interactive import RespondReason  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--skill", default="interactive-game-solver")
    parser.add_argument("--visual", default="off")
    parser.add_argument("--png-scale", type=int, default=8)
    parser.add_argument("--max-actions-per-turn", type=int, default=20)
    parser.add_argument("--reflect-every", type=int, default=8)
    return parser.parse_args()


def _skill_path(name: str) -> Path:
    path = CHASSIS / "skills" / name / "SKILL.md"
    if not path.exists():
        raise FileNotFoundError(f"skill missing: {path}")
    return path


TERMINAL_STATES = ("WIN", "GAME_OVER")


def _done_refusal(latest: dict) -> str:
    """The host note that answers an Actor DONE on an unfinished game.

    In the 2026-09-12 objection-closure run the Actor declared LS20 "stuck" with
    152s and 193 actions remaining, returned DONE, and the worker fell silent
    until the host's decision timeout killed the whole three-game run. The Actor
    does not own that verdict. Redelivering the state costs nothing; if the
    model budget really is spent, the admission gate ends the game by the
    ordinary budget_exhausted path.
    """
    return (
        f"HOST: the game is not finished (state={latest.get('state')}, "
        f"levels_completed={latest.get('levels_completed')}). Your DONE was not "
        "accepted; the host ends games, not you. Submit an action now. If the "
        "model budget is spent the host will stop admitting calls on its own."
    )


async def _headless_dispatch(agent: MdArcSolverAgent) -> None:
    delivery = StateDelivery()
    redeliver: dict[str, list] | None = None
    while True:
        if (agent.ipc_dir / "ABORT").exists():
            raise RuntimeError((agent.ipc_dir / "ABORT").read_text())
        if redeliver is not None:
            notification, redeliver = redeliver, None
            latest = agent.latest_state()
        else:
            try:
                wins = await agent.queue_manager.race()
            except (asyncio.CancelledError, ValueError):
                break
            notification = {}
            for name, item in wins:
                notification.setdefault(name, []).append(item)
            incoming = notification.get("game_states", [])
            latest = agent.latest_state()
            notification = delivery.prepare(notification, latest)
            with (agent.ipc_dir / "state_delivery.jsonl").open("a") as log:
                log.write(json.dumps({"latest": {"turn": latest.get("turn"), "step": latest.get("step")}, "incoming": incoming, "delivered": notification.get("game_states", []) if notification else [], "suppressed": notification is None}) + "\n")
            if notification is None:
                continue
        # The notification has been consumed and handle() may already have
        # submitted actions. Propagate failure to worker supervision: waiting
        # for another notification can strand this decision, and replaying it
        # here could duplicate side effects.
        try:
            continuation = (
                getattr(agent, '_continue_contingent_plan', None)
                if contingent_enabled()
                else None
            )
            if continuation is not None and continuation(notification):
                continue
            agent._active_dispatch_turn = latest.get('turn')
            try:
                result = await agent.handle(notification)
            finally:
                agent._active_dispatch_turn = None
        except asyncio.CancelledError:
            break
        if getattr(result, "kind", None) == RespondReason.DONE:
            print(
                f"[headless] agent returned DONE: {result.explanation}", file=sys.stderr
            )
            latest = agent.latest_state()
            if latest and latest.get("state") not in TERMINAL_STATES:
                with (agent.ipc_dir / "done_refused.jsonl").open("a") as log:
                    log.write(json.dumps({"turn": latest.get("turn"), "step": latest.get("step"),
                                          "state": latest.get("state"),
                                          "explanation": str(getattr(result, "explanation", ""))[:400]}) + "\n")
                redeliver = {"game_states": [json.dumps(latest)],
                             "user_messages": [_done_refusal(latest)]}


def _limit_worker_memory() -> None:
    # Actor code runs in this CPU worker, separate from the GPU model server.
    # Bound allocation failures such as accidentally accumulating awaitables.
    _worker_memory = 4 * 1024**3
    _memory_soft, _memory_hard = resource.getrlimit(resource.RLIMIT_AS)
    _worker_memory = min(_worker_memory, _memory_hard) if _memory_hard != resource.RLIM_INFINITY else _worker_memory
    resource.setrlimit(resource.RLIMIT_AS, (_worker_memory, _worker_memory))


async def run() -> None:
    _limit_worker_memory()
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    (run_dir / "team_nemo" / "shared").mkdir(parents=True, exist_ok=True)
    llm = build_llm(request_timeout=float(os.environ.get('ARC_LLM_TIMEOUT_SECONDS','300')))
    truncation = TruncationConfig(
        max_context_tokens=20_000,
        capture=CaptureConfig(max_stdout=3_000),
        event_format=FormatConfig(max_string=1_200, max_length=50, max_depth=3),
    )
    procedural_diagnostic = os.environ.get('ARC_PROCEDURAL_DIAGNOSTIC', '0') == '1'
    agent_class = ProceduralAgent if procedural_diagnostic else MdArcSolverAgent
    # The contingent interface is the isolated treatment.  In the off arm the two
    # tools leave the documented surface, their prompt sentences are withheld, and
    # the dispatcher fast path is skipped; solver_agent also rejects a direct call.
    # Each arm is a separate worker process, so marking the class is process-local.
    if not contingent_enabled():
        for tool in ('compare_contingent_plans', 'submit_contingent_plan'):
            hidden(getattr(MdArcSolverAgent, tool))

    agent = agent_class(
        llm=llm,
        truncation=truncation,
        run_dir=run_dir,
        game_id="the game",
        alias="the game",
        reflect_every=args.reflect_every,
        visual=args.visual,
        png_scale=args.png_scale,
        skill_path=None if procedural_diagnostic else _skill_path(args.skill),
        max_actions_per_turn=args.max_actions_per_turn,
        # This short, fresh-session competition baseline must not leave an
        # independently scheduled summary request behind at a game boundary.
        enable_summarizer=False,
        # Model-free eviction bounds the transcript instead (see
        # chassis/transcript_eviction.py). 0 restores v11's unbounded growth.
        transcript_keep_turns=int(os.environ.get("ARC_TRANSCRIPT_KEEP_TURNS", "4")),
    )
    # Discovery factors are deliberately independent.  A is the common fresh
    # baseline; B adds measured evidence, C adds one execution-generated
    # discovery procedure, and D adds both.  Working-belief instructions are
    # common to all four arms and are not a treatment factor.
    if procedural_diagnostic:
        from nooa.context_blocks import Context
        assert not getattr(agent, '_summarizers', ()), 'diagnostic installed a summarizer'
        for name in ('prediction_workflow', 'knowledge_api', 'arc_game'):
            del agent.context[name]
        agent.context['decision_workflow'] = Context(
            'PROCEDURAL TRANSFER DIAGNOSTIC. Reuse the procedure, never the example conclusion. '
            'Within at most two execute_python calls, inspect the current grid, call the exact '
            'self.set_experiment(question, hypotheses, action, outcomes) signature, and submit one '
            'discriminating action. outcomes must name a target and compatible metric and give each hypothesis '
            'a distinct typed prediction, its update, and an otherwise update. After the returned host '
            'transition, inspect decision_packet()["experiment_result"] plus changed_region() or '
            'last_transition() and within at most two execute_python calls submit the next action. '
            'Do not create or edit helpers, call deep_investigate, or save a procedure. Make no claim '
            'broader than the returned pixels support.', prefix=True)
        agent.context['discovery_demonstration'] = Context(
            (CHASSIS / 'discovery-demo.md').read_text(), prefix=True)
        case = os.environ.get('ARC_PROCEDURAL_CASE', 'salient_noop')
        case_briefs = {
            'changed_action_mapping': 'Action labels may map differently than the example. Test rather than assume the mapping.',
            'competing_regions': 'State two hypotheses that predict different changed-region coordinates before acting.',
            'reshaped_target': 'The target may reshape rather than translate. Separate shape evidence from displacement.',
            'translated_variant': 'Colors and positions differ from the example. Reuse only the measurement procedure.',
            'salient_noop': 'The visually salient probe may have no effect. Treat an unchanged successor as evidence.',
            'shape_transform': 'Test transformation form. Use target=shape and predict distinct resulting grid shapes before acting.',
            'color_transition': 'Test color transformation. Use target=color and predict distinct exact color-change signatures before acting.',
            'effect_magnitude': 'Test effect magnitude. Use target=change_count and predict distinct changed-cell counts before acting.',
        }
        if case not in case_briefs:
            raise ValueError(f'unknown procedural case {case!r}')
        agent.context['transfer_case'] = Context(case_briefs[case], prefix=True)
        discovery_arm = 'a'
    else:
        discovery_arm = os.environ.get('ARC_DISCOVERY_ARM', 'a').lower()
    if discovery_arm not in ('a', 'b', 'c', 'd'):
        raise ValueError('ARC_DISCOVERY_ARM must be one of a, b, c, d')
    with_evidence = discovery_arm in ('b', 'd')
    with_demo = discovery_arm in ('c', 'd')
    from nooa.context_blocks import Context
    agent.context['tempo_contract'] = Context(ACTOR_TEMPO_CONTEXT, prefix=True)
    # Only these sentences differ between the contingent arms; every other
    # instruction, gate, budget and tool is shared.
    contingent_instructions = (
        'When plausible explanations imply different next actions, use compare_contingent_plans(proposal) '
        'and submit_contingent_plan(proposal) to compare short observation-contingent choices. '
        'Supply provisional completion and remaining/recovery action costs for each continuation in every hypothesis. '
        'The controller may take one matching continuation without another model call, then hands back. '
        'If these estimates lack a useful basis, use ordinary single-action exploration. '
        'Do not invent extra hypotheses for an agreed next action. '
    ) if contingent_enabled() else ''
    contingent_scope = ' outside a contingent plan' if contingent_enabled() else ''
    if not procedural_diagnostic:
        agent.context['decision_workflow'] = Context(
        'Use the measured capsule when present; otherwise start with self.decision_packet(). '
        'Retrieve exact grids or trajectory rows only when a current question requires them. '
        'One Python cell may inspect several local views, update a helper or save_procedure(), '
        'preflight it, set_experiment(), and submit_actions(). Cross back to the model only '
        'when a genuinely new semantic decision or environment observation is required. '
        + contingent_instructions +
        'For a new unresolved experiment' + contingent_scope + ', call self.set_experiment(question, hypotheses, '
        'action, outcomes) in the SAME cell as your justified self.submit_actions(...). '
        'outcomes must contain target, metric, predictions, updates, and otherwise. Prediction keys and '
        'update keys exactly match the two or three hypotheses, and predicted values must differ. '
        'Read the board with self.grid_letters(), not by printing grid_array or model_grid. '
        'Do not print full grids, helper source, or old trajectories unless the missing detail '
        'is decision-relevant. '
        + (
            'When a short action sequence is selected, submit the whole sequence in one '
            'self.submit_actions call. Open batching is on: a predicted_path is not required. '
            if os.environ.get('ARC_DUCK_BATCH', '0') == '1' else ''
        ) +
        'World-model construction is optional: build only enough code '
        'to answer the live question. Compile a supported recurring operation with '
        'save_procedure() so future actions do not require rediscovery. Routine reasoning is '
        'off. Use await self.deep_investigate(...) only for a hard abstraction failure whose '
        'answer could enable many actions, never for routine inspection. '
        'Keep records short; they are conjectures, not proof of progress. '
        'Queued actions stop at the first frozen-model mismatch or unavailable check; '
        'inspect the reason before continuing. If prediction and retrodiction gates pass, '
        'submit the already-decided safe sequence without asking for reconfirmation. One '
            'exploratory action remains allowed.', prefix=True)
    if with_evidence and not procedural_diagnostic:
        # Use the supported registration API. Persistence through summarizer
        # requests still requires qualification; registration alone is no proof.
        agent.context['measured_evidence'] = Context(expr='self.render_measured_evidence()')
    if with_demo and not procedural_diagnostic:
        # Fixed interface-feature routing; no retrieval/model call is spent.
        interface = os.environ.get('ARC_DISCOVERY_INTERFACE', 'directional')
        demos = {'directional': CHASSIS / 'discovery-demo.md'}
        if interface not in demos:
            raise ValueError(f'no frozen discovery demonstration for interface {interface!r}')
        demo_path = demos[interface]
        if not demo_path.is_file():
            raise FileNotFoundError(f'execution-generated demonstration missing: {demo_path}')
        agent.context['discovery_demonstration'] = Context(demo_path.read_text(), prefix=True)
    states_path = run_dir / "ipc" / "states.jsonl"

    async def _kickoff() -> None:
        while True:
            try:
                if states_path.stat().st_size > 0:
                    break
            except OSError:
                pass
            await asyncio.sleep(0.05)
        agent._user_messages_in.put(
            "Start solving the game. Read self.latest_state() synchronously for the current "
            "observation; the dispatcher owns the notification queue. Follow the decision_workflow context block."
        )

    kickoff = asyncio.create_task(_kickoff())
    try:
        await _headless_dispatch(agent)
    finally:
        kickoff.cancel()
        await asyncio.gather(kickoff, return_exceptions=True)
        await agent.queue_manager.shutdown()


if __name__ == "__main__":
    if os.environ.get("ARC_LLM_MODEL"):
        os.environ.setdefault("ARC_LLM_MODEL", os.environ["ARC_LLM_MODEL"])
    asyncio.run(run())
