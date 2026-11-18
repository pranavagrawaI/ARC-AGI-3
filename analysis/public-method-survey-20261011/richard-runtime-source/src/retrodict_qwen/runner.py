"""Per-game controller: Retrodict's ``arc3/runner.py`` on our bridge and clock.

The queue drains one action per env step with zero LLM calls. The agent is re-invoked when
the queue empties, the level or game state changes, or a planned action stops being
available. On GAME_OVER the runner issues RESET (an attempt reset — whole-game restarts are
never used) and tells the agent.

Kept verbatim from upstream: ``_accept``, ``_available_names``, ``_render_action`` and
``_board_lists``, and ``_loop`` made ``async`` (tests/test_retrodict_qwen.py diffs them
against the submodule). Removed: pricing and the USD cost cap (a gameplay clock ends the
run instead), image priming, ``--resume`` replay, the containment check, the
Arcade/scorecard glue and the CLI.

v5 (18 September, docs/retrodict-qwen-baseline.md): per-invocation budget (tool-less parse
retries, budget counters), HUD-tolerant ``expect`` checks with board receipts, ``[OBJECTS]``
log lines for large diffs, a stuck signal rescaled to invocations and the clock, no-change
repeats and level-start RESETs skipped, and the re-invoke prompt inlining the new steps'
diffs, the current board's objects and the runner's notes.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from arc_infra.objects import object_view
from arc_infra.perception import board_hash

from . import prompts
from .feedback import (
    LARGE_DIFF,
    LevelChanges,
    TransitionCache,
    format_objects,
    receipt_text,
    transition,
)
from .logwriter import LogWriter, StepRecord, action_name, diff_boards, format_diff
from .plan_parser import ParsedPlan, PlannedAction, PlanParseError, parse_actions
from .thin.types import HarnessError

_PARSE_RETRY_LIMIT = 3  # re-prompt the model this many times on a bad [ACTIONS] block before giving up
_ESCALATE_RESETS = 2  # self-issued RESETs on one level before the stuck directive fires
_ESCALATE_ACTIONS = 300  # upstream: actions on one level before the directive fires (never reached at ~40 actions per game)
# v5 (18 September): the stuck signal is rescaled to our budget. Tier 1 fires after this many
# invocations without a level change, or once the level has consumed this share of the game
# clock; tier 2 after the same number of further invocations.
_ESCALATE_INVOCATIONS = 4
_ESCALATE_CLOCK_FRACTION = 0.4
_SUMMARY_STEPS = 20  # v5: new steps inlined per re-invoke prompt (older ones folded into one line)
_OBJECT_VIEW_LIMIT = 40


@dataclass(frozen=True)
class RunnerConfig:
    """Per-run knobs: upstream's pilot protocol minus pricing, plus the gameplay clock."""

    game_id: str
    model: str = "local:qwen"
    reasoning_effort: str | None = "xhigh"
    max_output_tokens: int = 32_768
    action_cap: int = 2_000
    fresh_session_input_tokens: int = 150_000
    max_model_requests: int = 128
    request_timeout: int = 600
    gameplay_seconds: float = 900


@dataclass(frozen=True)
class AgentReply:
    """One agent invocation's model-facing outcome."""

    text: str
    resume_state: dict[str, Any] | None
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    model_requests: int
    stop_reason: str
    truncated: bool
    # Input tokens of the invocation's final provider request = the current
    # conversation size. The RunUsage.input_tokens total re-counts the cached
    # prefix once per tool round, so it is a cost number, not a context size.
    last_context_tokens: int = 0
    # v5: final-request notices (request or time budget) and length continuations.
    budget_notices: int = 0
    length_cuts: int = 0
    # v5.2: requests spent on rejected tool calls (not charged to the request budget).
    error_requests: int = 0


class AgentClient(Protocol):
    async def invoke(self, prompt: str, resume_from: dict[str, Any] | None, *, tools: bool = True) -> AgentReply: ...


class GameEnv(Protocol):
    async def reset(self) -> Any: ...

    async def step(self, action: Any, data: dict[str, Any] | None = None) -> Any: ...


@dataclass
class RunState:
    """Mutable per-run accounting."""

    step_no: int = 0
    actions_taken: int = 0
    invocations: int = 0
    fresh_sessions: int = 0
    parse_retries: int = 0
    surprises: int = 0
    surprise_note: str | None = None
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    model_requests: int = 0
    resume_state: dict[str, Any] | None = None
    context_tokens: int = 0
    last_planned_step: int = 0
    level_start_actions: int = 0
    level_self_resets: int = 0
    escalation_tier: int = 0
    escalated_at_actions: int = 0
    budget_notices: int = 0
    length_cuts: int = 0
    # v5: expectation cells on the HUD strip ignored (count, and notes for the next prompt).
    hud_ignored: int = 0
    hud_notes: list[str] = field(default_factory=list)
    # v5: stuck signal per level (invocations and clock), no-change repeats and RESET handling.
    level_start_invocations: int = 0
    level_start_time: float = 0.0
    escalated_at_invocations: int = 0
    level_fresh: bool = True  # no action executed since the level (re)started: RESET does nothing
    no_ops: set = field(default_factory=set)  # (levels, board hash, name, x, y) that changed nothing
    no_change_actions: list[str] = field(default_factory=list)  # this level, rendered
    skipped_repeats: list[str] = field(default_factory=list)  # since the last invocation
    skipped_total: int = 0
    reset_skipped: int = 0
    drain_notes: list[str] = field(default_factory=list)  # for the next prompt
    recent_steps: list[str] = field(default_factory=list)  # v5: one summary line per logged step
    invocation_log: list[dict[str, Any]] = field(default_factory=list)


class GameRunner:
    """Run one game to a terminal condition."""

    def __init__(self, env: GameEnv, agent: AgentClient, cfg: RunnerConfig, run_dir: Path, *, clock, archive) -> None:
        self.env = env
        self.agent = agent
        self.cfg = cfg
        self.clock, self.archive = clock, archive
        self.run_dir = Path(run_dir)
        self.workspace = self.run_dir / "workspace"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.log = LogWriter(self.workspace / "log.txt")
        self.transcript_path = self.run_dir / "transcript.jsonl"
        self.state = RunState()
        self.frame: Any = None
        self.prime_note: str | None = None
        # v5: cached segmentation for [OBJECTS] lines and receipts; per-level HUD statistics.
        self.transitions = TransitionCache()
        self.changes = LevelChanges()
        self._prev_settled: list[list[int]] | None = None

    async def run(self) -> dict[str, Any]:
        """Play until WIN, a cap, or a failure; return (and write) metrics."""
        started = time.time()
        stop_reason = await self._start_fresh()
        if stop_reason is None:
            try:
                stop_reason = await self._loop("start")
            except HarnessError as exc:
                # A provider failure must not lose the run record.
                self.state.invocation_log.append({"error": str(exc)})
                stop_reason = "provider_error"
        metrics = self._metrics(stop_reason, time.time() - started)
        (self.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        self.archive.event("retrodict_metrics", **{k: v for k, v in metrics.items() if k != "invocation_log"})
        return metrics

    async def _start_fresh(self) -> str | None:
        # Arcade.make already performed the opening RESET (recorded as observation 0), so
        # the run starts from that frame without charging an action. Upstream called
        # env.reset() itself and counted it as action 1.
        self.frame = await self.env.current()
        if self.frame is None:
            return "env_error"
        self._log_frame("RESET")
        self._start_level()
        return None

    async def _loop(self, reason: str) -> str:
        while True:
            if self.clock.remaining() <= 0:
                return "deadline"
            if self.state.actions_taken >= self.cfg.action_cap:
                return "action_cap"
            plan = await self._invoke(reason)
            if plan is None:
                return "plan_parse_failed"
            reason = await self._drain(plan)
            if reason in {"win", "action_cap", "env_error"}:
                return reason

    # -- agent invocation ---------------------------------------------------

    async def _invoke(self, reason: str) -> ParsedPlan | None:
        prompt, resume = self._build_prompt(reason)
        reply = await self._call(prompt, resume)
        for attempt in range(_PARSE_RETRY_LIMIT + 1):
            try:
                return self._accept(reply)
            except PlanParseError as exc:
                if attempt == _PARSE_RETRY_LIMIT:
                    return None
                self.state.parse_retries += 1
                # v5: a parse retry is a tool-less request, one request of the budget.
                reply = await self._call(prompts.parse_retry_prompt(str(exc)), reply.resume_state, tools=False)
        return None  # unreachable: the loop always returns

    def _build_prompt(self, reason: str) -> tuple[str, dict[str, Any] | None]:
        reason_text = prompts.REINVOKE_REASONS.get(reason, reason)
        if self.state.surprise_note is not None:
            reason_text = f"{reason_text} ({self.state.surprise_note})"
            self.state.surprise_note = None
        if self.state.invocations == 0:
            return prompts.initial_prompt(self.cfg.game_id, self.prime_note), None
        directive = self._escalation_directive()
        if self.state.resume_state is None or self.state.context_tokens > self.cfg.fresh_session_input_tokens:
            self.state.fresh_sessions += 1
            summary = self._step_summary(self.state.step_no - min(5, self.state.step_no) + 1)
            return prompts.fresh_session_prompt(self.cfg.game_id, self.state.step_no, reason_text, summary) + directive, None
        summary = self._step_summary(self.state.last_planned_step + 1)
        prompt = prompts.reinvoke_prompt(reason_text, self.state.last_planned_step + 1, self.state.step_no, summary)
        return prompt + directive, self.state.resume_state

    def _step_summary(self, first_step: int) -> str:
        """v5: the new steps' actions and diffs, the state line, the current board's objects
        and the runner's notes (HUD cells ignored, skipped repeats, no-change actions)."""
        state = self.state
        lines = state.recent_steps[max(0, len(state.recent_steps) - (state.step_no - first_step + 1)) :]
        if len(lines) > _SUMMARY_STEPS:
            folded = len(lines) - _SUMMARY_STEPS
            lines = [f"({folded} earlier steps omitted; see log.txt)"] + lines[-_SUMMARY_STEPS:]
        frame = self.frame
        parts = ["New steps:", *lines] if lines else []
        parts.append(
            f"[LEVELS] {frame.levels_completed}/{frame.win_levels} [STATE] {frame.state.value} "
            f"[AVAILABLE] {' '.join(action_name(a) for a in frame.available_actions)}"
        )
        parts.append("Current board objects (largest first):")
        parts.append(object_view(self.transitions.objects(_board_lists(frame.frame[-1])), limit=_OBJECT_VIEW_LIMIT))
        notes = [*state.hud_notes, *state.drain_notes]
        if state.skipped_repeats:
            notes.append(
                f"skipped {len(state.skipped_repeats)} planned repeat(s) that already changed nothing on that "
                f"exact board: {', '.join(state.skipped_repeats[:8])}"
            )
        if state.no_change_actions:
            recent = ", ".join(state.no_change_actions[-6:])
            notes.append(f"{len(state.no_change_actions)} action(s) changed nothing on this level so far (latest: {recent})")
        if notes:
            parts.append("Notes: " + " | ".join(notes))
        state.hud_notes.clear()
        state.drain_notes.clear()
        state.skipped_repeats.clear()
        return "\n".join(parts)

    def _escalation_directive(self) -> str:
        """Escalate a stuck level (par-free signals only) and render the binding directive, or ''.

        v5: upstream fired after 300 actions or two self-resets on a level; at ~40 actions per
        900 s game it never fired. The signal is now invocations on the level (four without a
        level change) or the level's share of the game clock (40%), plus the two self-resets.
        """
        state = self.state
        actions_here = state.actions_taken - state.level_start_actions
        invocations_here = state.invocations - state.level_start_invocations
        seconds_here = time.monotonic() - state.level_start_time
        stuck = (
            state.level_self_resets >= _ESCALATE_RESETS
            or invocations_here >= _ESCALATE_INVOCATIONS
            or seconds_here >= _ESCALATE_CLOCK_FRACTION * self.cfg.gameplay_seconds
        )
        if state.escalation_tier == 0 and stuck:
            state.escalation_tier = 1
            state.escalated_at_actions = actions_here
            state.escalated_at_invocations = invocations_here
            self._record_escalation(actions_here)
        elif state.escalation_tier == 1 and invocations_here - state.escalated_at_invocations >= _ESCALATE_INVOCATIONS:
            state.escalation_tier = 2
            self._record_escalation(actions_here)
        if state.escalation_tier == 0:
            return ""
        return "\n\n" + prompts.escalation_directive(state.escalation_tier, actions_here, state.level_self_resets)

    def _record_escalation(self, actions_here: int) -> None:
        record = {
            "escalation_tier": self.state.escalation_tier,
            "levels_completed": self.frame.levels_completed,
            "actions_this_level": actions_here,
            "level_self_resets": self.state.level_self_resets,
            "invocations_this_level": self.state.invocations - self.state.level_start_invocations,
            "seconds_this_level": round(time.monotonic() - self.state.level_start_time, 1),
        }
        self.state.invocation_log.append(record)
        with self.transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.archive.event("retrodict_escalation", **record)

    async def _call(self, prompt: str, resume: dict[str, Any] | None, *, tools: bool = True) -> AgentReply:
        started = time.monotonic()
        try:
            reply = await self.agent.invoke(prompt, resume, tools=tools)
        except HarnessError:
            # One retry: provider 400s (e.g. moderation false positives) and
            # transient 5xx often clear on resubmission of the same request.
            await asyncio.sleep(5)
            reply = await self.agent.invoke(prompt, resume, tools=tools)
        self.state.invocations += 1
        self.state.budget_notices += reply.budget_notices
        self.state.length_cuts += reply.length_cuts
        self.state.input_tokens += reply.input_tokens
        self.state.cached_tokens += reply.cached_tokens
        self.state.output_tokens += reply.output_tokens
        self.state.model_requests += reply.model_requests
        self.state.context_tokens = reply.last_context_tokens
        self.state.resume_state = reply.resume_state
        record = {
            "invocation": self.state.invocations,
            "prompt": prompt,
            "resumed": resume is not None,
            "text": reply.text,
            "stop_reason": reply.stop_reason,
            "truncated": reply.truncated,
            "model_requests": reply.model_requests,
            "input_tokens": reply.input_tokens,
            "cached_tokens": reply.cached_tokens,
            "output_tokens": reply.output_tokens,
            "context_tokens": reply.last_context_tokens,
            "budget_notices": reply.budget_notices,
            "length_cuts": reply.length_cuts,
            "error_requests": reply.error_requests,
        }
        self.state.invocation_log.append(record)
        with self.transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.archive.event(
            "retrodict_call",
            seconds=time.monotonic() - started,
            fresh_session=resume is None and self.state.invocations > 1,
            **{k: v for k, v in record.items() if k not in {"prompt", "text"}},
        )
        return reply

    def _accept(self, reply: AgentReply) -> ParsedPlan:
        plan = parse_actions(
            reply.text,
            available=self._available_names(),
            max_actions=self.cfg.action_cap - self.state.actions_taken,
            truncated=reply.truncated,
        )
        rendered = " ".join(_render_action(action) for action in plan.actions)
        self.log.append_plan(self.state.invocations, f"{plan.reasoning}\nplan: {rendered}")
        self.state.last_planned_step = self.state.step_no
        return plan

    def _available_names(self) -> set[str]:
        return {action_name(a) for a in self.frame.available_actions} | {"RESET"}

    # -- queue drain ---------------------------------------------------------

    async def _drain(self, plan: ParsedPlan) -> str:
        """Execute the plan one action per env step.

        v5: a RESET planned as a level's first action is skipped (the engine ignores it), and
        an action that already produced no change on this exact board is skipped and reported
        (Duck v5's repeat rule). A plan whose every action was such a repeat re-invokes with
        ``no_change_repeat``.
        """
        queue = list(plan.actions)
        executed = skipped = 0
        while queue:
            if self.state.actions_taken >= self.cfg.action_cap:
                return "action_cap"
            action = queue[0]
            if action.name != "RESET" and action.name not in self._available_names():
                return "unavailable_action"
            queue.pop(0)
            if action.name == "RESET" and self.state.level_fresh:
                self.state.reset_skipped += 1
                self.state.drain_notes.append("RESET as a level's first action does nothing here and was skipped")
                self.archive.event("retrodict_skip", cause="reset_at_level_start", step=self.state.step_no)
                continue
            key = self._repeat_key(action)
            if key in self.state.no_ops:
                skipped += 1
                self.state.skipped_total += 1
                self.state.skipped_repeats.append(_render_action(action))
                self.archive.event("retrodict_skip", cause="no_change_repeat", step=self.state.step_no, action=_render_action(action))
                continue
            executed += 1
            outcome = await self._step(action)
            if outcome is not None:
                return outcome
        if skipped and not executed:
            return "no_change_repeat"
        if plan.expect_levels is not None and not plan.clamped and self.frame.levels_completed != plan.expect_levels:
            self.state.surprises += 1
            note = f"you expected {plan.expect_levels} completed levels after the plan; the board shows {self.frame.levels_completed}"
            self.state.surprise_note = note
            return "prediction_mismatch"
        return "queue_empty"

    def _repeat_key(self, action: PlannedAction) -> tuple:
        return (self.frame.levels_completed, board_hash(self.frame.frame[-1]), action.name, action.x, action.y)

    async def _step(self, action: PlannedAction) -> str | None:
        prev_levels = self.frame.levels_completed
        prev_state = self.frame.state
        prev_settled = _board_lists(self.frame.frame[-1])
        key = self._repeat_key(action)
        data = {"x": action.x, "y": action.y} if action.name == "ACTION6" else None
        frame = await self.env.step(action.name, data)
        if frame is None:
            return "env_error"
        self.frame = frame
        self.state.actions_taken += 1
        self.state.step_no += 1
        self._log_frame(action.name, x=action.x, y=action.y, prior_settled=None if action.name == "RESET" else prev_settled)
        if getattr(frame, "full_reset", False):  # never expected under level-only resets
            self.archive.event("full_reset", step=self.state.step_no, action=action.name)
        if action.name == "RESET":
            self.state.level_self_resets += 1
            self.state.level_fresh = True
            self.state.no_ops.clear()
        else:
            self.state.level_fresh = False
            if _board_lists(frame.frame[-1]) == prev_settled:
                self.state.no_ops.add(key)
                self.state.no_change_actions.append(_render_action(action))
        state = frame.state.value
        if state == "WIN":
            return "win"
        if state == "GAME_OVER":
            return await self._reset_after_game_over()
        if frame.levels_completed != prev_levels:
            self._start_level()
            return "level_change"
        if frame.state != prev_state:
            return "state_change"
        return self._check_expectations(action)

    def _check_expectations(self, action: PlannedAction) -> str | None:
        """v5: mismatches confined to the HUD strip are noted, not fatal; a real mismatch
        stops the plan with the cells and a board receipt (changed cells, objects)."""
        if not action.expect:
            return None
        settled = self.frame.frame[-1]
        mismatches, hud = [], []
        for x, y, color in action.expect:
            actual = int(settled[y][x])
            if actual == color:
                continue
            if self.changes.hud(x, y):
                hud.append(f"(x={x},y={y})")
            else:
                mismatches.append(f"cell (x={x},y={y}) is {actual}, you expected {color}")
        if hud:
            self.state.hud_ignored += len(hud)
            self.state.hud_notes.append(
                f"after {action.name}: expect on HUD cell{'s' if len(hud) > 1 else ''} {', '.join(hud)} ignored "
                "(that border strip changes on most steps)"
            )
        if not mismatches:
            return None
        self.state.surprises += 1
        summary = transition(self.transitions, self._prev_settled, _board_lists(settled))
        self.state.surprise_note = receipt_text(action.name, mismatches, summary)
        return "prediction_mismatch"

    def _start_level(self) -> None:
        """A new level began: reset the per-level stuck signals and stand down any escalation."""
        state = self.state
        state.level_start_actions = state.actions_taken
        state.level_self_resets = 0
        state.escalation_tier = 0
        state.escalated_at_actions = 0
        state.level_start_invocations = state.invocations
        state.level_start_time = time.monotonic()
        state.escalated_at_invocations = 0
        state.level_fresh = True
        state.no_ops.clear()
        state.no_change_actions.clear()
        self.changes.reset()

    async def _reset_after_game_over(self) -> str:
        if self.state.actions_taken >= self.cfg.action_cap:
            return "action_cap"
        frame = await self.env.reset()
        if frame is None:
            return "env_error"
        self.frame = frame
        self.state.actions_taken += 1
        self.state.step_no += 1
        self._log_frame("RESET")
        self.state.level_fresh = True
        self.state.no_ops.clear()
        return "game_over"

    def _log_frame(
        self,
        action: str,
        *,
        x: int | None = None,
        y: int | None = None,
        prior_settled: list[list[int]] | None = None,
    ) -> None:
        frame = self.frame
        frames = [_board_lists(board) for board in frame.frame]
        diff = diff_boards(prior_settled, frames[-1]) if prior_settled is not None else None
        objects = None
        if diff is not None:
            self.changes.note(diff)
            if len(diff) > LARGE_DIFF:  # v5: the listed diff stops here; say what moved instead
                objects = format_objects(transition(self.transitions, prior_settled, frames[-1]))
        self._prev_settled = prior_settled
        rendered = action if x is None else f"{action} x={x} y={y}"
        change = "no diff (attempt start)" if diff is None else format_diff(diff)[len("[DIFF] ") :]
        line = f"step {self.state.step_no}: {rendered} -> {change}"
        if objects:
            line += f"; objects: {objects}"
        self.state.recent_steps.append(line)
        del self.state.recent_steps[:-_SUMMARY_STEPS - 1]
        self.log.append_step(
            StepRecord(
                step=self.state.step_no,
                action=action,
                frames=frames,
                levels_completed=frame.levels_completed,
                win_levels=frame.win_levels,
                state=frame.state.value,
                available_actions=list(frame.available_actions),
                x=x,
                y=y,
            ),
            diff=diff,
            objects=objects,
        )

    # -- accounting ----------------------------------------------------------

    def _metrics(self, stop_reason: str, wall_seconds: float) -> dict[str, Any]:
        state = self.state
        return {
            "game_id": self.cfg.game_id,
            "model": self.cfg.model,
            "reasoning_effort": self.cfg.reasoning_effort,
            "image_prime": False,
            "stop_reason": stop_reason,
            "state": self.frame.state.value if self.frame is not None else None,
            "levels_completed": self.frame.levels_completed if self.frame is not None else None,
            "win_levels": self.frame.win_levels if self.frame is not None else None,
            "actions": state.actions_taken,
            "invocations": state.invocations,
            "fresh_sessions": state.fresh_sessions,
            "parse_retries": state.parse_retries,
            "surprises": state.surprises,
            "model_requests": state.model_requests,
            "input_tokens": state.input_tokens,
            "cached_tokens": state.cached_tokens,
            "output_tokens": state.output_tokens,
            "escalation_tier": state.escalation_tier,
            "budget_notices": state.budget_notices,
            "length_cuts": state.length_cuts,
            "hud_ignored": state.hud_ignored,
            "skipped_repeats": state.skipped_total,
            "reset_skipped": state.reset_skipped,
            "wall_seconds": round(wall_seconds, 1),
            "invocation_log": state.invocation_log,
        }


def _render_action(action: PlannedAction) -> str:
    if action.x is not None:
        return f"{action.name}(x={action.x},y={action.y})"
    return action.name


def _board_lists(board: Any) -> list[list[int]]:
    return board.tolist() if hasattr(board, "tolist") else [[int(cell) for cell in row] for row in board]
