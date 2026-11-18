"""Provider projection; canonical events and workspace evidence remain intact."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from nooa.config.truncation_config import TruncationConfig, FormatConfig
from tycho_nooa.llm_trace import LLMTrace
from inference.agent.tool_agent import _estimate_tokens, _split_messages_for_estimate
from inference.agent.runtime_state import Frame
from inference.agent.vision_context import frame_to_png_data_url, diff_image_part

# One worker owns one game. Every model-using role contributes to this ledger.
LEDGER = {"completion_tokens": 0, "prompt_tokens": 0, "cached_tokens": 0,
          "evictions": 0, "calls": 0, "roles": {}}
OBSERVATION = None
PREVIOUS = None
DEATH = None
# (offset, reason) of the last queued batch the host stopped; reported once.
BATCH_STOP = None
# dfranzen's gameplay_changed: a HUD such as a budget bar sits in this border.
BORDER = 4

# Constant across a game, so it stays inside the cached system prefix. The task
# statement restores canonical Tycho's actor.system.j2 objective, which the NOOA
# Actor never received; the rest ports dfranzen's flag and level-transfer text.
SYSTEM_GUIDANCE = """Game rules and scoring (host facts, not hypotheses):
- You play an unfamiliar turn-based game on a grid of colored cells (usually 64x64, colors 0-15). It has several levels of usually increasing complexity; a level ends when you reach its unknown goal. Objective: complete as many levels as possible, each in as few actions as possible. Every action you commit with take_actions, including RESET, is a scored environment action; tool calls cost time, not actions.
- Flag semantics in `turn`: state='GAME_OVER' means this attempt FAILED (death or a limit ran out); it never means the run is won. While state is GAME_OVER the only legal action is RESET, which restarts the current level from its initial state and keeps completed levels. A rising levels_completed means you advanced one level. state='WIN' means the whole game is won.
- Deaths come either from the action itself (e.g. a hazard cell) or from a step/time budget depleting. Watch for a bar at the grid border that shrinks each action, and count the remaining budget into your plans.
- The outer 4 cells often hold such HUD elements. A change confined to that border does not show that an action did anything in play, and a queued batch stops after an action that leaves the interior unchanged. Genuine game elements can also sit near the border.
- Levels usually build on mechanics learned in earlier levels, especially the most recent one. Carry forward supported knowledge as a starting hypothesis, while re-checking anything contradicted by new evidence. New levels often introduce additional mechanics, sometimes through unfamiliar board elements; these are often needed to solve the level. The goal may stay the same but need new mechanics, or the goal itself may change."""

_GAME_OVER = """GAME OVER: the previous action ended this attempt in FAILURE (it does NOT mean the run is finished). Only RESET is legal now; it restarts the current level from its initial state and keeps completed levels. Before resetting, diagnose the cause. The current board shows the fatal state; wmlib.death_events() returns the board before the fatal action ('prev') and the death frame ('next'), also saved as the newest death_* files of this level. Check what the fatal action targeted and inspect the budget bar in the death frame. BAR RULE: the bar shrinking with each action is NORMAL; a slightly shorter bar is NOT evidence of the cause of death. The budget caused the death ONLY if the bar is fully (or almost fully) depleted in the death frame. If it is: reach the goal in fewer actions, or find a way to restore the bar (e.g. an item that refills it). If the bar clearly had budget left, the fatal action itself caused the death (hazard or forbidden move): do NOT retry the same path. The game is deterministic, so replaying the sequence that killed you will kill you again; change the plan BEFORE the fatal point. Record the diagnosed cause in your notes."""

_LEVEL_UP = """You have completed the previous level. The current board is the starting state of the next level; build a new plan for this layout rather than continuing the previous level's action sequence. Start from the mechanics you established on the previous level; do not rediscover them without reason. Inspect the new board for unfamiliar elements, changed arrangements, or interactions your previous understanding does not explain. New elements often introduce mechanics needed to solve this level, so prioritize small, informative tests when their behavior is unclear. Reassess the goal: does the previous objective still apply, now requiring the new mechanics, or does the evidence suggest a different objective?"""


def interior_changed(before, after, border=BORDER):
    """True if any cell inside the border differs; a shape change counts as change."""
    if len(before) != len(after) or any(len(a) != len(b) for a, b in zip(before, after)):
        return True
    return any(before[r][c] != after[r][c]
               for r in range(border, len(before) - border)
               for c in range(border, len(before[r]) - border))


def observation_notes():
    """Event guidance for the decision prompt, from the last public transition."""
    global BATCH_STOP
    notes = []
    if BATCH_STOP is not None:
        offset, reason = BATCH_STOP
        BATCH_STOP = None
        notes.append(f"Your queued batch stopped before step {offset + 1}: {reason}. "
                     "The remaining queued actions were not executed.")
    if OBSERVATION is not None and OBSERVATION.state == "GAME_OVER":
        notes.append(_GAME_OVER)
    elif OBSERVATION is not None and PREVIOUS is not None:
        if OBSERVATION.levels_completed > PREVIOUS.levels_completed:
            notes.append(_LEVEL_UP)
        elif (PREVIOUS.grid != OBSERVATION.grid
              and not interior_changed(PREVIOUS.grid, OBSERVATION.grid)):
            notes.append("The previous action changed only the 4-cell border (e.g. a budget bar); "
                         "nothing in the interior board changed.")
    return "".join("\n\n" + note for note in notes)


def unbounded_render():
    # This removes NOOA's separate half-window renderer eviction. The explicit
    # provider projection below owns the actual bounded input, including tools/images.
    return TruncationConfig(max_context_tokens=10**9, response_reserve_tokens=0,
        event_format=FormatConfig(max_string=96*1024, max_length=200, max_depth=5))


def actor_images(agent=None, turn=None):
    if OBSERVATION is None:
        return []
    def frame(grid):
        return Frame(grid=tuple(tuple(row) for row in grid), step=0, level=0)
    images = [{"type": "image_url", "image_url": {
        "url": frame_to_png_data_url(frame(OBSERVATION.grid), upscale=10)}}]
    if PREVIOUS is not None and PREVIOUS.grid != OBSERVATION.grid:
        diff = diff_image_part(frame(PREVIOUS.grid), frame(OBSERVATION.grid))
        if diff is not None:
            images.append(diff)
    if DEATH is not None:
        images.append({"type": "image_url", "image_url": {
            "url": frame_to_png_data_url(frame(DEATH), upscale=10)}})
    return images


def digest(message):
    return hashlib.sha256(json.dumps(message, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ContextProjection:
    """Trim whole user-led exchanges in blocks; retain the chosen head thereafter."""
    def __init__(self, window=128*1024, drain=58*1024):
        self.window, self.drain = window, drain
        self.anchor = None
        self.chars_per_token = 3.0
        self.last_estimate = 0
        self.last_images = 0
        self.last_chars = 0
        self.compacted = {}
        self.last_compactions = []

    def estimate(self, messages, tools):
        text, images = _split_messages_for_estimate(messages)
        return _estimate_tokens({"messages": text, "tools": tools}, self.chars_per_token) + images + 1024

    def project(self, messages, tools=None, output=12*1024):
        tools = [{"name": t.name, "description": t.description, "parameters": t.get_parameter_schema()}
                 if hasattr(t, "get_parameter_schema") else t for t in (tools or [])]
        prefix = 0
        while prefix < len(messages) and messages[prefix].get("role") == "system":
            prefix += 1
        head = prefix
        if self.anchor is not None:
            matches = [i for i in range(prefix, len(messages)) if digest(messages[i]) == self.anchor]
            if matches:
                head = matches[0]
            else:
                # Method/formatter changed the exchange identities. Never silently
                # claim stable cache reuse; start a new projection and record it.
                self.anchor = None
                LEDGER["projection_resets"] = LEDGER.get("projection_resets", 0) + 1
        messages = [self.compacted.get(digest(m),m) for m in messages]
        self.last_compactions = []
        result = messages[:prefix] + messages[head:]
        budget = self.window - output
        if budget <= 0:
            raise ValueError("output reserve leaves no input budget")
        estimate = self.estimate(result, tools)
        if estimate > budget:
            target = max(1, budget - min(self.drain, budget//2))
            # Start only at a user message, keeping assistant/tool pairs together.
            boundaries = [i for i in range(head+1, len(messages))
                          if messages[i].get("role") == "user"]
            for next_head in boundaries:
                candidate = messages[:prefix] + messages[next_head:]
                if self.estimate(candidate, tools) <= target:
                    result, head = candidate, next_head
                    break
            else:
                if boundaries:
                    head = boundaries[-1]
                    result = messages[:prefix] + messages[head:]
            if self.estimate(result, tools) > budget:
                # Long tool loops contain only one user boundary. Preserve the
                # observation, arguments and matching result IDs, shedding old
                # reasoning first, then older tool bodies. The latest result
                # stays intact. Full originals remain in canonical raw traces.
                latest_tool = max((i for i,m in enumerate(result) if m.get('role')=='tool'),default=-1)
                candidates=[(i,'reasoning_content') for i,m in enumerate(result)
                            if m.get('role')=='assistant' and m.get('reasoning_content')]
                candidates += [(i,'content') for i,m in enumerate(result)
                               if m.get('role')=='tool' and i!=latest_tool]
                for i,field in candidates:
                    original=result[i]; replacement=dict(original)
                    if field=='content':
                        replacement[field]='[Older tool output omitted to fit context; rerun a targeted computation or read the workspace.]'
                    else:
                        replacement.pop(field,None)
                    self.compacted[digest(original)]=replacement
                    result[i]=replacement
                    self.last_compactions.append({'message_sha256':digest(original),'field':field})
                    if self.estimate(result,tools)<=budget: break
                if self.estimate(result, tools) > budget:
                    raise ValueError("context window exceeded by fixed prompt or newest required result")
            self.anchor = digest(messages[head])
            LEDGER["evictions"] += 1
        self.last_estimate = self.estimate(result, tools)
        text, self.last_images = _split_messages_for_estimate(result)
        self.last_chars = len(json.dumps({"messages": text, "tools": tools}, ensure_ascii=False, default=str))
        return result

    def calibrate(self, prompt_tokens):
        text_tokens = max(1, int(prompt_tokens) - self.last_images)
        if prompt_tokens and self.last_chars:
            self.chars_per_token = min(3.3, max(1.0, self.last_chars/text_tokens))


class HybridLLMTrace(LLMTrace):
    """Install the projection BEFORE canonical exact-request checkpoints."""
    def attach(self, event_manager, *, source):
        super().attach(event_manager, source=source)
        projection = ContextProjection()
        self._projections = getattr(self, "_projections", {})
        self._projections[id(event_manager)] = projection
        path = self._path.with_name("context-projection.jsonl") if self._path else None

        async def project(ctx, nxt):
            # Persist an inspectable commitment to the full rendered source as
            # well as the actual projected request captured downstream.
            full_digest = digest({"messages": ctx.messages})
            # NOOA renders reasoning under `reasoning`; the donor's Qwen chat
            # template consumes `reasoning_content` when preserve_thinking is on.
            messages = []
            for original in ctx.messages:
                message = dict(original)
                if message.get("role") == "assistant" and message.get("reasoning"):
                    message["reasoning_content"] = message.pop("reasoning")
                messages.append(message)
            ctx.messages = projection.project(messages, ctx.params.get("tools"),
                                              ctx.params.get("max_tokens", 12*1024))
            if path:
                with path.open("a") as stream:
                    stream.write(json.dumps({"source": source, "full_render_sha256": full_digest,
                        "projected_sha256": digest({"messages": ctx.messages}),
                        "estimated_input_tokens": projection.last_estimate,
                        "compactions": projection.last_compactions,
                        "evictions": LEDGER["evictions"]}) + "\n")
            result = await nxt(ctx)
            usage = getattr(result.response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens", 0)
            projection.calibrate(prompt_tokens or 0)
            return result
        event_manager.intercept("llm_call", project)

    def _begin(self, event, source):
        super()._begin(event, source)
        source = self._source_override or source
        LEDGER["calls"] += 1
        role = LEDGER["roles"].setdefault(source, {"calls": 0, "completion_tokens": 0})
        role["calls"] += 1
        role["completion_tokens"] += event.completion_tokens or 0
        for key in ("completion_tokens", "prompt_tokens", "cached_tokens"):
            LEDGER[key] += getattr(event, key, 0) or 0
