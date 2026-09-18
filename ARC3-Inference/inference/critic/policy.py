"""The critic policy: a bounded, investigation-only model loop that turns a
`CriticPacket` into a `CriticReport`.

Ported from the Tycho/NOOA trigger builder (solvers/tycho_nooa/builder.py): the
investigator-not-actor role, the scratchpad carried across firings, the
host-authoritative budget notice that moves the model from *investigate* to
*commit*, the forced final report when the budget is spent, and the fallback
report when the model does not produce one. Not ported: world_model.py, the
verifier, and the workspace file tools -- Duck has no executable world model,
so the critic investigates with Duck's own sandbox instead.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

from inference.agent.action_names import to_model_actions
from inference.critic.packet import CriticPacket
from inference.critic.report import CriticReport
from inference.critic.sandbox import run_investigation_python
from inference.utils.grid_utils import ARC_COLOR_LEGEND
from inference.utils.openai_compat import build_chat_payload, build_headers

REPORT_FIELDS = (
    "confidence",
    "model",
    "objective",
    "plan",
    "recommended_action",
    "subgoal",
    "note",
    "scratchpad",
)

SYSTEM_PROMPT = (
    "You are the critic for an agent playing an unfamiliar 64x64 grid game. You do not take game "
    "actions and you cannot: `action(...)` is refused in your Python tool. Your only job is to "
    "investigate the evidence the actor has already gathered, decide whether the actor's beliefs "
    "and plan survive that evidence, and hand back a compact advisory report the actor reads on "
    "its next turn.\n\n"
    "You are invoked at level start, when the actor asks, or when the actor is stalled (actions "
    "that do not change the board, or many actions without level progress). Treat the actor's "
    "beliefs, when present, as a starting hypothesis; the observed transitions are authoritative. "
    "When the actor's beliefs are empty, reconstruct them from the transitions and from what the "
    "actor's code was trying to do.\n\n"
    f"Color legend: {ARC_COLOR_LEGEND}.\n\n"
    "Your Python tool is Duck's own inspection sandbox: `current_frame`, `previous_frame`, "
    "`history`, `transitions`, `last_transition`, `valid_actions`, `last_action_result` are "
    "preloaded; each frame exposes only `.ascii`, `.segmentation`, `.step`, `.level`, `.shape`; "
    "each history entry exposes only `.action` and `.frame`; each transition exposes `.action`, "
    "`.before_frame`, `.after_frame`, `.result`. `segmentation` returns "
    "`{'nodes': [...], 'adjacency_list': [...]}`. Each node is one 4-connected same-color object "
    "with `id` (index, top-most-left-most order), `color` (ARC color character), `hash` (a "
    "position-invariant signature of color and shape), `pixels` (an integer cell COUNT, not a "
    "list), `boundary` (clockwise outer-perimeter corner points as `[row, col]`), and `children` "
    "(ids of objects fully enclosed by this one). `adjacency_list` is `[i, j]` id pairs that "
    "share an edge. The raw numeric grid is not exposed. Every call starts fresh; only these "
    "stdlib modules import: bisect, collections, copy, fractions, functools, heapq, itertools, "
    "json, math, operator, random, re, statistics, string. Print compact summaries -- diffs, "
    "counts, object lists, tiny crops -- never whole boards.\n\n"
    "A scratchpad line from your earlier invocations in this game is included when it exists; "
    "read it so insight compounds instead of restarting. Keep competing objective hypotheses "
    "open when the evidence does not separate them, and name the single action that would.\n\n"
    "Use investigation calls to understand first, then commit. When the budget notice says "
    "commit, stop investigating and write the report.\n\n"
    "Report format (plain labeled lines, exactly these labels, one line each, no markdown):\n"
    "confidence: none|low|medium|high\n"
    "model: <one line: how actions change the board, or 'none -- insufficient evidence'>\n"
    "objective: <committed hypothesis> | open: <h1 vs h2> | unknown\n"
    "plan: <short route in actions, or none>\n"
    "recommended_action: <one legal action from the list: UP, DOWN, LEFT, RIGHT, SPACE, "
    "MOUSE(row=r, col=c); a short comma-separated sequence is allowed; or none>\n"
    "subgoal: <one grounded subgoal, or none>\n"
    "note: <one line for the actor -- especially what would disambiguate open hypotheses>\n"
    "scratchpad: <one line for your next invocation: confirmed, ruled out, next probe>"
)

FALLBACK_REPORT_TEXT = (
    "confidence: none\n"
    "model: none -- critic did not finish within its budget\n"
    "objective: unknown\n"
    "plan: none\n"
    "recommended_action: none\n"
    "subgoal: none\n"
    "note: critic budget exhausted before a report; keep exploring\n"
    "scratchpad: no new findings this invocation"
)

REPORT_RETRY_REQUEST = (
    "That reply contained no report (a tool call or nothing). Tools are closed for this "
    "invocation. Reply with ONLY the eight labeled lines -- confidence, model, objective, plan, "
    "recommended_action, subgoal, note, scratchpad -- using what you already know. If you "
    "know little, say so in the lines; do not investigate further."
)

FINAL_REPORT_REQUEST = (
    "Budget reached. Do NOT call any more tools. Right now, in your reply, give the report in "
    "the required labeled format (confidence/model/objective/plan/recommended_action/subgoal/"
    "note/scratchpad) summarizing what you learned -- even if it is 'still exploring'. Keep any "
    "reasoning brief; the labeled lines are the deliverable. Be "
    "concrete about what you DID establish and the action that would disambiguate open "
    "hypotheses, so the actor can use it."
)

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Run one ephemeral investigation snippet against the frozen game state. Globals: "
            "`current_frame`, `previous_frame`, `history`, `transitions`, `last_transition`, "
            "`valid_actions`, `last_action_result`. Frames expose `.ascii`, `.segmentation`, "
            "`.step`, `.level`, `.shape`. `action(...)` is refused. Print compact output or "
            "assign to `result`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to run; not saved across calls."},
            },
            "required": ["code"],
        },
    },
}

_LABEL_RE = re.compile(r"^\s*([a-z_]+)\s*:\s*(.*?)\s*$")
_MOUSE_RE = re.compile(r"MOUSE\s*\(\s*(?:row\s*=\s*)?(\d+)\s*,\s*(?:col\s*=\s*)?(\d+)\s*\)", re.IGNORECASE)
_SIMPLE_ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "SPACE")


@dataclass(frozen=True)
class CriticModelConfig:
    provider: str
    base_url: str
    model_id: str
    api_key: str = ""
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    thinking: bool = True
    request_timeout_seconds: float | None = 120.0
    referer: str = ""
    title: str = "ARC3 Duck Critic"


@dataclass(frozen=True)
class CriticPolicyConfig:
    """The budget is the design variable: one Duck turn costs ~46 s on Kaggle at
    10-way concurrency, so every critic firing is priced against that.

    Investigation calls per firing = max_tool_steps - commit_calls_remaining (the
    last call is the report). Each request's timeout is bounded by the seconds
    left plus `request_grace_seconds`, so one runaway reasoning call cannot blow
    the firing past its budget by more than that grace.
    """

    max_tool_steps: int = 2
    max_seconds: float = 90.0
    max_output_tokens: int | None = 4096
    tool_timeout_seconds: int = 20
    tool_output_chars: int = 4000
    commit_calls_remaining: int = 1
    commit_seconds_remaining: float = 30.0
    final_output_multiplier: int = 2
    report_retries: int = 1
    request_grace_seconds: float = 20.0
    max_transitions_in_kickoff: int = 12


@dataclass
class CriticFiring:
    """Everything a gate or an uptake analysis needs about one invocation."""

    packet_id: str
    report: CriticReport
    report_text: str
    scratchpad: str
    seconds: float
    model_calls: int
    tool_calls: int
    tool_errors: int
    refused_action_calls: int
    prompt_tokens: int
    completion_tokens: int
    parsed: bool
    fallback: bool
    forced_final: bool
    recommended_legal: bool | None
    error: str = ""
    transcript: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CriticNotes:
    """Scratchpad carried across firings within one game (Tycho's notes/world_model.md)."""

    lines: list[str] = field(default_factory=list)
    max_lines: int = 12

    def add(self, level: int, action_num: int, text: str) -> None:
        text = " ".join(str(text or "").split())
        if not text:
            return
        self.lines.append(f"[L{level} a{action_num}] {text}")
        del self.lines[: max(0, len(self.lines) - self.max_lines)]

    def render(self) -> str:
        return "\n".join(self.lines)


def parse_report_text(text: str) -> dict[str, str]:
    """Pull the labeled lines out of a reply; tolerant of prose before/after and of
    `**label**:` markdown. Later duplicates win."""
    fields: dict[str, str] = {}
    for raw in str(text or "").splitlines():
        line = raw.replace("**", "").strip().lstrip("-• ").strip()
        match = _LABEL_RE.match(line)
        if not match:
            continue
        key, value = match.group(1).lower(), match.group(2)
        if key in REPORT_FIELDS:
            fields[key] = value.strip()
    return fields


def parse_recommended_actions(text: str) -> list[dict[str, Any]]:
    """`UP, MOUSE(row=3, col=7)` -> Duck `action(...)` dicts; `none` -> []."""
    value = str(text or "").strip()
    if not value or value.lower().startswith("none"):
        return []
    actions: list[dict[str, Any]] = []
    cursor = 0
    for match in _MOUSE_RE.finditer(value):
        for token in re.split(r"[,\s;]+", value[cursor : match.start()]):
            if token.upper() in _SIMPLE_ACTIONS:
                actions.append({"action": token.upper()})
        actions.append({"action": "MOUSE", "row": int(match.group(1)), "col": int(match.group(2))})
        cursor = match.end()
    for token in re.split(r"[,\s;]+", value[cursor:]):
        if token.upper() in _SIMPLE_ACTIONS:
            actions.append({"action": token.upper()})
    return actions


def report_from_fields(packet_id: str, fields: dict[str, str]) -> CriticReport:
    objective = fields.get("objective", "")
    open_questions: list[str] = []
    if "open:" in objective:
        open_questions.append(objective.split("open:", 1)[1].strip())
    summary_parts = [
        f"{key}: {fields[key]}"
        for key in ("model", "objective", "plan", "subgoal", "note")
        if fields.get(key)
    ]
    return CriticReport(
        packet_id=packet_id,
        summary=" | ".join(summary_parts),
        recommended_actions=tuple(parse_recommended_actions(fields.get("recommended_action", ""))),
        confidence=fields.get("confidence", "").split()[0].lower() if fields.get("confidence") else "",
        open_questions=tuple(open_questions),
        evidence=(),
    )


def _legal(actions: list[dict[str, Any]], valid_actions: tuple[str, ...]) -> bool | None:
    if not actions:
        return None
    legal = {name.upper() for name in to_model_actions(list(valid_actions))} | {name.upper() for name in valid_actions}
    return all(str(item.get("action", "")).upper() in legal for item in actions)


def _compact_transitions(packet: CriticPacket, limit: int) -> list[str]:
    lines: list[str] = []
    for record in packet.transitions[-limit:]:
        flags = []
        if record.get("board_changed"):
            flags.append("board_changed")
        if record.get("level_completed"):
            flags.append("LEVEL_COMPLETED")
        if record.get("game_over"):
            flags.append("GAME_OVER")
        lines.append(f"  a{record.get('action_num')}: {record.get('action')} -> {', '.join(flags) or 'no change'}")
    return lines


def render_kickoff(packet: CriticPacket, notes: CriticNotes, config: CriticPolicyConfig) -> str:
    beliefs = {key: value for key, value in packet.actor_beliefs.items() if value}
    legal = to_model_actions(list(packet.valid_actions))
    lines = [
        f"You were invoked at level {packet.level}, after action {packet.action_num}. "
        f"Trigger: {packet.trigger_cause} -- {packet.trigger_reason}",
        f"Current legal actions for the actor: {', '.join(legal) or 'none'}.",
    ]
    if beliefs:
        lines.append("Actor's carried beliefs (starting hypothesis, not ground truth):")
        lines.extend(f"  {key}: {value}" for key, value in beliefs.items())
    else:
        lines.append("Actor's carried beliefs: none recorded -- reconstruct from the transitions.")
    if packet.step_summary:
        lines.append(f"Actor's last step summary: {json.dumps(packet.step_summary, sort_keys=True)}")
    if packet.last_action_result:
        keep = {k: packet.last_action_result.get(k) for k in ("board_changed", "level_completed", "game_over", "stop_reason", "executed_count") if k in packet.last_action_result}
        lines.append(f"Last action result: {json.dumps(keep, sort_keys=True)}")
    lines.append(
        f"Recent transitions ({packet.transitions_total} total in this game; last {min(config.max_transitions_in_kickoff, len(packet.transitions))} shown):"
    )
    lines.extend(_compact_transitions(packet, config.max_transitions_in_kickoff) or ["  (none yet)"])
    scratch = notes.render()
    lines.append("Your scratchpad from earlier invocations in this game:")
    lines.append(scratch if scratch else "  (empty -- first invocation)")
    lines.append(
        "Inspect the evidence with the python tool, then give the report in the labeled format. "
        "The budget notice below is host-authoritative."
    )
    return "\n".join(lines)


def budget_notice(*, calls_remaining: int, seconds_remaining: float, config: CriticPolicyConfig) -> tuple[str, bool]:
    commit = calls_remaining <= config.commit_calls_remaining or seconds_remaining <= config.commit_seconds_remaining
    mode = "commit" if commit else "investigate"
    instruction = (
        "Do not call tools. Write the report now."
        if commit
        else "Investigate, but externalize one falsifiable hypothesis before commit mode."
    )
    return (
        "CRITIC BUDGET (host-authoritative; supersedes earlier notices)\n"
        f"Investigation calls remaining (including this one): {max(0, calls_remaining)}\n"
        f"Time remaining: {max(0, int(seconds_remaining))} seconds\n"
        f"Mode: {mode}\n{instruction}"
    ), commit


class CriticPolicy:
    def __init__(
        self,
        model: CriticModelConfig,
        config: CriticPolicyConfig | None = None,
        *,
        post: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self._model = model
        self._config = config or CriticPolicyConfig()
        self._post = post or self._http_post
        self._post_is_http = post is None

    @property
    def config(self) -> CriticPolicyConfig:
        return self._config

    def _http_post(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        response = requests.post(
            f"{self._model.base_url.rstrip('/')}/chat/completions",
            headers=build_headers(
                provider=self._model.provider,
                api_key=self._model.api_key,
                referer=self._model.referer,
                title=self._model.title,
            ),
            json=payload,
            timeout=timeout if timeout is not None else self._model.request_timeout_seconds,
        )
        if response.status_code >= 400:
            raise requests.RequestException(f"{response.status_code} | {response.text.strip()[:500]}")
        return response.json()

    def _chat(self, messages: list[dict[str, Any]], *, tools: bool, final: bool = False, seconds_left: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        max_tokens = self._config.max_output_tokens
        if final and max_tokens is not None:
            # The report call must survive a long reasoning prefix; give it headroom.
            max_tokens = max_tokens * self._config.final_output_multiplier
        payload = build_chat_payload(
            provider=self._model.provider,
            model=self._model.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self._model.temperature,
            top_p=self._model.top_p,
            top_k=self._model.top_k,
            thinking=self._model.thinking,
            tools=[PYTHON_TOOL] if tools else None,
            tool_choice="auto" if tools else None,
        )
        timeout = self._model.request_timeout_seconds
        if seconds_left is not None:
            bounded = max(5.0, seconds_left + self._config.request_grace_seconds)
            timeout = bounded if timeout is None else min(timeout, bounded)
        body = self._http_post(payload, timeout) if self._post_is_http else self._post(payload)
        choices = body.get("choices") or []
        if not choices:
            raise requests.RequestException("server returned no choices")
        return choices[0].get("message", {}) or {}, body.get("usage") or {}

    def run(self, packet: CriticPacket, notes: CriticNotes) -> CriticFiring:
        config = self._config
        started = time.monotonic()
        transcript: list[dict[str, Any]] = []
        prompt_tokens = completion_tokens = 0
        model_calls = tool_calls = tool_errors = refused = 0
        forced_final = False
        report_retries_left = config.report_retries
        error = ""
        final_text = ""

        def remaining() -> float:
            return config.max_seconds - (time.monotonic() - started)

        kickoff = render_kickoff(packet, notes, config)
        notice, commit = budget_notice(calls_remaining=config.max_tool_steps, seconds_remaining=remaining(), config=config)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{kickoff}\n\n{notice}"},
        ]
        transcript.append({"role": "user", "content": messages[-1]["content"]})

        try:
            while True:
                message, usage = self._chat(messages, tools=not commit, final=commit, seconds_left=remaining())
                model_calls += 1
                prompt_tokens += int(usage.get("prompt_tokens") or 0)
                completion_tokens += int(usage.get("completion_tokens") or 0)
                content = _text(message.get("content"))
                reasoning = _text(message.get("reasoning") or message.get("reasoning_content"))
                calls = message.get("tool_calls") or []
                transcript.append({"role": "assistant", "content": content, "reasoning": reasoning[:2000], "tool_calls": calls})
                if not calls or commit:
                    if parse_report_text(content):
                        final_text = content
                        break
                    if parse_report_text(reasoning):
                        # Thinking models sometimes leave the report inside the reasoning channel.
                        final_text = reasoning
                        break
                    if forced_final:
                        if report_retries_left <= 0 or remaining() <= 0:
                            final_text = content
                            break
                        # Same failure Duck's actor loop retries once: markup or nothing where
                        # the report should be. One sterner ask, tools still closed.
                        report_retries_left -= 1
                        messages.append({"role": "assistant", "content": content or None})
                        messages.append({"role": "user", "content": REPORT_RETRY_REQUEST})
                        transcript.append({"role": "user", "content": REPORT_RETRY_REQUEST})
                        continue
                    # No report and no tool call: force the report once.
                    forced_final = True
                    commit = True
                    assistant: dict[str, Any] = {"role": "assistant", "content": content or None}
                    messages.append(assistant)
                    messages.append({"role": "user", "content": FINAL_REPORT_REQUEST})
                    transcript.append({"role": "user", "content": FINAL_REPORT_REQUEST})
                    continue

                assistant = {"role": "assistant", "content": content or None, "tool_calls": calls}
                messages.append(assistant)
                for call in calls:
                    function = call.get("function", {}) if isinstance(call, dict) else {}
                    raw_args = function.get("arguments", "{}")
                    try:
                        arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
                    except json.JSONDecodeError:
                        arguments = {}
                    code = str(arguments.get("code", "") or "")
                    tool_calls += 1
                    outcome = run_investigation_python(packet, code, timeout_seconds=config.tool_timeout_seconds)
                    refused += int(outcome.get("refused_action_calls") or 0)
                    if outcome.get("error"):
                        tool_errors += 1
                    rendered = _render_tool_outcome(outcome, config.tool_output_chars)
                    transcript.append({"role": "tool", "code": code, "content": rendered})
                    messages.append({"role": "tool", "tool_call_id": str(call.get("id", "")), "content": rendered})
                calls_left = config.max_tool_steps - tool_calls
                notice, commit = budget_notice(calls_remaining=calls_left, seconds_remaining=remaining(), config=config)
                if commit:
                    forced_final = True
                    messages.append({"role": "user", "content": f"{notice}\n\n{FINAL_REPORT_REQUEST}"})
                else:
                    messages.append({"role": "user", "content": notice})
                transcript.append({"role": "user", "content": messages[-1]["content"]})
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            error = f"{type(exc).__name__}: {str(exc)[:300]}"

        fields = parse_report_text(final_text)
        parsed = bool(fields.get("recommended_action") is not None or fields.get("note") or fields.get("model"))
        fallback = not parsed
        report_text = final_text if parsed else FALLBACK_REPORT_TEXT
        if fallback:
            fields = parse_report_text(FALLBACK_REPORT_TEXT)
        report = report_from_fields(packet.packet_id, fields)
        scratch = fields.get("scratchpad", "")
        if not fallback:
            notes.add(packet.level, packet.action_num, scratch)
        return CriticFiring(
            packet_id=packet.packet_id,
            report=report,
            report_text=report_text,
            scratchpad=scratch,
            seconds=round(time.monotonic() - started, 3),
            model_calls=model_calls,
            tool_calls=tool_calls,
            tool_errors=tool_errors,
            refused_action_calls=refused,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            parsed=parsed,
            fallback=fallback,
            forced_final=forced_final,
            recommended_legal=_legal(list(report.recommended_actions), packet.valid_actions),
            error=error,
            transcript=transcript,
        )


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(part.get("text", "")) for part in value if isinstance(part, dict))
    return "" if value is None else str(value)


def _render_tool_outcome(outcome: dict[str, Any], max_chars: int) -> str:
    parts: list[str] = []
    stdout = str(outcome.get("stdout", "") or "").strip()
    if stdout:
        parts.append(stdout)
    if outcome.get("result") is not None:
        parts.append(f"result = {json.dumps(outcome.get('result'), sort_keys=True, default=str)}")
    if outcome.get("error"):
        parts.append(f"error: {outcome.get('error')}")
    if outcome.get("refused_action_calls"):
        parts.append("note: action() is refused in the critic sandbox; this is investigation only.")
    text = "\n".join(parts) or "(no output)"
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated to {max_chars} chars]"
    return text


__all__ = [
    "FALLBACK_REPORT_TEXT",
    "REPORT_FIELDS",
    "REPORT_RETRY_REQUEST",
    "SYSTEM_PROMPT",
    "CriticFiring",
    "CriticModelConfig",
    "CriticNotes",
    "CriticPolicy",
    "CriticPolicyConfig",
    "budget_notice",
    "parse_recommended_actions",
    "parse_report_text",
    "render_kickoff",
    "report_from_fields",
]
