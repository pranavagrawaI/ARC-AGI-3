"""Run one persistent NOOA game-solving conversation over JSONL IPC."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import hashlib
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from nooa import Agent, hidden, strategy
from nooa.config import CodeActConfig, PredictConfig
from nooa.context_blocks import (
    DynamicContext,
    ResultStatus,
    ToolCallEvent,
    ToolResult,
)
from nooa.errors import GenerationError
from nooa.events import Feedback
from nooa.media import Image as TransportImage
from nooa.runtime.media_capture import show
from nooa.runtime.restrictions import DEFAULT_BLOCKED_MODULES, RestrictionsConfig
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.runtime.sandbox.errors import SandboxError
from nooa.runtime.sandbox.executor import check_enforceable
from nooa.strategies import CodeActStrategy, PredictStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, RetryConfig, Tool, ToolCall
from nooa.unifiedllm.registry import get_llm_client
from PIL import Image as PILImage
from pydantic import BaseModel, ConfigDict, model_validator

from guided_first_use import GUIDED_FIRST_USE_SCHEMA, render_guided_first_use
from theory_guidance import (
    COMPLETE_PRACTICE_EXAMPLES,
    GUIDE_SCHEMA,
    practice_guide_sha256,
    render_practice_guide,
)
from theory_workbench import (
    MAX_RENDER_BYTES,
    TheoryWorkbenchViewBuilder,
    theory_snapshot_from_lane_view,
)
from world_model_lane import E0, EditProvenance, ExecutableTheoryLane, ModelEdit

_PROTOCOL_FD = os.dup(sys.stdout.fileno())

DEFAULT_MODEL_PROFILE = "muse-glimmer-30b"
MODEL_PROFILES = {
    DEFAULT_MODEL_PROFILE: {
        "model": "meta/muse-glimmer-30b",
        "model_route": "openai/meta/muse-glimmer-30b",
        "context_window": 131_072,
        "provider_route": "qualified-pool-v18",
        "provider_routes": [
            "deepinfra-bf16",
            "together",
            "fireworks",
            "phala",
        ],
        "single_provider": {
            "order": ["deepinfra/bf16"],
            "only": ["deepinfra/bf16"],
            "allow_fallbacks": False,
            "require_parameters": False,
            "quantizations": ["bf16"],
        },
    },
    "gpt-5.6-luna": {
        "model": "openai/gpt-5.6-luna",
        "model_route": "openai/openai/gpt-5.6-luna",
        "context_window": 1_050_000,
        "provider_route": "openai",
        "provider_routes": ["openai"],
        "single_provider": {
            "order": ["openai"],
            "only": ["openai"],
            "allow_fallbacks": False,
            "require_parameters": False,
        },
    },
}
MODEL_PROFILE = os.environ.get("NOOA_0159_MODEL_PROFILE", DEFAULT_MODEL_PROFILE)
if MODEL_PROFILE not in MODEL_PROFILES:
    raise ValueError(f"unknown NOOA_0159_MODEL_PROFILE {MODEL_PROFILE!r}")
MODEL_CONFIG = MODEL_PROFILES[MODEL_PROFILE]
MODEL = MODEL_CONFIG["model"]
MODEL_ROUTE = MODEL_CONFIG["model_route"]
API_BASE = "https://openrouter.ai/api/v1"
MODEL_CONTEXT_WINDOW = MODEL_CONFIG["context_window"]
MAX_TOKENS = 8_192
PROVIDER_ATTEMPT_TIMEOUT_SECONDS = 90.0
SINGLE_PROVIDER = MODEL_CONFIG["single_provider"]
QUALIFIED_PROVIDERS = (
    ("deepinfra-bf16", "deepinfra/bf16", ["bf16"]),
    ("together", "together", None),
    ("fireworks", "fireworks", None),
    ("phala", "phala", None),
)
RESPONSE_PREFIX = "0159_RESPONSE\t"
READY_PREFIX = "0159_READY\t"
CURRENT_SEQUENCE: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "current_sequence", default=None
)
ANALYSIS_MAX_ITERATIONS = 11
COMMIT_MAX_ATTEMPTS = 3
MAX_PROVIDER_CALLS_PER_TURN = ANALYSIS_MAX_ITERATIONS + COMMIT_MAX_ATTEMPTS
NOTE_NAME = re.compile(r"^[a-z0-9_-]+\.md$")
MAX_NOTE_CHARS = 24_000
MAX_KNOWLEDGE_CHARS = 72_000
WORLD_MODEL_VARIANT = "codeact_notes_compact_world_model"
COMPACT_VARIANTS = frozenset({"codeact_notes_compact", WORLD_MODEL_VARIANT})
KNOWLEDGE_VARIANTS = frozenset({"codeact_notes", *COMPACT_VARIANTS})
ActionName = Literal[
    "RESET", "UP", "DOWN", "LEFT", "RIGHT", "SPACEBAR", "CLICK", "UNDO"
]

# Exact ARC-AGI-3 public rendering palette from arc_agi.rendering.COLOR_MAP.
# The worker cannot import arc_agi because it intentionally runs in the pinned
# Python 3.12 NOOA environment, separate from the Python 3.14 game host.
ARC_PALETTE = (
    (255, 255, 255),
    (204, 204, 204),
    (153, 153, 153),
    (102, 102, 102),
    (51, 51, 51),
    (0, 0, 0),
    (229, 58, 163),
    (255, 123, 204),
    (249, 60, 49),
    (30, 147, 255),
    (136, 216, 241),
    (255, 220, 0),
    (255, 133, 27),
    (146, 18, 49),
    (79, 204, 48),
    (163, 86, 214),
)

_ANALYSIS_RESTRICTIONS = RestrictionsConfig(
    blocked_modules=DEFAULT_BLOCKED_MODULES
    | frozenset(
        {
            "aiohttp",
            "ctypes",
            "dbm",
            "glob",
            "http",
            "httpx",
            "importlib",
            "io",
            "linecache",
            "marshal",
            "multiprocessing",
            "os",
            "pathlib",
            "pickle",
            "requests",
            "runpy",
            "shelve",
            "shutil",
            "sqlite3",
            "sys",
            "tempfile",
            "urllib",
        }
    )
)

_CODEACT_CONFIG = CodeActConfig(
    max_iterations=ANALYSIS_MAX_ITERATIONS,
    max_retries=4,
    max_tool_calls=ANALYSIS_MAX_ITERATIONS,
    max_tokens=MAX_TOKENS,
    cell_timeout=20.0,
    restrictions=_ANALYSIS_RESTRICTIONS,
    execution_backend="sandbox",
    sandbox=SandboxConfig(
        filesystem=True,
        workspace=None,
        network=False,
        max_memory_mb=1_024,
        max_cpu_seconds=30,
        require=True,
    ),
)

_COMMIT_CONFIG = PredictConfig(
    max_retries=COMMIT_MAX_ATTEMPTS,
    max_tokens=MAX_TOKENS,
    output_serialization="tool_call",
)


class AnalysisBudgetExhausted(GenerationError):
    """The analysis allowance ended normally and the reserved commit call is due."""


class ImplicitContextArchivalDisabled(RuntimeError):
    """An oversized request was rejected without mutating active history."""


class History:
    """Read-only namespaced access to a host-frozen searchable evidence snapshot."""

    _METHODS = frozenset(
        {
            "get",
            "turns",
            "find",
            "turn",
            "note",
            "provider_episode",
            "around",
        }
    )

    def __init__(self, snapshot: dict[str, Any] | None = None):
        snapshot = snapshot or {
            "schema": "0159-history-snapshot-v1",
            "events": [],
            "turn_views": {},
            "provider_episodes": {},
            "source_manifest": {},
        }
        if snapshot.get("schema") != "0159-history-snapshot-v1":
            raise ValueError("history snapshot has the wrong schema")
        events = snapshot.get("events")
        if not isinstance(events, list):
            raise TypeError("history snapshot events must be a list")
        self._events = copy.deepcopy(events)
        self._by_id = {
            event["event_id"]: event
            for event in self._events
            if isinstance(event, dict) and isinstance(event.get("event_id"), str)
        }
        if len(self._by_id) != len(self._events):
            raise ValueError("history snapshot event IDs must be unique strings")
        self._turn_views = copy.deepcopy(snapshot.get("turn_views", {}))
        self._provider_episodes = copy.deepcopy(snapshot.get("provider_episodes", {}))

    def get(self, event_ids: list[str]) -> list[dict[str, Any]]:
        """Return exact redacted archive records by stable event ID."""
        if not isinstance(event_ids, list) or not all(
            isinstance(event_id, str) for event_id in event_ids
        ):
            raise TypeError("event_ids must be a list of strings")
        missing = [event_id for event_id in event_ids if event_id not in self._by_id]
        if missing:
            raise KeyError(f"unknown archive event ids: {missing}")
        return copy.deepcopy([self._by_id[event_id] for event_id in event_ids])

    def turns(
        self, start: int, end: int, streams: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return redacted archive records from a contiguous turn range."""
        if type(start) is not int or type(end) is not int or start > end:
            raise ValueError("start and end must be integers with start <= end")
        selected_streams = set(streams) if streams else None
        return copy.deepcopy(
            [
                event
                for event in self._events
                if type(event.get("turn")) is int
                and start <= event["turn"] <= end
                and (
                    selected_streams is None or event.get("stream") in selected_streams
                )
            ]
        )

    def find(
        self, query: str, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Deterministically search the redacted projection with bounded results."""
        terms = [term for term in str(query).casefold().split() if term]
        if not terms:
            raise ValueError("query must contain at least one term")
        filters = filters or {}
        if not isinstance(filters, dict):
            raise TypeError("filters must be a dictionary or None")
        streams = set(filters.get("streams") or []) or None
        kinds = set(filters.get("kinds") or []) or None
        start_turn = filters.get("start_turn")
        end_turn = filters.get("end_turn")
        limit = filters.get("limit", 20)
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("filters.limit must be an integer from 1 through 20")
        matches = []
        for event in self._events:
            turn = event.get("turn")
            if streams is not None and event.get("stream") not in streams:
                continue
            if kinds is not None and event.get("kind") not in kinds:
                continue
            if start_turn is not None and (type(turn) is not int or turn < start_turn):
                continue
            if end_turn is not None and (type(turn) is not int or turn > end_turn):
                continue
            text = json.dumps(
                event.get("record"), sort_keys=True, default=str
            ).casefold()
            if all(term in text for term in terms):
                matches.append(event)
            if len(matches) == limit:
                break
        return copy.deepcopy(matches)

    def turn(self, turn: int) -> dict[str, Any]:
        """Return the decision-shaped transcript and mechanical view of one turn."""
        try:
            return copy.deepcopy(self._turn_views[str(turn)])
        except KeyError as error:
            raise KeyError(f"no archived turn view for turn {turn}") from error

    def note(self, name: str) -> dict[str, Any]:
        """Return one exact durable note by name."""
        if not NOTE_NAME.fullmatch(name):
            raise ValueError("note name must match [a-z0-9_-]+.md")
        event_id = f"knowledge:{name}"
        try:
            event = self._by_id[event_id]
        except KeyError as error:
            raise KeyError(f"unknown durable note: {name}") from error
        return copy.deepcopy(
            {"schema": "0159-history-note-v1", **event, "truncated": False}
        )

    def provider_episode(self, sequence: int, phase: str) -> dict[str, Any]:
        """Return a bounded provider episode grouped by causal phase."""
        key = f"{sequence}:{phase}"
        try:
            return copy.deepcopy(self._provider_episodes[key])
        except KeyError as error:
            raise KeyError(
                f"no provider episode for sequence {sequence}, phase {phase!r}"
            ) from error

    def around(self, event_id: str, before: int = 1, after: int = 1) -> dict[str, Any]:
        """Return a bounded same-stream neighborhood around a stable event ID."""
        if event_id not in self._by_id:
            raise KeyError(f"unknown archive event id: {event_id}")
        if before < 0 or after < 0 or before + after > 12:
            raise ValueError(
                "before and after must be non-negative and total at most 12"
            )
        target = self._by_id[event_id]
        same_stream = [
            event
            for event in self._events
            if event.get("stream") == target.get("stream")
        ]
        index = same_stream.index(target)
        selected = same_stream[max(0, index - before) : index + after + 1]
        items = []
        for event in selected:
            item = copy.deepcopy(event)
            record = item.get("record")
            encoded = json.dumps(record, sort_keys=True, default=str)
            if len(encoded) > 6_000:
                item["record"] = {
                    "truncated": True,
                    "available_keys": sorted(record)
                    if isinstance(record, dict)
                    else [],
                    "preview": encoded[:2_000],
                }
            items.append(item)
        return {
            "schema": "0159-history-around-v1",
            "target": event_id,
            "stream": target.get("stream"),
            "items": items,
            "truncated": any(
                isinstance(item.get("record"), dict)
                and item["record"].get("truncated") is True
                for item in items
            ),
        }


class CommitActionStrategy(PredictStrategy):
    """Action-only inference with strict tool or JSON Decision transport."""

    def get_block_overrides(self) -> dict[str, str | DynamicContext | None]:
        return {
            "strategy_prompt": """## Action commitment

Commit exactly one Decision now. Prefer invoking the `return_result` tool with
`result={"action": "<available action>", "x": null, "y": null,
"explanation": "<concise reason>"}`. For CLICK, replace null with integer x/y
coordinates from 0 through 63.

Some providers may fail to encode a requested tool call. In that case, output
ONLY the exact JSON Decision object instead:
`{"action": "<available action>", "x": null, "y": null,
"explanation": "<concise reason>"}`.

Do not continue analysis. Do not emit prose, Markdown, code fences, or any
additional fields. Either exact encoding is validated identically; the host
never repairs, infers, or substitutes an action."""
        }

    @staticmethod
    def _tool() -> Tool:
        class ReturnResultArguments(BaseModel):
            model_config = ConfigDict(extra="forbid")

            result: Decision

        def return_result(result: Decision) -> Decision:
            """Submit the one legal environment action selected for this turn."""
            return result

        return Tool(
            name="return_result",
            description=(
                "Submit exactly one action for the current game turn. The action must be "
                "one of the supplied human-readable legal_actions. CLICK also "
                "requires x and y coordinates from 0 through 63."
            ),
            callable=return_result,
            parameters_model=ReturnResultArguments,
        )

    async def _call_llm_raw(
        self,
        runtime: Any,
        response_model: type[BaseModel],
    ) -> tuple[Any, str]:
        del response_model
        response, event_id = await runtime.generate(
            tools=[self._tool()],
            tool_choice={
                "type": "function",
                "function": {"name": "return_result"},
            },
            **self._build_sampling_kwargs(),
        )
        audit = getattr(runtime.agent, "_analysis_audit", None)
        if audit is not None:
            audit.write(
                "action_commitment_response",
                finish_reason=getattr(response, "finish_reason", None),
                content=getattr(response, "content", None),
                reasoning=getattr(response, "reasoning", None),
                tool_calls=[
                    {
                        "name": getattr(call, "name", None),
                        "arguments": getattr(call, "arguments", None),
                    }
                    for call in (getattr(response, "tool_calls", None) or [])
                ],
            )
        return response, event_id

    @staticmethod
    def _submitted_arguments(llm_response: Any) -> dict[str, Any]:
        calls = list(getattr(llm_response, "tool_calls", None) or [])
        if calls and (len(calls) != 1 or calls[0].name != "return_result"):
            names = [getattr(call, "name", None) for call in calls]
            raise ValueError(
                "Reserved commitment tool encoding must contain exactly one "
                f"return_result call; received {names}."
            )
        if calls:
            arguments = calls[0].arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise TypeError("return_result arguments must be a JSON object")
            if set(arguments) != {"result"} or not isinstance(
                arguments["result"], dict
            ):
                raise ValueError(
                    "return_result must wrap the Decision as "
                    '{"result":{"action":...,"explanation":"..."}}'
                )
            return arguments["result"]

        content = getattr(llm_response, "content", None)
        if isinstance(content, BaseModel):
            content = content.model_dump()
        elif isinstance(content, str):
            if not content.strip():
                raise ValueError(
                    "Reserved commitment contained neither return_result nor an exact "
                    "JSON Decision."
                )
            content = json.loads(content)
        if not isinstance(content, dict):
            raise TypeError(
                "Text fallback must be exactly one JSON Decision object with action, "
                "x, y, and explanation fields."
            )
        return content

    def _extract_raw_from_llm_response(self, llm_response: Any) -> str:
        try:
            return json.dumps(self._submitted_arguments(llm_response), sort_keys=True)
        except (TypeError, ValueError, json.JSONDecodeError):
            return super()._extract_raw_from_llm_response(llm_response) or ""

    async def _format_validation_error(
        self,
        error: Exception,
        return_type: Any,
        runtime: Any,
    ) -> str:
        del return_type, runtime
        return f"""ACTION COMMITMENT REJECTED

The response did not contain a valid Decision: {error}

Invoke `return_result` exactly once with:
`result={{"action": "<one supplied label>", "x": null, "y": null,
"explanation": "<concise reason>"}}`

If the provider cannot emit the tool call, output only this exact JSON object:
`{{"action": "<one supplied label>", "x": null, "y": null,
"explanation": "<concise reason>"}}`

For CLICK only, replace null with integer x and y coordinates from 0 through 63.

Do not continue analysis. Do not output prose, Markdown, code fences, or extra fields."""

    def _parse_llm_response(
        self,
        llm_response: Any,
        method_name: str,
    ) -> dict[str, Any]:
        del method_name
        return self._submitted_arguments(llm_response)

    def _replace_with_tool_call(
        self,
        runtime: Any,
        event_id: str,
        result: Any,
    ) -> None:
        llm_output = runtime.event_manager.get(event_id)
        reasoning = getattr(llm_output, "reasoning", None)
        runtime.event_manager.remove(event_id)
        tool_call_id = (
            f"return_result_{hashlib.sha256(event_id.encode()).hexdigest()[:8]}"
        )
        runtime.event_manager.add(
            ToolCallEvent(
                tool_call_id=tool_call_id,
                name="return_result",
                arguments={"result": self._jsonable(result)},
                reasoning=reasoning,
                result=ToolResult(
                    tool_call_id=tool_call_id,
                    content="Action submission accepted.",
                    result_status=ResultStatus.COMPLETE,
                ),
            )
        )


def _analysis_budget_feedback(used: int) -> str:
    remaining = max(0, ANALYSIS_MAX_ITERATIONS - used)
    lines = [
        "TURN COMPUTE BUDGET",
        f"Analysis calls used: {used} of {ANALYSIS_MAX_ITERATIONS}",
        f"Analysis calls remaining: {remaining}",
        (
            "One action-only commitment phase is reserved, with up to "
            f"{COMMIT_MAX_ATTEMPTS} validation attempts."
        ),
        "You may submit return_result(action=..., x=..., y=..., explanation=...) now.",
    ]
    if remaining == 0:
        lines.append(
            "The analysis allowance is exhausted. The next inference is the "
            "reserved action-commitment inference."
        )
    elif remaining == 1:
        lines.append(
            "This is your final remaining analysis call; after it, only action "
            "commitment will be available."
        )
    elif remaining == 2:
        lines.append("Only two analysis calls remain; consolidate toward an action.")
    elif remaining == 3:
        lines.append("Choose only analysis that could change the next action.")
    elif remaining <= 5:
        lines.append(
            "Begin consolidating; avoid investigations that cannot finish in budget."
        )
    return "\n".join(lines)


class ReadOnlyTrajectoryCodeActStrategy(CodeActStrategy):
    """Expose only the analysis inputs and NumPy, not the worker module."""

    def get_block_overrides(self) -> dict[str, str | DynamicContext | None]:
        return {
            "strategy_prompt": f"""## Strategy

Use `execute_python(code)` to inspect the supplied image and trajectory with
ordinary Python and NumPy. Variables persist between analysis cells in this
decision turn. You have up to {ANALYSIS_MAX_ITERATIONS} analysis calls.
You also have one reserved action-only commitment inference. Finish by calling
`return_result(action=..., x=..., y=..., explanation=...)`,
either directly or from a cell. You may analyze repeatedly, but you must return
exactly one legal environment action. Plain text does not finish the turn.""",
            "sandbox": """## Analysis boundary

Code runs in a fresh, network-disabled, filesystem-confined process with a
20-second cell limit, 30 CPU seconds, and 1024 MB memory. It cannot execute an
environment action or mutate host evidence.""",
            "execution_context": """## Execution context

The current `image`, immutable-copy `trajectory`, `legal_actions`,
`previous_action`, `turn`, and `levels_completed` are preloaded. `image` is the
settled public 64x64 palette-mode PIL image. Use `image.size`, `image.crop(...)`,
or `np.asarray(image)[y, x]` for exact cell computation, and `show(image)` for
visual inspection. NumPy is available as `np`. Use
`return_result(action=..., x=..., y=..., explanation=...)` to finish. Use null
coordinates for every action except CLICK.""",
            "self": None,
        }

    def get_static_block_keys(self) -> set[str]:
        return {"strategy_prompt", "sandbox", "execution_context"}

    def get_block_order(self) -> list[str] | None:
        return ["system_prompt", "strategy_prompt", "sandbox", "execution_context"]

    def _extract_module_context(
        self, agent_module: Any, agent: Any | None = None
    ) -> dict[str, Any]:
        del agent_module, agent
        return {"np": np}

    async def _run_generation(
        self,
        runtime: Any,
        call: Any,
        session_holder: dict[str, Any],
    ) -> Any:
        try:
            return await super()._run_generation(runtime, call, session_holder)
        except GenerationError as error:
            session = session_holder.get("session")
            if (
                session is not None
                and session.iteration >= ANALYSIS_MAX_ITERATIONS
                and session.error_count < session.max_retries
            ):
                raise AnalysisBudgetExhausted(
                    "Analysis budget exhausted; reserved action commitment is due."
                ) from error
            raise

    async def _process_tool_calls(
        self,
        tool_calls: list[Any],
        runtime: Any,
        builtins: dict[str, Any],
        session: Any,
        call: Any,
        return_type: Any,
        event_id: str,
    ) -> Any:
        result = await super()._process_tool_calls(
            tool_calls,
            runtime,
            builtins,
            session,
            call,
            return_type,
            event_id,
        )
        if not result.completed:
            remaining = max(0, ANALYSIS_MAX_ITERATIONS - session.iteration)
            runtime.event_manager.add(
                Feedback(content=_analysis_budget_feedback(session.iteration))
            )
            audit = getattr(runtime.agent, "_analysis_audit", None)
            if audit is not None:
                audit.write(
                    "analysis_budget_updated",
                    analysis_calls_used=session.iteration,
                    analysis_calls_remaining=remaining,
                    commit_call_reserved=True,
                )
        return result


class DurableBeliefsCodeActStrategy(ReadOnlyTrajectoryCodeActStrategy):
    """Add only the NOOA-style notes API to trajectory analysis."""

    def get_block_overrides(self) -> dict[str, str | DynamicContext | None]:
        blocks = super().get_block_overrides()
        blocks["strategy_prompt"] = f"""## Strategy

Use `execute_python(code)` to inspect the supplied image and trajectory with
ordinary Python and NumPy. Variables persist between analysis cells in this
decision turn. You have up to {ANALYSIS_MAX_ITERATIONS} analysis calls.

Your conversation may be compacted, so preserve curated semantic knowledge in
the durable notes API. Read relevant notes before deciding. Write or revise
notes when evidence changes a useful belief; do not copy raw grids into notes.
The host never writes game interpretations for you.

You also have one reserved action-only commitment inference. Finish by calling
`return_result(action=..., x=..., y=..., explanation=...)`, either directly or
from a cell.
You may analyze repeatedly, but you must return exactly one legal environment
action. Plain text does not finish the turn."""
        blocks["sandbox"] = """## Analysis boundary

Code runs in a fresh, network-disabled, filesystem-confined process with a
20-second cell limit, 30 CPU seconds, and 1024 MB memory. It cannot execute an
environment action or mutate host evidence. The brokered notes API is its only
durable writable surface."""
        blocks["execution_context"] = """## Execution context

The current `image`, immutable-copy `trajectory`, `legal_actions`,
`previous_action`, `turn`, and `levels_completed` are preloaded. `image` is the
settled public 64x64 palette-mode PIL image. Use `image.size`, `image.crop(...)`,
or `np.asarray(image)[y, x]` for exact cell computation, and `show(image)` for
visual inspection. NumPy is available as `np`. Use
`self.list_notes()`, `self.read_notes(name)`, `self.append_notes(name, content)`,
and `self.write_notes(name, content)` for durable beliefs. Use
`return_result(action=..., x=..., y=..., explanation=...)` to finish. Use null
coordinates for every action except CLICK."""
        blocks["self"] = """## Durable notes API

- `self.list_notes() -> list[str]`
- `self.read_notes(name: str) -> str`
- `self.append_notes(name: str, content: str) -> str`
- `self.write_notes(name: str, content: str) -> str`

Names must match `[a-z0-9_-]+.md`. Suggested files are `hypotheses.md` for
mechanics and action meanings, `step_log.md` for concise probes and results,
and `level_log.md` for knowledge that survives level boundaries."""
        return blocks

    def get_block_order(self) -> list[str] | None:
        return [
            "system_prompt",
            "strategy_prompt",
            "sandbox",
            "execution_context",
            "self",
        ]


class AuditLog:
    """Append redacted, structured timing events for the worker and provider boundary."""

    def __init__(self, path: Path | None):
        self.path = path

    def write(self, event: str, **fields: Any) -> None:
        if self.path is None:
            return
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "sequence": CURRENT_SEQUENCE.get(),
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            handle.flush()


class ProviderPoolClient:
    """Sticky ordered failover across independently pinned provider clients."""

    model = MODEL_ROUTE

    def __init__(
        self,
        providers: list[tuple[str, Any]],
        audit: AuditLog,
        *,
        attempt_timeout_seconds: float = PROVIDER_ATTEMPT_TIMEOUT_SECONDS,
    ):
        if not providers:
            raise ValueError("provider pool must not be empty")
        self.providers = providers
        self.audit = audit
        self.attempt_timeout_seconds = attempt_timeout_seconds
        self.preferred_index = 0
        self.unhealthy: set[int] = set()

    @property
    def context_window(self) -> int | None:
        """Expose the model window NOOA uses for honest context accounting."""
        for index, (_, client) in enumerate(self.providers):
            if index not in self.unhealthy:
                window = getattr(client, "context_window", None)
                if window is not None:
                    return int(window)
        return None

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: Any = None,
        output_model: Any = None,
        **kwargs: Any,
    ) -> Any:
        errors: list[tuple[str, Exception]] = []
        order = list(range(self.preferred_index, len(self.providers))) + list(
            range(self.preferred_index)
        )
        for index in order:
            if index in self.unhealthy:
                continue
            name, client = self.providers[index]
            started = time.monotonic()
            self.audit.write(
                "provider_attempt_started",
                provider=name,
                timeout_seconds=self.attempt_timeout_seconds,
            )
            try:
                async with asyncio.timeout(self.attempt_timeout_seconds):
                    response = await client.acall(
                        messages,
                        tools=tools,
                        output_model=output_model,
                        **kwargs,
                    )
            except Exception as error:
                if not _provider_failover_eligible(error):
                    self.audit.write(
                        "provider_attempt_request_rejected",
                        provider=name,
                        elapsed_seconds=round(time.monotonic() - started, 3),
                        **_error_fields(error),
                    )
                    raise
                self.unhealthy.add(index)
                errors.append((name, error))
                self.audit.write(
                    "provider_attempt_failed",
                    provider=name,
                    elapsed_seconds=round(time.monotonic() - started, 3),
                    **_error_fields(error),
                )
                continue
            self.preferred_index = index
            self.audit.write(
                "provider_attempt_completed",
                provider=name,
                elapsed_seconds=round(time.monotonic() - started, 3),
                finish_reason=response.finish_reason,
            )
            return response

        names = ", ".join(name for name, _ in errors) or "none attempted"
        failure = RuntimeError(f"all qualified providers failed: {names}")
        if errors:
            raise failure from errors[-1][1]
        raise failure


def _provider_failover_eligible(error: Exception) -> bool:
    """Return whether another provider can plausibly repair this request."""
    message = str(error).lower()
    if (
        "contextwindowexceeded" in type(error).__name__.lower()
        or "maximum context length" in message
        or "context window" in message
    ):
        return False
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return isinstance(error, (TimeoutError, ConnectionError))


class Decision(BaseModel):
    """The single action to execute now and the solver's explanation."""

    model_config = ConfigDict(extra="forbid")

    action: ActionName
    x: int | None = None
    y: int | None = None
    explanation: str

    @model_validator(mode="after")
    def validate_coordinates(self) -> Decision:
        if self.action == "CLICK":
            if self.x is None or self.y is None:
                raise ValueError("CLICK requires both x and y coordinates")
            if not 0 <= self.x <= 63 or not 0 <= self.y <= 63:
                raise ValueError("CLICK coordinates must be between 0 and 63")
        elif self.x is not None or self.y is not None:
            raise ValueError(f"{self.action} does not accept x or y coordinates")
        return self


class GameSolver(Agent):
    """You are a game solver playing an unfamiliar ARC-AGI-3 interactive grid game.

    Your goal is to beat the game. You do not know the game rules or what the
    controls do inside this game. The action interface uses the same labels as
    the ARC Prize human interface: UP, DOWN, LEFT, RIGHT, SPACEBAR, CLICK x y,
    UNDO, and RESET. Only listed legal actions may be selected. Preserve and
    revise what you learn about their game-specific effects across turns.
    """

    @strategy(PredictStrategy())
    async def next_action(
        self,
        image: TransportImage,
        legal_actions: list[ActionName],
        previous_action: ActionName | None,
        turn: int,
        levels_completed: int,
    ) -> Decision:
        """What is the next best action? Please also give an explanation."""
        ...  # noqa: PIE790 - NOOA uses ellipsis as its generation-method marker


def _forbidden_cell_primitive(
    code: str,
    *,
    allowed_self_methods: frozenset[str] = frozenset(),
    allowed_history_methods: frozenset[str] = frozenset(),
) -> str | None:
    """Reject access outside immutable inputs before a generated cell executes."""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    banned_names = {"__import__", "compile", "eval", "exec", "open"}
    banned_attributes = {
        "__bases__",
        "__builtins__",
        "__class__",
        "__dict__",
        "__func__",
        "__globals__",
        "__mro__",
        "__self__",
        "__subclasses__",
    }
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "history":
            attribute = parents.get(node)
            call = (
                parents.get(attribute) if isinstance(attribute, ast.Attribute) else None
            )
            if not (
                isinstance(attribute, ast.Attribute)
                and attribute.attr in allowed_history_methods
                and isinstance(call, ast.Call)
                and call.func is attribute
            ):
                return "history"
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "history"
            and node.attr not in allowed_history_methods
        ):
            return f"history.{node.attr}"
        if isinstance(node, ast.Name) and node.id == "self":
            attribute = parents.get(node)
            call = (
                parents.get(attribute) if isinstance(attribute, ast.Attribute) else None
            )
            if not (
                isinstance(attribute, ast.Attribute)
                and attribute.attr in allowed_self_methods
                and isinstance(call, ast.Call)
                and call.func is attribute
            ):
                return "self"
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr not in allowed_self_methods
        ):
            return f"self.{node.attr}" if allowed_self_methods else "self"
        if isinstance(node, ast.Name) and node.id in banned_names:
            return node.id
        if isinstance(node, ast.Attribute) and node.attr in banned_attributes:
            return node.attr
    return None


def _auto_echo_broker_result(
    code: str, *, allowed_self_methods: frozenset[str]
) -> tuple[str, bool]:
    """Give bare broker calls notebook-like result visibility.

    CodeAct cells are scripts, so a final expression normally disappears. Models
    repeatedly use broker methods as if they were notebook tools. Echo only a
    top-level final direct call on the allowed `self` surface; assignments and
    explicit prints keep ordinary Python behavior.
    """
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, False
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code, False
    expression = tree.body[-1].value
    if not (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Attribute)
        and isinstance(expression.func.value, ast.Name)
        and expression.func.value.id == "self"
        and expression.func.attr in allowed_self_methods
    ):
        return code, False
    statement = tree.body[-1]
    if statement.end_lineno is None or statement.end_col_offset is None:
        return code, False
    lines = code.splitlines(keepends=True)

    def character_offset(line_number: int, byte_column: int) -> int:
        line = lines[line_number - 1]
        character_column = len(line.encode("utf-8")[:byte_column].decode("utf-8"))
        return sum(len(item) for item in lines[: line_number - 1]) + character_column

    start = character_offset(statement.lineno, statement.col_offset)
    end = character_offset(statement.end_lineno, statement.end_col_offset)
    expression_source = code[start:end]
    return f"{code[:start]}print({expression_source}){code[end:]}", True


class TrajectoryGameSolver(Agent):
    """You are a game solver playing an unfamiliar ARC-AGI-3 interactive grid game.

    Your goal is to beat the game. You do not know the game rules or what the
    controls do inside this game. The action interface uses the same labels as
    the ARC Prize human interface: UP, DOWN, LEFT, RIGHT, SPACEBAR, CLICK x y,
    UNDO, and RESET. Infer game-specific effects from the exact current image
    and complete executed trajectory. Choose exactly one listed legal action.

    The host supplies `trajectory`, a private analysis copy of immutable evidence
    ordered by execution time. Editing that local copy cannot change host evidence.
    Each entry has `turn`, `action`, `before`, `after`, `levels_completed`,
    `state`, and `full_reset`. `before` and `after` are settled public grids as
    rows of integer cell values. You may use Python and NumPy to compare any
    transitions before deciding. For example, `np.argwhere(np.array(t["before"])
    != np.array(t["after"]))` locates changed cells. This analysis cannot alter
    the host evidence or execute a game action. Only the validated final Decision
    is returned to the host, which checks legality and executes one action.
    """

    def __init__(self, *args: Any, audit: AuditLog | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._analysis_audit = audit or AuditLog(None)
        self._analysis_failure_stage: str | None = None

        async def guard_and_record(ctx: Any, nxt: Any) -> Any:
            code = getattr(ctx, "code", "")
            tool_call_id = str(getattr(ctx, "params", {}).get("tool_call_id", ""))
            allowed_self_methods = getattr(self, "_allowed_self_methods", frozenset())
            blocked = (
                _forbidden_cell_primitive(
                    code,
                    allowed_self_methods=allowed_self_methods,
                    allowed_history_methods=getattr(
                        self, "_allowed_history_methods", frozenset()
                    ),
                )
                if isinstance(code, str)
                else None
            )
            executed_code, auto_echoed_result = (
                _auto_echo_broker_result(
                    code, allowed_self_methods=allowed_self_methods
                )
                if isinstance(code, str) and blocked is None
                else (code, False)
            )
            self._analysis_audit.write(
                "codeact_cell_submitted",
                code=code,
                code_sha256=hashlib.sha256(str(code).encode()).hexdigest(),
                executed_code=(executed_code if auto_echoed_result else None),
                auto_echoed_result=auto_echoed_result,
                blocked_primitive=blocked,
                framework_prefill=tool_call_id.startswith("prefill_"),
                tool_call_id=tool_call_id,
            )
            if blocked:
                ctx.code = (
                    f"print('BLOCKED: {blocked} is outside the read-only analysis surface. "
                    "Use image, trajectory, legal_actions, and NumPy only.')"
                )
            elif auto_echoed_result:
                ctx.code = executed_code
            try:
                ctx = await nxt(ctx)
            except Exception as error:
                if _failure_stage(error) == "sandbox":
                    self._analysis_failure_stage = "sandbox"
                raise
            result = getattr(ctx, "result", None)
            result_error = getattr(result, "error", None)
            if (
                isinstance(result_error, Exception)
                and _failure_stage(result_error) == "sandbox"
            ):
                self._analysis_failure_stage = "sandbox"
            self._analysis_audit.write(
                "codeact_cell_completed",
                blocked=blocked is not None,
                stdout=str(getattr(result, "stdout", ""))[:2_000],
                stderr=str(getattr(result, "stderr", ""))[:2_000],
                error=str(getattr(result, "error", ""))[:2_000],
            )
            return ctx

        self.event_manager.intercept("execute_python", guard_and_record)

    async def choose_action(
        self,
        image: PILImage.Image,
        trajectory: list[dict[str, Any]],
        legal_actions: list[ActionName],
        previous_action: ActionName | None,
        turn: int,
        levels_completed: int,
    ) -> Decision:
        """Analyze within budget, then use the reserved commit inference if needed."""
        used_reserved_commit = False
        try:
            decision = await self.analyze_trajectory_and_choose_action(
                image=image,
                trajectory=trajectory,
                legal_actions=legal_actions,
                previous_action=previous_action,
                turn=turn,
                levels_completed=levels_completed,
            )
        except AnalysisBudgetExhausted:
            used_reserved_commit = True
            self._analysis_audit.write(
                "action_commitment_started",
                analysis_calls_used=ANALYSIS_MAX_ITERATIONS,
                max_validation_attempts=COMMIT_MAX_ATTEMPTS,
                legal_actions=legal_actions,
            )
            decision = await self.commit_action(
                legal_actions=legal_actions,
                turn=turn,
                levels_completed=levels_completed,
            )
        if decision.action not in legal_actions and not used_reserved_commit:
            used_reserved_commit = True
            self._analysis_audit.write(
                "action_commitment_started",
                reason="analysis_selected_illegal_action",
                rejected_action=decision.action,
                max_validation_attempts=COMMIT_MAX_ATTEMPTS,
                legal_actions=legal_actions,
            )
            decision = await self.commit_action(
                legal_actions=legal_actions,
                turn=turn,
                levels_completed=levels_completed,
            )
        if decision.action not in legal_actions:
            raise GenerationError(
                f"Model-selected action {decision.action} is not one of the legal actions "
                f"{legal_actions}; no fallback action will be selected."
            )
        if used_reserved_commit:
            self._analysis_audit.write(
                "action_commitment_completed",
                action=decision.action,
                x=decision.x,
                y=decision.y,
            )
        return decision

    @strategy(ReadOnlyTrajectoryCodeActStrategy(config=_CODEACT_CONFIG))
    async def analyze_trajectory_and_choose_action(
        self,
        image: PILImage.Image,
        trajectory: list[dict[str, Any]],
        legal_actions: list[ActionName],
        previous_action: ActionName | None,
        turn: int,
        levels_completed: int,
    ) -> Decision:
        """Analyze recorded experience, then return the next best action and explanation."""
        show(image)
        ...  # noqa: PIE790 - NOOA uses ellipsis as its generation-method marker

    @strategy(CommitActionStrategy(config=_COMMIT_CONFIG))
    async def commit_action(
        self,
        legal_actions: list[ActionName],
        turn: int,
        levels_completed: int,
    ) -> Decision:
        """Commit one legal action now using the retained analysis and reasoning.

        The analysis budget is exhausted. Only `return_result` is available in
        this reserved final inference. Call it with
        `result={"action": <available label>, "x": null, "y": null,
        "explanation": <concise text>}`. CLICK requires integer x/y coordinates.
        """
        ...  # noqa: PIE790 - NOOA uses ellipsis as its generation-method marker


class DurableBeliefsGameSolver(TrajectoryGameSolver):
    """Trajectory solver with model-authored knowledge persisted outside context."""

    knowledge_dir: Annotated[Path, hidden]
    _allowed_self_methods = frozenset(
        {"list_notes", "read_notes", "append_notes", "write_notes"}
    )

    def __init__(
        self,
        *args: Any,
        knowledge_dir: Path,
        audit: AuditLog | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, audit=audit, **kwargs)
        self.knowledge_dir = knowledge_dir
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

    def list_notes(self) -> list[str]:
        """List durable knowledge files created during this game."""
        return [path.name for path in sorted(self.knowledge_dir.glob("*.md"))]

    def read_notes(self, name: str) -> str:
        """Read one durable knowledge file."""
        rejected = self._rejected_note_name(name)
        if rejected:
            return rejected
        path = self.knowledge_dir / name
        result = (
            path.read_text(encoding="utf-8")
            if path.exists()
            else f"(no file {name} yet)"
        )
        self._analysis_audit.write(
            "knowledge_read", name=name, chars=len(result), exists=path.exists()
        )
        return result

    def append_notes(self, name: str, content: str) -> str:
        """Append curated semantic knowledge to a durable file."""
        rejected = self._validate_note_write(name, content, append=True)
        if rejected:
            return rejected
        path = self.knowledge_dir / name
        with path.open("a", encoding="utf-8") as handle:
            handle.write(content.rstrip() + "\n\n")
        return self._record_note_write("append", path)

    def write_notes(self, name: str, content: str) -> str:
        """Replace one durable knowledge file when curating beliefs."""
        rejected = self._validate_note_write(name, content, append=False)
        if rejected:
            return rejected
        path = self.knowledge_dir / name
        path.write_text(content, encoding="utf-8")
        return self._record_note_write("write", path)

    @hidden
    def _rejected_note_name(self, name: str) -> str | None:
        if isinstance(name, str) and NOTE_NAME.fullmatch(name):
            return None
        return f"REJECTED: name {name!r} must match [a-z0-9_-]+.md"

    @hidden
    def _validate_note_write(
        self, name: str, content: str, *, append: bool
    ) -> str | None:
        rejected = self._rejected_note_name(name)
        if rejected:
            return rejected
        if not isinstance(content, str):
            return "REJECTED: note content must be text"
        path = self.knowledge_dir / name
        resulting_chars = len(content.rstrip()) + 2 if append else len(content)
        if append and path.exists():
            resulting_chars += path.stat().st_size
        if resulting_chars > MAX_NOTE_CHARS:
            return f"REJECTED: {name} would exceed {MAX_NOTE_CHARS} characters; curate it first"
        other_chars = sum(
            item.stat().st_size
            for item in self.knowledge_dir.glob("*.md")
            if item != path
        )
        if other_chars + resulting_chars > MAX_KNOWLEDGE_CHARS:
            return (
                f"REJECTED: knowledge would exceed {MAX_KNOWLEDGE_CHARS} characters; "
                "curate existing notes first"
            )
        return None

    @hidden
    def _record_note_write(self, operation: str, path: Path) -> str:
        content = path.read_text(encoding="utf-8")
        self._analysis_audit.write(
            "knowledge_written",
            operation=operation,
            name=path.name,
            chars=len(content),
            sha256=hashlib.sha256(content.encode()).hexdigest(),
        )
        return (
            f"{operation} accepted: {path.name} now contains {len(content)} characters"
        )

    @strategy(DurableBeliefsCodeActStrategy(config=_CODEACT_CONFIG))
    async def analyze_trajectory_and_choose_action(
        self,
        image: PILImage.Image,
        trajectory: list[dict[str, Any]],
        legal_actions: list[ActionName],
        previous_action: ActionName | None,
        turn: int,
        levels_completed: int,
    ) -> Decision:
        """Use evidence and durable beliefs to choose one action."""
        show(image)
        ...  # noqa: PIE790 - NOOA uses ellipsis as its generation-method marker


class CompactBeliefsCodeActStrategy(DurableBeliefsCodeActStrategy):
    """Expose durable notes plus namespaced archived-history retrieval."""

    def get_block_overrides(self) -> dict[str, str | DynamicContext | None]:
        blocks = super().get_block_overrides()
        blocks["strategy_prompt"] = f"""## Strategy

Use `execute_python(code)` to inspect the exact current image and recent
trajectory suffix, durable notes, and archived evidence. Variables persist between analysis cells in
this decision turn. You have up to {ANALYSIS_MAX_ITERATIONS} analysis calls.

After a context-recovery Summary, use the namespaced `history` API when an older
fact could change the next action. Prefer `history.turn(turn)` and the dedicated
bounded views over broad raw retrieval. Preserve revised semantic knowledge in
durable notes; do not copy raw grids into notes.

Finish by calling `return_result(action=..., x=..., y=..., explanation=...)`,
either directly or from a cell. Plain text does not finish the turn."""
        blocks["execution_context"] = """## Execution context

The current `image`, recent immutable-copy `trajectory`, `legal_actions`,
`previous_action`, `turn`, and `levels_completed` are preloaded. `image` is the
settled public 64x64 palette-mode PIL image. Use `image.size`, `image.crop(...)`,
or `np.asarray(image)[y, x]` for exact cell computation, and `show(image)` for
visual inspection. NumPy is available as `np`. The namespaced
read-only `history` object accesses older immutable evidence. Durable notes use
the four documented `self.*_notes` methods. Use
`return_result(action=..., x=..., y=..., explanation=...)` to finish."""
        blocks["history"] = """## Archived history API

- `history.turn(turn: int) -> dict`: decision-shaped single-turn evidence.
- `history.get(event_ids: list[str]) -> list[dict]`: cited exact originals.
- `history.turns(start: int, end: int, streams: list[str] | None = None) -> list[dict]`.
- `history.find(query: str, filters: dict | None = None) -> list[dict]`.
- `history.note(name: str) -> dict`.
- `history.provider_episode(sequence: int, phase: str) -> dict` where phase is
  `context_recovery`, `commitment`, or `failures`.
- `history.around(event_id: str, before: int = 1, after: int = 1) -> dict`.

The projection is credential-redacted. Stable event IDs refer to immutable
source evidence committed in the compaction manifest."""
        return blocks

    def get_static_block_keys(self) -> set[str]:
        return super().get_static_block_keys() | {"history"}

    def get_block_order(self) -> list[str] | None:
        return [
            "system_prompt",
            "strategy_prompt",
            "sandbox",
            "execution_context",
            "history",
            "self",
        ]

    def _extract_module_context(
        self, agent_module: Any, agent: Any | None = None
    ) -> dict[str, Any]:
        del agent_module
        if agent is None or not isinstance(getattr(agent, "history", None), History):
            raise RuntimeError("compact solver has no installed history snapshot")
        return {"np": np, "history": agent.history}


class CompactBeliefsGameSolver(DurableBeliefsGameSolver):
    """Persistent solver with a bounded recent trajectory and searchable archive."""

    history: Annotated[History, hidden]
    _allowed_history_methods = History._METHODS

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.history = History()

    @hidden
    def install_history(self, snapshot: dict[str, Any]) -> None:
        """Install a host-built credential-redacted immutable projection."""
        self.history = History(snapshot)

    @strategy(CompactBeliefsCodeActStrategy(config=_CODEACT_CONFIG))
    async def analyze_trajectory_and_choose_action(
        self,
        image: PILImage.Image,
        trajectory: list[dict[str, Any]],
        legal_actions: list[ActionName],
        previous_action: ActionName | None,
        turn: int,
        levels_completed: int,
    ) -> Decision:
        """Use recent evidence, archived history, and durable beliefs to act."""
        show(image)
        ...  # noqa: PIE790 - NOOA uses ellipsis as its generation-method marker


class ExecutableTheoryCodeActStrategy(CompactBeliefsCodeActStrategy):
    """Expose quick theories and consolidated Moore commitments beside prose."""

    def get_block_overrides(self) -> dict[str, str | DynamicContext | None]:
        blocks = super().get_block_overrides()
        blocks["strategy_prompt"] = f"""## Strategy

Use `execute_python(code)` to inspect the exact current image and recent
trajectory, archived evidence, prose notes, and your executable world-model commitment.
Variables persist between analysis cells in this decision turn. You have up to
{ANALYSIS_MAX_ITERATIONS} analysis calls.

Prose notes remain the place for competing or unresolved hypotheses. An
executable commitment can begin as one small `predict(grid, action)` theory; it
does not need to explain the whole game. You may commit a concrete transition
rule when you judge it useful, let replay test it, and revise, replace, or defer
it. The full Moore-machine form remains available if you decide that history or
hidden state warrants promotion.
Each edit executes outside this sandbox against host evidence and returns only
mechanical support, contradiction, or insufficiency. Evaluation never edits the
commitment.

Finish by calling `return_result(action=..., x=..., y=..., explanation=...)`,
either directly or from a cell. Plain text does not finish the turn."""
        if os.environ.get("NOOA_0159_CAPABILITY_PROBE") == "1":
            blocks["strategy_prompt"] += """

## One-turn capability and architecture probe

Before choosing the action, write `capability_probe.md` with a deeply detailed
self-audit. Enumerate everything you can do in this turn, including exact call
forms, limits, persistence, authority, feedback, and the consequences of using
each capability. Reconstruct the harness architecture end to end: observation
and evidence, persistent actor and conversation, analysis sandbox, durable
notes, archived history, executable-theory broker and isolated evaluator,
decision commitment, host validation, and environment execution.

For every claim, state how you know it: name the prompt block, visible runtime
value, tool result, observed behavior, inference, or prior assumption that
supports it. Separate directly stated facts from inferences. Identify
uncertainties, contradictions, inaccessible components, and capabilities you
suspect exist but cannot access. Explain what using each capability would imply
for the game, evidence, persistence, and security boundaries. Do not invoke a
capability merely to demonstrate that it exists. After saving the audit, inspect
the normal game input and commit one action normally."""
        blocks["sandbox"] = """## Analysis boundary

Code runs in a fresh, network-disabled, filesystem-confined process with a
20-second cell limit, 30 CPU seconds, and 1024 MB memory. It cannot execute an
environment action or mutate host evidence. Durable writes are mediated only by
the notes broker and executable-theory broker. World-model candidates execute in
a separate fresh evaluator with tighter authority and resource limits."""
        blocks["execution_context"] = """## Execution context

The current `image`, recent immutable-copy `trajectory`, `legal_actions`,
`previous_action`, `turn`, and `levels_completed` are preloaded. `image` is the
settled public 64x64 palette-mode PIL image. Use `image.size`, `image.crop(...)`,
or `np.asarray(image)[y, x]` for exact cell computation, and `show(image)` for
visual inspection. NumPy is available as `np`; `history` provides older
immutable evidence. Durable prose
uses the four documented notes methods. Executable commitments use
`self.read_world_model()`, `self.write_world_model(source)`, and
`self.edit_world_model_function(name, source)`. Use `return_result(...)` to
finish with exactly one action."""
        blocks["self"] = blocks["self"] + """

## Executable theory API

- `self.read_world_model() -> dict`
- `self.write_world_model(source: str) -> dict`
- `self.edit_world_model_function(name: str, source: str) -> dict`

The smallest source contract is `predict(grid, action) -> grid | None`.
`grid` is the observed integer NumPy array; `action` is a dict like
`{"action": "UP", "x": None, "y": None}`. Return a same-shaped integer grid,
using `-1` for cells you do not claim, or return `None` to abstain for that
action. This lets you commit one local or action-specific rule quickly.

When a theory needs durable hidden state, replace it with the complete
Moore-machine contract: `State`, `init_state(grid0, level)`,
`transition(state, action)`, `render(state)`, and `outcome(state)`. `render` may
also use `-1`; `outcome` returns `ongoing`, `level_complete`, or `game_over`.
`self.read_world_model()` returns both templates whenever you need them.
Operational validity controls activation; an executable but contradicted edit
still becomes your current commitment. Only a later edit from you replaces it."""
        blocks["self"] += """

Bare direct calls to broker methods automatically display their structured
result. Assigning a result keeps ordinary Python behavior, so print the value
when you want to inspect it. Failures report phase, code, state effect, and
retry surface; a failed candidate never silently replaces the current model."""
        if os.environ.get("NOOA_0159_WM2_GUIDANCE") == "1":
            blocks["theory_practice"] = render_practice_guide()
        if os.environ.get("NOOA_0159_WM2_WORKBENCH") == "1":
            blocks["theory_workbench"] = DynamicContext(
                "self.theory_workbench_context()"
            )
        return blocks

    def get_static_block_keys(self) -> set[str]:
        return super().get_static_block_keys() | {"theory_practice"}

    def get_block_order(self) -> list[str] | None:
        order = [
            "system_prompt",
            "strategy_prompt",
            "sandbox",
            "execution_context",
            "history",
        ]
        if os.environ.get("NOOA_0159_WM2_GUIDANCE") == "1":
            order.append("theory_practice")
        if os.environ.get("NOOA_0159_WM2_WORKBENCH") == "1":
            order.append("theory_workbench")
        order.append("self")
        return order


class ExecutableTheoryGameSolver(CompactBeliefsGameSolver):
    """Compact persistent investigator with prose and executable commitments."""

    theory_lane: Annotated[ExecutableTheoryLane, hidden]
    _theory_workbench_rendered: Annotated[str, hidden]
    _latest_world_model_edit_status: Annotated[str | None, hidden]
    _world_model_was_read: Annotated[bool, hidden]
    _allowed_self_methods = DurableBeliefsGameSolver._allowed_self_methods | frozenset(
        {
            "read_world_model",
            "write_world_model",
            "edit_world_model_function",
        }
    )

    def __init__(self, *args: Any, world_model_dir: Path, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.theory_lane = ExecutableTheoryLane(world_model_dir)
        self._world_model_edit_index = 0
        self._theory_workbench_rendered = (
            "The Theory Workbench View is waiting for authoritative evidence."
        )
        self._latest_world_model_edit_status = None
        self._world_model_was_read = False

    @hidden
    def install_world_model_evidence(self, snapshot_path: Path) -> None:
        state = self.theory_lane.advance(snapshot_path)
        workbench = None
        if (
            os.environ.get("NOOA_0159_WM2_GUIDANCE") == "1"
            and os.environ.get("NOOA_0159_WM2_WORKBENCH") != "1"
        ):
            rendered_guide = render_practice_guide()
            self._analysis_audit.write(
                "theory_practice_available",
                schema=GUIDE_SCHEMA,
                guide_sha256=practice_guide_sha256(),
                rendered_bytes=len(rendered_guide.encode("utf-8")),
            )
        if os.environ.get("NOOA_0159_WM2_WORKBENCH") == "1":
            evidence = E0.load_snapshot(snapshot_path)
            blob_root = snapshot_path.parent.parent / "blobs"

            def read_grid(digest: str) -> tuple[tuple[int, ...], ...]:
                value = json.loads(
                    (blob_root / f"grid-{digest}.json").read_text(encoding="utf-8")
                )
                return tuple(tuple(int(cell) for cell in row) for row in value)

            theory = theory_snapshot_from_lane_view(
                self.theory_lane.read_world_model(),
                latest_edit_status=self._latest_world_model_edit_status,
            )
            workbench = TheoryWorkbenchViewBuilder(read_grid=read_grid).build(
                evidence, theory
            )
            self._theory_workbench_rendered = workbench.render()
            guided_first_use = None
            if os.environ.get("NOOA_0159_WM2_GUIDED_FIRST_USE") == "1":
                guided_first_use = render_guided_first_use(
                    condition_codes=workbench.condition_codes,
                    current_model_version_id=workbench.current_model_version_id,
                    world_model_was_read=self._world_model_was_read,
                )
                if guided_first_use is not None:
                    self._theory_workbench_rendered += "\n\n" + guided_first_use
                    if (
                        len(self._theory_workbench_rendered.encode("utf-8"))
                        > MAX_RENDER_BYTES
                    ):
                        raise ValueError(
                            "guided theory workbench exceeds its frozen byte bound"
                        )
            self._analysis_audit.write(
                "theory_workbench_built",
                schema=workbench.schema,
                view_digest=workbench.view_digest,
                evidence_snapshot_id=workbench.evidence_snapshot_id,
                theory_snapshot_id=workbench.theory_snapshot_id,
                condition_codes=list(workbench.condition_codes),
                displayed_card_codes=[card.code for card in workbench.cards],
                omitted_card_count=workbench.omitted_card_count,
                omitted_card_codes=list(workbench.omitted_card_codes),
                cited_evidence_event_ids=sorted(
                    {
                        event_id
                        for card in workbench.cards
                        for event_id in card.evidence_event_ids
                    }
                ),
                rendered_bytes=len(self._theory_workbench_rendered.encode("utf-8")),
                guided_first_use_schema=(
                    GUIDED_FIRST_USE_SCHEMA
                    if os.environ.get("NOOA_0159_WM2_GUIDED_FIRST_USE") == "1"
                    else None
                ),
                guided_first_use_rendered=guided_first_use is not None,
                guide_schema=(
                    GUIDE_SCHEMA
                    if os.environ.get("NOOA_0159_WM2_GUIDANCE") == "1"
                    else None
                ),
                guide_sha256=(
                    practice_guide_sha256()
                    if os.environ.get("NOOA_0159_WM2_GUIDANCE") == "1"
                    else None
                ),
            )
        self._analysis_audit.write(
            "world_model_evidence_installed",
            evidence_snapshot_id=state.evidence_snapshot_id,
            current_model_version_id=state.current_model_version_id,
            wm2_workbench_built=workbench is not None,
        )

    @hidden
    def theory_workbench_context(self) -> str:
        """Return the view frozen before the current environment decision."""

        return self._theory_workbench_rendered

    def read_world_model(self) -> dict[str, Any]:
        """Read the current commitment plus quick and Moore-machine templates."""
        first_world_model_read = not self._world_model_was_read
        self._world_model_was_read = True
        view = self.theory_lane.read_world_model()
        if os.environ.get("NOOA_0159_WM2_GUIDANCE") == "1":
            view.update(
                {
                    "practice_guide_schema": GUIDE_SCHEMA,
                    "practice_guide_sha256": practice_guide_sha256(),
                    "practice_examples": COMPLETE_PRACTICE_EXAMPLES,
                }
            )
        view["tool_feedback"] = {
            "status": "ok",
            "operation": "read_world_model",
            "message": (
                "No executable commitment is active; choose either supplied template."
                if view["current_model_version_id"] is None
                else "Returned the current Muse-authored executable commitment."
            ),
            "state_effect": "read_only",
            "current_model_version_id": view["current_model_version_id"],
            "evaluation_stale": view["evaluation_stale"],
            "available_contracts": ["partial_predictor", "moore_machine"],
        }
        self._analysis_audit.write(
            "world_model_read",
            current_model_version_id=view["current_model_version_id"],
            evidence_snapshot_id=view["installed_evidence_snapshot_id"],
            first_world_model_read=first_world_model_read,
        )
        return view

    def write_world_model(self, source: str) -> dict[str, Any]:
        """Apply one quick predictor or full Moore commitment through the broker."""
        if not isinstance(source, str):
            return self._world_model_input_error(
                operation="write_world_model",
                code="SOURCE_NOT_TEXT",
                message="source must be a Python source string",
                expected="self.write_world_model(source: str)",
            )
        return self._apply_world_model_edit(
            ModelEdit.whole_file(source), operation="write_world_model"
        )

    def edit_world_model_function(self, name: str, source: str) -> dict[str, Any]:
        """Apply one function replacement through the same WM0 transaction."""
        if not isinstance(name, str) or not isinstance(source, str):
            return self._world_model_input_error(
                operation="edit_world_model_function",
                code="EDIT_ARGUMENT_NOT_TEXT",
                message="name and source must both be strings",
                expected="self.edit_world_model_function(name: str, source: str)",
            )
        return self._apply_world_model_edit(
            ModelEdit.function(name, source), operation="edit_world_model_function"
        )

    def _world_model_input_error(
        self, *, operation: str, code: str, message: str, expected: str
    ) -> dict[str, Any]:
        current = self.theory_lane.read_world_model()["current_model_version_id"]
        return {
            "candidate_status": "rejected_input",
            "active_model_version_id": current,
            "tool_feedback": {
                "status": "error",
                "operation": operation,
                "phase": "input_validation",
                "code": code,
                "message": message,
                "expected": expected,
                "state_effect": "No candidate was created; the current model is unchanged.",
                "retry": {
                    "allowed": True,
                    "current_model_version_id": current,
                    "calls": [
                        "self.read_world_model()",
                        "self.write_world_model(source)",
                        "self.edit_world_model_function(name, source)",
                    ],
                },
            },
        }

    @hidden
    def _apply_world_model_edit(
        self, edit: ModelEdit, *, operation: str
    ) -> dict[str, Any]:
        sequence = CURRENT_SEQUENCE.get()
        if sequence is None:
            raise RuntimeError("world-model edits require an active worker turn")
        self._world_model_edit_index += 1
        provenance = EditProvenance(
            author_event_id=(
                f"worker-turn-{sequence}:world-model-edit-"
                f"{self._world_model_edit_index:04d}"
            )
        )
        evaluation = self.theory_lane.apply(edit, provenance)
        self._latest_world_model_edit_status = evaluation.candidate_status
        self._analysis_audit.write(
            "world_model_edit_evaluated",
            edit_kind=edit.kind,
            function_name=edit.function_name,
            evaluation_id=evaluation.evaluation_id,
            candidate_id=evaluation.candidate_id,
            candidate_status=evaluation.candidate_status,
            active_model_version_id=evaluation.active_model_version_id,
            evidence_snapshot_id=evaluation.evidence_snapshot_id,
            operational_failure_code=evaluation.operational.get("failure_code"),
        )
        # Deliberately return measurements, not a repair instruction.
        result = {
            "evaluation_id": evaluation.evaluation_id,
            "candidate_id": evaluation.candidate_id,
            "candidate_status": evaluation.candidate_status,
            "active_model_version_id": evaluation.active_model_version_id,
            "previous_model_version_id": evaluation.previous_model_version_id,
            "evidence_snapshot_id": evaluation.evidence_snapshot_id,
            "operational": evaluation.operational,
            "initial_render": evaluation.initial_render,
            "dynamics": evaluation.dynamics,
            "outcome": evaluation.outcome,
            "first_divergence": evaluation.first_divergence,
        }
        result["tool_feedback"] = self._world_model_evaluation_feedback(
            operation=operation,
            evaluation=evaluation,
        )
        return result

    @staticmethod
    def _world_model_evaluation_feedback(
        *, operation: str, evaluation: Any
    ) -> dict[str, Any]:
        status = evaluation.candidate_status
        rejected = status == "rejected_operationally"
        contradicted = status == "active_contradicted"
        inconclusive = status == "active_inconclusive"
        operational = evaluation.operational
        code = operational.get("failure_code")
        phase = (
            "source_validation"
            if code in {
                "SOURCE_LIMIT",
                "SYNTAX_ERROR",
                "PROHIBITED_IMPORT",
                "PROHIBITED_OPERATION",
            }
            else "contract_validation"
            if code == "MISSING_CONTRACT"
            or code == "LOAD_FAILED"
            or str(code).startswith("CONTRACT_")
            else "isolated_execution"
            if code
            in {
                "ISOLATION_UNAVAILABLE",
                "MEMORY_LIMIT",
                "OUTPUT_LIMIT",
                "PROCESS_TERMINATED",
                "RESULT_LIMIT",
                "WALL_TIME_LIMIT",
            }
            else "operational_validation"
            if code
            else "empirical_replay"
        )
        if rejected:
            headline = "Candidate rejected; current model preserved."
            severity = "error"
            state_effect = (
                "No model version was activated; the previous current version remains active."
            )
        elif contradicted:
            headline = "Candidate activated, then contradicted by replay evidence."
            severity = "warning"
            state_effect = "The candidate is now current; only a later edit from you replaces it."
        elif inconclusive:
            headline = "Candidate activated; available evidence is insufficient."
            severity = "info"
            state_effect = "The candidate is now current; its evaluation may become stale."
        else:
            headline = "Candidate activated and supported by available replay evidence."
            severity = "success"
            state_effect = "The candidate is now current; support is provisional, not correctness."
        diagnostics = {
            key: operational[key]
            for key in ("failure_code", "message", "line", "column", "contract_kind")
            if operational.get(key) is not None
        }
        return {
            "status": severity,
            "operation": operation,
            "phase": phase,
            "code": code,
            "message": headline,
            "state_effect": state_effect,
            "candidate_status": status,
            "diagnostics": diagnostics,
            "first_divergence": evaluation.first_divergence,
            "retry": {
                "allowed": True,
                "active_model_version_id": evaluation.active_model_version_id,
                "previous_model_version_id": evaluation.previous_model_version_id,
                "calls": [
                    "self.read_world_model()",
                    "self.write_world_model(source)",
                    "self.edit_world_model_function(name, source)",
                ],
                "note": (
                    "Inspect the mechanical diagnostics. A new edit is optional and remains authored by you."
                ),
            },
        }

    @strategy(ExecutableTheoryCodeActStrategy(config=_CODEACT_CONFIG))
    async def analyze_trajectory_and_choose_action(
        self,
        image: PILImage.Image,
        trajectory: list[dict[str, Any]],
        legal_actions: list[ActionName],
        previous_action: ActionName | None,
        turn: int,
        levels_completed: int,
    ) -> Decision:
        """Use evidence, hypotheses, and executable commitments to act."""
        show(image)
        ...  # noqa: PIE790 - NOOA uses ellipsis as its generation-method marker


def _response(content: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
        reasoning=None,
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def _codeact_response(code: str, index: int) -> LLMResponse:
    arguments = json.dumps({"code": code})
    call = ToolCall(id=f"analysis-{index}", name="execute_python", arguments=arguments)
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[call],
        finish_reason="tool_calls",
        assistant_message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": arguments},
                }
            ],
        },
        reasoning=None,
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def _error_fields(error: Exception) -> dict[str, Any]:
    status = getattr(error, "status_code", None)
    return {
        "error_type": type(error).__name__,
        "status_code": status if isinstance(status, int) else None,
        "rate_limited": status == 429 or "429" in str(error),
        "message": str(error).replace("\n", " ")[:300],
    }


def _validated_palette_image(value: Any) -> PILImage.Image:
    """Build the one model-facing PIL observation from host-authored cells."""
    if not isinstance(value, list) or not value:
        raise TypeError("codeact request image_cells must be a non-empty row list")
    width: int | None = None
    for row in value:
        if not isinstance(row, list) or not row:
            raise TypeError("codeact request image_cells rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("codeact request image_cells must be rectangular")
        if any(type(cell) is not int for cell in row):
            raise TypeError("codeact request image_cells cells must be integers")
        if any(cell < 0 or cell >= len(ARC_PALETTE) for cell in row):
            raise ValueError("codeact request image_cells cells must be in [0, 15]")
    image = PILImage.fromarray(np.asarray(value, dtype=np.uint8))
    palette = [channel for color in ARC_PALETTE for channel in color]
    image.putpalette(palette + [0] * (768 - len(palette)))
    return image


def _failure_stage(error: Exception) -> str:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, SandboxError) or type(current).__module__.startswith(
            "nooa.runtime.sandbox"
        ):
            return "sandbox"
        if (
            isinstance(current, TimeoutError)
            or _error_fields(current)["status_code"] is not None
        ):
            return "provider"
        current = current.__cause__ or current.__context__
    if isinstance(error, GenerationError):
        return "model"
    return "worker"


def _find_implicit_context_archival_error(
    error: BaseException,
) -> ImplicitContextArchivalDisabled | None:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ImplicitContextArchivalDisabled):
            return current
        current = current.__cause__ or current.__context__
    return None


def _message_stats(messages: list[dict[str, Any]]) -> dict[str, int]:
    text_chars = 0
    image_blocks = 0
    encoded_image_chars = 0

    def visit(value: Any) -> None:
        nonlocal text_chars, image_blocks, encoded_image_chars
        if isinstance(value, str):
            if value.startswith("data:image/"):
                image_blocks += 1
                encoded_image_chars += len(value)
            else:
                text_chars += len(value)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(messages)
    return {
        "message_count": len(messages),
        "text_chars": text_chars,
        "image_blocks": image_blocks,
        "encoded_image_chars": encoded_image_chars,
    }


def _serializable_usage(usage: Any) -> Any:
    if usage is None or isinstance(usage, dict):
        return usage
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    return str(usage)


def _content_chars(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    model_dump_json = getattr(content, "model_dump_json", None)
    if callable(model_dump_json):
        return len(model_dump_json())
    return len(json.dumps(content, default=str))


def install_provider_logging(
    llm: Any,
    audit: AuditLog,
    *,
    heartbeat_seconds: float = 30.0,
) -> Any:
    """Log one NOOA provider call, its wait time, completion, or exception."""
    original_acall = llm.acall

    async def logged_acall(
        messages: list[dict[str, Any]],
        tools: Any = None,
        output_model: Any = None,
        **kwargs: Any,
    ) -> Any:
        started = time.monotonic()
        audit.write(
            "provider_call_started",
            model=getattr(llm, "model", None),
            structured_output=output_model is not None,
            tool_names=[getattr(tool, "name", None) for tool in (tools or [])],
            tool_choice=kwargs.get("tool_choice"),
            **_message_stats(messages),
        )
        task = asyncio.create_task(
            original_acall(messages, tools=tools, output_model=output_model, **kwargs)
        )
        while True:
            try:
                response = await asyncio.wait_for(
                    asyncio.shield(task), timeout=heartbeat_seconds
                )
                break
            except TimeoutError as error:
                if task.done():
                    if not task.cancelled() and task.exception() is None:
                        response = task.result()
                        break
                    terminal_error = (
                        error if task.cancelled() else task.exception() or error
                    )
                    audit.write(
                        "provider_call_failed",
                        elapsed_seconds=round(time.monotonic() - started, 3),
                        **_error_fields(terminal_error),
                    )
                    raise terminal_error
                audit.write(
                    "provider_call_waiting",
                    elapsed_seconds=round(time.monotonic() - started, 3),
                )
            except Exception as error:
                audit.write(
                    "provider_call_failed",
                    elapsed_seconds=round(time.monotonic() - started, 3),
                    **_error_fields(error),
                )
                raise
        audit.write(
            "provider_call_completed",
            elapsed_seconds=round(time.monotonic() - started, 3),
            finish_reason=response.finish_reason,
            response_chars=_content_chars(response.content),
            usage=_serializable_usage(response.usage),
        )
        return response

    llm.acall = logged_acall
    return llm


def retry_config(audit: AuditLog, *, max_retries: int = 3) -> RetryConfig:
    """Build NOOA retries with a structured callback for every scheduled retry."""

    def on_retry(attempt: int, error: Exception, delay: float) -> None:
        audit.write(
            "provider_retry_scheduled",
            retry_number=attempt,
            delay_seconds=round(delay, 3),
            **_error_fields(error),
        )

    return RetryConfig(
        max_retries=max_retries,
        rate_limit_extra_retries=0 if max_retries == 0 else 5,
        on_retry=on_retry,
    )


def _provider_preferences(
    route: str, quantizations: list[str] | None = None
) -> dict[str, Any]:
    preferences: dict[str, Any] = {
        "order": [route],
        "only": [route],
        "allow_fallbacks": False,
        "require_parameters": False,
    }
    if quantizations:
        preferences["quantizations"] = quantizations
    return preferences


def _live_client(api_key: str, provider: dict[str, Any], audit: AuditLog) -> Any:
    return get_llm_client(
        MODEL_ROUTE,
        api_base=API_BASE,
        api_key=api_key,
        context_window=MODEL_CONTEXT_WINDOW,
        max_tokens=MAX_TOKENS,
        timeout=PROVIDER_ATTEMPT_TIMEOUT_SECONDS - 5.0,
        drop_params=True,
        extra_body={"provider": provider},
        retry_config=retry_config(audit, max_retries=0),
    )


def build_llm(audit: AuditLog | None = None, *, multi_provider: bool = False) -> Any:
    """Build the pinned live client, or a scripted client for hermetic tests."""
    audit = audit or AuditLog(None)
    scripted_cells = os.environ.get("NOOA_0159_SCRIPTED_CODEACT_CELLS")
    if scripted_cells is not None:
        cells = json.loads(scripted_cells)
        if not isinstance(cells, list) or not all(
            isinstance(cell, str) for cell in cells
        ):
            raise ValueError(
                "NOOA_0159_SCRIPTED_CODEACT_CELLS must be a JSON list of strings"
            )
        return install_provider_logging(
            FakeLLMClient(
                [_codeact_response(cell, index) for index, cell in enumerate(cells)]
            ),
            audit,
        )
    scripted = os.environ.get("NOOA_0159_SCRIPTED_RESPONSES")
    if scripted is not None:
        values = json.loads(scripted)
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(
                "NOOA_0159_SCRIPTED_RESPONSES must be a JSON list of strings"
            )
        return install_provider_logging(
            FakeLLMClient([_response(value) for value in values]), audit
        )

    api_key = os.environ.get("ARC_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("ARC_LLM_API_KEY or OPENROUTER_API_KEY is required")
    if multi_provider:
        clients = [
            (
                name,
                _live_client(
                    api_key,
                    _provider_preferences(route, quantizations),
                    audit,
                ),
            )
            for name, route, quantizations in QUALIFIED_PROVIDERS
        ]
        llm = ProviderPoolClient(clients, audit)
    else:
        llm = _live_client(api_key, SINGLE_PROVIDER, audit)
    return install_provider_logging(llm, audit)


def _write(prefix: str, payload: dict[str, Any]) -> None:
    data = (prefix + json.dumps(payload, separators=(",", ":")) + "\n").encode()
    while data:
        written = os.write(_PROTOCOL_FD, data)
        data = data[written:]


def disable_implicit_context_archival(solver: Agent, audit: AuditLog) -> None:
    """Replace NOOA's lossy context-error fallback with an explicit hard failure.

    The compaction-enabled variant has one authority for deleting active history:
    the host's prepare-validate-commit transaction. If a provider rejects an
    in-loop request before that transaction can run, preserve the event store and
    surface the incident rather than mechanically collapsing old events.
    """

    def archival_blocked(
        context_window: int | None, exc: BaseException | None = None
    ) -> None:
        audit.write(
            "implicit_context_archival_blocked",
            context_window=context_window,
            original_error_type=type(exc).__name__ if exc is not None else None,
            original_message=(str(exc).replace("\n", " ")[:300] if exc else None),
        )
        error = ImplicitContextArchivalDisabled(
            "provider rejected an oversized request; implicit NOOA archival is "
            "disabled and active history was preserved"
        )
        if exc is None:
            raise error
        raise error from exc

    solver.runtime._archive_on_context_error = archival_blocked


def _context_snapshot(solver: Agent) -> dict[str, Any] | None:
    """Return provider-anchored context usage at a completed generation boundary."""
    stats = solver.context_stats
    if stats is None:
        return None
    return {
        "prompt_tokens": stats.prompt_tokens,
        "model_context_window": stats.model_context_window,
        "reserved_output_tokens": stats.reserved_output_tokens,
        "effective_window": stats.effective_window,
        "utilization": stats.overall_utilization,
        "context_blocks_chars": stats.context_blocks_chars,
        "events_chars": stats.events_chars,
        "events_count": stats.events_count,
    }


async def serve() -> None:
    """Construct one Agent once, then serve every turn through that same object."""
    audit_path = os.environ.get("NOOA_0159_AUDIT_LOG")
    audit = AuditLog(Path(audit_path) if audit_path else None)
    variant = os.environ.get("NOOA_0159_VARIANT", "predict")
    if variant not in {"predict", "codeact", *KNOWLEDGE_VARIANTS}:
        raise ValueError(f"unknown NOOA_0159_VARIANT {variant!r}")
    if (
        os.environ.get("NOOA_0159_WM2_GUIDED_FIRST_USE") == "1"
        and os.environ.get("NOOA_0159_WM2_WORKBENCH") != "1"
    ):
        raise ValueError("WM2-C guided first use requires the WM2-B workbench")
    scripted_client = any(
        os.environ.get(name) is not None
        for name in (
            "NOOA_0159_SCRIPTED_CODEACT_CELLS",
            "NOOA_0159_SCRIPTED_RESPONSES",
        )
    )
    client_kind = "scripted" if scripted_client else "live"
    missing_sandbox_guards = (
        check_enforceable(_CODEACT_CONFIG.sandbox) if variant != "predict" else []
    )
    multi_provider = (
        variant in COMPACT_VARIANTS
        and MODEL_PROFILE == DEFAULT_MODEL_PROFILE
        and not scripted_client
    )
    llm = build_llm(audit, multi_provider=multi_provider)
    solver: (
        GameSolver
        | TrajectoryGameSolver
        | DurableBeliefsGameSolver
        | CompactBeliefsGameSolver
        | ExecutableTheoryGameSolver
    )
    if variant in KNOWLEDGE_VARIANTS:
        knowledge_path = os.environ.get("NOOA_0159_KNOWLEDGE_DIR")
        if not knowledge_path:
            raise RuntimeError(f"{variant} requires NOOA_0159_KNOWLEDGE_DIR")
        if variant == WORLD_MODEL_VARIANT:
            world_model_path = os.environ.get("NOOA_0159_WORLD_MODEL_DIR")
            if not world_model_path:
                raise RuntimeError(
                    f"{variant} requires NOOA_0159_WORLD_MODEL_DIR"
                )
            solver = ExecutableTheoryGameSolver(
                llm=llm,
                audit=audit,
                knowledge_dir=Path(knowledge_path),
                world_model_dir=Path(world_model_path),
            )
        else:
            solver_type = (
                CompactBeliefsGameSolver
                if variant == "codeact_notes_compact"
                else DurableBeliefsGameSolver
            )
            solver = solver_type(
                llm=llm, audit=audit, knowledge_dir=Path(knowledge_path)
            )
    elif variant == "codeact":
        solver = TrajectoryGameSolver(llm=llm, audit=audit)
    else:
        solver = GameSolver(llm=llm)
    if isinstance(solver, CompactBeliefsGameSolver):
        disable_implicit_context_archival(solver, audit)
    reported_model = "scripted-fake" if scripted_client else MODEL
    provider_route = None
    provider_routes = None
    provider_attempt_timeout_seconds = None
    if not scripted_client:
        if multi_provider:
            provider_route = "qualified-pool-v18"
            provider_routes = [name for name, _, _ in QUALIFIED_PROVIDERS]
        else:
            provider_route = MODEL_CONFIG["provider_route"]
            if variant in COMPACT_VARIANTS:
                provider_routes = MODEL_CONFIG["provider_routes"]
        if variant in COMPACT_VARIANTS:
            provider_attempt_timeout_seconds = PROVIDER_ATTEMPT_TIMEOUT_SECONDS
    audit.write(
        "worker_ready",
        model_profile=MODEL_PROFILE,
        model=reported_model,
        pid=os.getpid(),
        variant=variant,
        client_kind=client_kind,
        provider_route=provider_route,
        provider_routes=provider_routes,
        provider_attempt_timeout_seconds=provider_attempt_timeout_seconds,
        sandbox_enforceable=not missing_sandbox_guards,
        missing_sandbox_guards=missing_sandbox_guards,
        durable_knowledge=variant in KNOWLEDGE_VARIANTS,
        searchable_history=variant in COMPACT_VARIANTS,
        executable_theory=variant == WORLD_MODEL_VARIANT,
        wm2_guidance=os.environ.get("NOOA_0159_WM2_GUIDANCE") == "1",
        wm2_workbench=os.environ.get("NOOA_0159_WM2_WORKBENCH") == "1",
        wm2_guided_first_use=(
            os.environ.get("NOOA_0159_WM2_GUIDED_FIRST_USE") == "1"
        ),
        implicit_context_archival=(False if variant in COMPACT_VARIANTS else None),
    )
    _write(
        READY_PREFIX,
        {
            "model_profile": MODEL_PROFILE,
            "model": reported_model,
            "pid": os.getpid(),
            "variant": variant,
            "client_kind": client_kind,
            "provider_route": provider_route,
            "provider_routes": provider_routes,
            "provider_attempt_timeout_seconds": provider_attempt_timeout_seconds,
            "sandbox_enforceable": not missing_sandbox_guards,
            "missing_sandbox_guards": missing_sandbox_guards,
            "durable_knowledge": variant in KNOWLEDGE_VARIANTS,
            "searchable_history": variant in COMPACT_VARIANTS,
            "executable_theory": variant == WORLD_MODEL_VARIANT,
            "wm2_guidance": os.environ.get("NOOA_0159_WM2_GUIDANCE") == "1",
            "wm2_workbench": os.environ.get("NOOA_0159_WM2_WORKBENCH") == "1",
            "wm2_guided_first_use": (
                os.environ.get("NOOA_0159_WM2_GUIDED_FIRST_USE") == "1"
            ),
            "implicit_context_archival": (
                False if variant in COMPACT_VARIANTS else None
            ),
        },
    )

    turn_event_ranges: dict[int, tuple[str, str]] = {}

    while line := await asyncio.to_thread(sys.stdin.readline):
        request: dict[str, Any] = {}
        token: contextvars.Token[int | None] | None = None
        try:
            request = json.loads(line)
            if request.get("type") == "close":
                return
            if request.get("type") == "compact":
                if not isinstance(solver, CompactBeliefsGameSolver):
                    raise ValueError("compact requests require a compact variant")
                sequence = int(request["sequence"])
                token = CURRENT_SEQUENCE.set(sequence)
                retain_from_turn = int(request["retain_from_turn"])
                eligible = sorted(
                    turn for turn in turn_event_ranges if turn < retain_from_turn
                )
                if not eligible:
                    raise ValueError("compaction range contains no complete old turns")
                summary_text = request.get("summary_text")
                if not isinstance(summary_text, str) or not summary_text.strip():
                    raise ValueError("compaction summary_text must be non-empty")
                snapshot = request.get("history_snapshot")
                if not isinstance(snapshot, dict):
                    raise TypeError("compaction history_snapshot must be an object")
                solver.install_history(snapshot)
                # Include an earlier Summary event on repeated compaction. The
                # immutable host archive, not stacked summaries, is the source
                # of old detail.
                start_tag = solver.event_manager.keys()[0]
                end_tag = turn_event_ranges[eligible[-1]][1]
                before_count = len(solver.event_manager.keys())
                summary_tag = solver.event_manager.collapse(
                    start_tag, end_tag, summary_text
                )
                after_count = len(solver.event_manager.keys())
                for collapsed_turn in eligible:
                    del turn_event_ranges[collapsed_turn]
                audit.write(
                    "context_compaction_applied",
                    resume_turn=sequence,
                    retained_start_turn=retain_from_turn,
                    collapsed_turns=eligible,
                    start_tag=start_tag,
                    end_tag=end_tag,
                    summary_tag=summary_tag,
                    active_events_before=before_count,
                    active_events_after=after_count,
                    archive_event_count=len(snapshot.get("events", [])),
                )
                _write(
                    RESPONSE_PREFIX,
                    {
                        "type": "compaction_applied",
                        "sequence": sequence,
                        "retained_start_turn": retain_from_turn,
                        "collapsed_turns": eligible,
                        "summary_tag": summary_tag,
                        "active_events_before": before_count,
                        "active_events_after": after_count,
                        "archive_event_count": len(snapshot.get("events", [])),
                        "context": _context_snapshot(solver),
                        "worker_pid": os.getpid(),
                    },
                )
                continue
            if request.get("type") != "turn":
                raise ValueError("unknown request type")
            sequence = int(request["sequence"])
            token = CURRENT_SEQUENCE.set(sequence)
            turn_started = time.monotonic()
            analysis_image = (
                _validated_palette_image(request.get("image_cells"))
                if isinstance(solver, TrajectoryGameSolver)
                else None
            )
            audit.write(
                "turn_received",
                image_path=request["image_path"],
                analysis_image_mode=(analysis_image.mode if analysis_image else None),
                analysis_image_size=(
                    list(analysis_image.size) if analysis_image else None
                ),
                legal_actions=request["legal_actions"],
                previous_action=request.get("previous_action"),
                levels_completed=request["levels_completed"],
                trajectory_length=len(request.get("trajectory", [])),
            )
            image = (
                analysis_image
                if analysis_image is not None
                else TransportImage.from_file(Path(request["image_path"]))
            )
            audit.write("nooa_generation_started")
            keys_before = set(solver.event_manager.keys())
            common = {
                "image": image,
                "legal_actions": [str(action) for action in request["legal_actions"]],
                "previous_action": (
                    None
                    if request.get("previous_action") is None
                    else str(request["previous_action"])
                ),
                "turn": sequence,
                "levels_completed": int(request["levels_completed"]),
            }
            if isinstance(solver, TrajectoryGameSolver):
                solver._analysis_failure_stage = None
                if isinstance(solver, ExecutableTheoryGameSolver):
                    evidence_snapshot_path = request.get("evidence_snapshot_path")
                    if not isinstance(evidence_snapshot_path, str):
                        raise TypeError(
                            "world-model request evidence_snapshot_path must be text"
                        )
                    solver.install_world_model_evidence(Path(evidence_snapshot_path))
                if isinstance(solver, CompactBeliefsGameSolver):
                    snapshot = request.get("history_snapshot")
                    if not isinstance(snapshot, dict):
                        raise TypeError(
                            "compact request history_snapshot must be an object"
                        )
                    solver.install_history(snapshot)
                trajectory = request.get("trajectory")
                if not isinstance(trajectory, list):
                    raise TypeError("codeact request trajectory must be a list")
                decision = await solver.choose_action(
                    trajectory=trajectory,
                    **common,
                )
            else:
                decision = await solver.next_action(**common)
            new_tags = [
                tag
                for tag in solver.event_manager.keys()  # noqa: SIM118 - custom API
                if tag not in keys_before and ".." not in tag
            ]
            if not new_tags:
                raise RuntimeError("NOOA turn produced no active event range")
            turn_event_ranges[sequence] = (new_tags[0], new_tags[-1])
            audit.write(
                "turn_completed",
                elapsed_seconds=round(time.monotonic() - turn_started, 3),
                action=decision.action,
                x=decision.x,
                y=decision.y,
                explanation_chars=len(decision.explanation),
            )
            _write(
                RESPONSE_PREFIX,
                {
                    "type": "decision",
                    "sequence": sequence,
                    "action": decision.action,
                    "x": decision.x,
                    "y": decision.y,
                    "explanation": decision.explanation,
                    "context": _context_snapshot(solver),
                    "worker_pid": os.getpid(),
                },
            )
        except Exception as error:  # noqa: BLE001 - protocol must terminalize cleanly
            audit.write("turn_failed", **_error_fields(error))
            archival_error = _find_implicit_context_archival_error(error)
            if archival_error is not None:
                audit.write(
                    "in_loop_compaction_requested",
                    active_events=len(solver.event_manager.keys()),
                )
                _write(
                    RESPONSE_PREFIX,
                    {
                        "type": "needs_compaction",
                        "sequence": request.get("sequence")
                        if isinstance(request, dict)
                        else None,
                        "reason": "context_window_rejected",
                        "message": str(archival_error),
                        "context": _context_snapshot(solver),
                        "active_events": len(solver.event_manager.keys()),
                        "worker_pid": os.getpid(),
                    },
                )
                continue
            failure_stage = getattr(
                solver, "_analysis_failure_stage", None
            ) or _failure_stage(error)
            _write(
                RESPONSE_PREFIX,
                {
                    "type": "error",
                    "sequence": request.get("sequence")
                    if isinstance(request, dict)
                    else None,
                    "error_type": type(error).__name__,
                    "failure_stage": failure_stage,
                    "message": str(error),
                    "worker_pid": os.getpid(),
                },
            )
        finally:
            if token is not None:
                CURRENT_SEQUENCE.reset(token)


if __name__ == "__main__":
    asyncio.run(serve())
