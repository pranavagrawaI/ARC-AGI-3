# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native NOOA control agent for one public action point.

Derived by copy-and-prune from the pinned NVIDIA NOOA ARC example recorded in
../SOURCE_MANIFEST.json. The retained module is the pruned control, not a
verbatim copy of the upstream ARC agent.
"""

from __future__ import annotations

from typing import Annotated, Any

from nooa import hidden

with hidden:
    import ast
    import base64
    from importlib import metadata as importlib_metadata
    import json
    import os

    from nooa import Agent, EventQuery
    from nooa.config import CodeActConfig
    from nooa.config.truncation_config import CaptureConfig, TruncationConfig
    from nooa.runtime.restrictions import DEFAULT_BLOCKED_MODULES, RestrictionsConfig
    from nooa.runtime.method_wrapper import create_agent_method_wrapper
    from nooa.runtime.sandbox.config import SandboxConfig
    from nooa.strategies import CodeActStrategy
    from nooa.strategies.codeact import _ReturnResultSignal
    from nooa.strategy_validation import InvariantError
    from nooa.unifiedllm import Tool
    from pydantic import BaseModel, ConfigDict, Field

    _BLOCKED_RUNTIME_MODULES = frozenset(
        {
            "ctypes",
            "importlib",
            "inspect",
            "litellm",
            "openai",
            "os",
            "site",
            "subprocess",
            "sys",
        }
    )
    _DECLARED_DISTRIBUTION_ROOTS = frozenset({"PIL", "numpy"})
    _UNDECLARED_DISTRIBUTION_ROOTS = (
        frozenset(importlib_metadata.packages_distributions())
        - _DECLARED_DISTRIBUTION_ROOTS
    )
    _CONTROL_RESTRICTIONS = RestrictionsConfig(
        blocked_modules=DEFAULT_BLOCKED_MODULES | _BLOCKED_RUNTIME_MODULES,
        restricted_imports=DEFAULT_BLOCKED_MODULES
        | _BLOCKED_RUNTIME_MODULES
        | _UNDECLARED_DISTRIBUTION_ROOTS
        | {"agent", "nooa"},
    )

    class _PythonCodeRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")

        code: str = Field(description="Python source to execute.")

    class _PersistentSandboxCodeActStrategy(CodeActStrategy):
        """Keep one guarded REPL worker alive across public action points."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._persistent_executor: Any = None

        def _create_sandbox_executor(
            self, runtime: Any, call: Any, builtins: dict[str, Any]
        ) -> Any:
            if self._persistent_executor is None:
                self._persistent_executor = super()._create_sandbox_executor(
                    runtime, call, builtins
                )
            return self._persistent_executor

        async def _close_sandbox(self, session: Any) -> None:
            # CodeAct normally scopes a sandbox worker to one generated method. Here
            # the method boundary is one environment action, so retain the worker and
            # its REPL namespace until the episode-level agent closes explicitly.
            session.sandbox_executor = None

        async def _run_prefill(
            self,
            runtime: Any,
            call: Any,
            builtins: dict[str, Any],
            session: Any,
        ) -> None:
            await super()._run_prefill(runtime, call, builtins, session)
            action_point = call.session_locals.get("action_point")
            if action_point is None:
                raise RuntimeError("public action point missing from the generated-code call")
            source = "action_point = " + repr(action_point.model_dump(mode="python"))
            result = await session.sandbox_executor.run_cell(source, execution_count=0)
            if result.error is not None:
                raise RuntimeError(f"failed to seed public action point: {result.error}")

        def _build_execute_python_tool(self) -> Tool:
            tool = super()._build_execute_python_tool()
            tool.description = (
                "Execute Python 3 in the persistent isolated workspace. "
                "State and files persist for this run; the repository, game "
                "implementation, network, and prior runs are absent."
            )
            tool.parameters_model = _PythonCodeRequest
            return tool

        async def aclose(self) -> None:
            executor, self._persistent_executor = self._persistent_executor, None
            if executor is not None:
                await executor.aclose()


class PublicActionPoint(BaseModel):
    """Everything generated Python may know at one environment action point."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    observation: dict[str, Any]
    legal_actions: tuple[int, ...]
    coordinate_actions: tuple[int, ...] = ()
    frame_width: int | None = None
    frame_height: int | None = None


class ActionDecision(BaseModel):
    """One model-authored environment action."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    action: int
    data: dict[str, int] | None
    reason: str


def validate_decision(action_point: PublicActionPoint, decision: ActionDecision) -> ActionDecision:
    """Validate a typed decision against the current public legal-action set."""

    if decision.action not in action_point.legal_actions:
        raise ValueError(
            f"action {decision.action} is not legal; legal actions are "
            f"{action_point.legal_actions}"
        )
    if decision.action in action_point.coordinate_actions:
        data = decision.data
        if data is None or not all(
            isinstance(data.get(axis), int) and not isinstance(data.get(axis), bool)
            for axis in ("x", "y")
        ):
            raise ValueError(f"action {decision.action} requires integer x and y data")
        width = action_point.frame_width
        height = action_point.frame_height
        if width is None or height is None:
            raise ValueError("coordinate action point requires frame width and height")
        if not (0 <= data["x"] < width and 0 <= data["y"] < height):
            raise ValueError(
                f"coordinates {(data['x'], data['y'])} are outside "
                f"width={width}, height={height}"
            )
    return decision

class NooaControlAgent(Agent):
    """A generic model-directed Python/files agent that returns one public action."""

    _python_state: Annotated[dict[str, Any], hidden]
    _deliberate: Annotated[Any, hidden]
    _control_strategy: Annotated[Any, hidden]
    _workspace: Annotated[str, hidden]
    _max_context_turns: Annotated[int, hidden]
    _visible_turns: Annotated[list[tuple[int, list[dict[str, Any]]]], hidden]
    _evicted_steps: Annotated[list[int], hidden]
    _current_turn_messages: Annotated[list[dict[str, Any]], hidden]
    _max_model_rounds_per_action: Annotated[int, hidden]
    _model_round_this_action: Annotated[int, hidden]
    _python_calls_this_action: Annotated[int, hidden]
    _max_model_calls: Annotated[int | None, hidden]
    _max_reported_tokens: Annotated[int | None, hidden]
    _model_calls_total: Annotated[int, hidden]
    _reported_tokens_total: Annotated[int, hidden]

    def __init__(
        self,
        *args: Any,
        workspace: str | None = None,
        max_context_turns: int = 6,
        max_model_rounds_per_action: int = 10,
        max_decision_repairs: int = 2,
        workspace_timeout_seconds: float = 20.0,
        max_model_calls: int | None = 100,
        max_reported_tokens: int | None = 250_000,
        **kwargs: Any,
    ) -> None:
        if max_context_turns < 1:
            raise ValueError("max_context_turns must be positive")
        if max_model_rounds_per_action < 1:
            raise ValueError("max_model_rounds_per_action must be positive")
        if max_decision_repairs < 0:
            raise ValueError("max_decision_repairs must not be negative")
        if workspace_timeout_seconds <= 0:
            raise ValueError("workspace_timeout_seconds must be positive")
        if max_model_calls is not None and max_model_calls < 1:
            raise ValueError("max_model_calls must be positive or null")
        if max_reported_tokens is not None and max_reported_tokens < 1:
            raise ValueError("max_reported_tokens must be positive or null")
        kwargs.setdefault("event_query", EventQuery.current_call())
        kwargs.setdefault(
            "truncation",
            TruncationConfig(capture=CaptureConfig(max_stdout=4_000)),
        )
        super().__init__(*args, **kwargs)
        self._python_state = {}
        workspace = workspace or os.getcwd()
        self._workspace = workspace
        self._max_context_turns = max_context_turns
        self._visible_turns = []
        self._evicted_steps = []
        self._current_turn_messages = []
        self._max_model_rounds_per_action = max_model_rounds_per_action
        self._model_round_this_action = 0
        self._python_calls_this_action = 0
        self._max_model_calls = max_model_calls
        self._max_reported_tokens = max_reported_tokens
        self._model_calls_total = 0
        self._reported_tokens_total = 0

        def require_current_legal_action(
            _agent: Any,
            decision: ActionDecision,
            call: Any,
        ) -> None:
            action_point = call.bound_parameters().get("action_point")
            if not isinstance(action_point, PublicActionPoint):
                raise RuntimeError("postcondition has no public action point")
            try:
                validate_decision(action_point, decision)
            except ValueError as error:
                raise InvariantError(str(error)) from error

        control_strategy = _PersistentSandboxCodeActStrategy(
            CodeActConfig(
                max_iterations=max_model_rounds_per_action,
                max_retries=max_decision_repairs,
                cell_timeout=workspace_timeout_seconds,
                prefill=None,
                execution_backend="sandbox",
                sandbox=SandboxConfig(
                    workspace=workspace,
                    network=False,
                    max_memory_mb=512,
                    max_cpu_seconds=max(1, int(workspace_timeout_seconds)),
                    recovery="restart_empty",
                    require=True,
                    context_block=False,
                ),
                restrictions=_CONTROL_RESTRICTIONS,
                postconditions=(require_current_legal_action,),
            )
        )
        async def deliberate(
            self: Any, action_point: PublicActionPoint
        ) -> ActionDecision:
            """Inspect the exact public action point, use generic Python and files as
            useful, then return exactly one object with action, data, and a brief reason.
            The action must be one of action_point.legal_actions. Do not assume any event's
            semantic effect from its integer identifier. No action is executed until this
            method returns successfully.
            """

            ...

        generated_method = create_agent_method_wrapper(
            deliberate,
            needs_generation=True,
            needs_tracing=True,
            strategy=control_strategy,
        )
        self._deliberate = generated_method.__get__(self, type(self))
        self._control_strategy = control_strategy
        self.context_manager.set_dynamic_protected("state", "'Python/files session active.'")

        async def reject_runtime_references(context: Any, next_handler: Any) -> Any:
            try:
                tree = ast.parse(context.code)
            except (SyntaxError, TypeError):
                return await next_handler(context)
            forbidden_names = {"self", "__builtins__", "_call"}
            forbidden_calls = {
                "compile",
                "delattr",
                "dir",
                "eval",
                "exec",
                "getattr",
                "globals",
                "help",
                "locals",
                "setattr",
                "vars",
                "__import__",
            }
            forbidden_attributes = {
                "context_manager",
                "event_manager",
                "events",
                "llm",
                "provider",
                "runtime",
            }
            denied = any(
                (isinstance(node, ast.Name) and node.id in forbidden_names)
                or (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in forbidden_calls
                )
                or (
                    isinstance(node, ast.Attribute)
                    and node.attr in forbidden_attributes
                )
                for node in ast.walk(tree)
            )
            if denied:
                context.code = (
                    "raise PermissionError('RUNTIME_ACCESS_DENIED: generated-code "
                    "introspection is outside the public interface')"
                )
            return await next_handler(context)

        self.event_manager.intercept("execute_python", reject_runtime_references)

        async def persist_before_inline_return(context: Any, next_handler: Any) -> Any:
            """Preserve assignments made in the same cell as its final return_result."""

            self._python_calls_this_action += 1
            try:
                tree = ast.parse(context.code)
            except (SyntaxError, TypeError):
                return await next_handler(context)
            final = tree.body[-1] if tree.body else None
            is_inline_return = (
                isinstance(final, ast.Expr)
                and isinstance(final.value, ast.Call)
                and isinstance(final.value.func, ast.Name)
                and final.value.func.id == "return_result"
                and len(final.value.args) == 1
                and not final.value.keywords
            )
            if not is_inline_return:
                return await next_handler(context)
            assert isinstance(final, ast.Expr)
            assert isinstance(final.value, ast.Call)
            result_name = "_action_decision_result"
            tree.body[-1:] = [
                ast.Assign(
                    targets=[ast.Name(id=result_name, ctx=ast.Store())],
                    value=final.value.args[0],
                ),
                ast.Expr(value=ast.Name(id=result_name, ctx=ast.Load())),
            ]
            ast.fix_missing_locations(tree)
            context.code = ast.unparse(tree)
            processed = await next_handler(context)
            execution = processed.result
            if execution.error is None:
                execution.signal = _ReturnResultSignal(result=execution.returned_value)
            return processed

        self.event_manager.intercept("execute_python", persist_before_inline_return)

        async def bound_generated_output(context: Any, next_handler: Any) -> Any:
            processed = await next_handler(context)
            execution = processed.result
            execution.stdout = _bounded_stream(execution.stdout, limit=4_000)
            execution.stderr = _bounded_stream(execution.stderr, limit=2_000)
            return processed

        self.event_manager.intercept("execute_python", bound_generated_output)

        async def expose_only_the_control_interface(context: Any, next_handler: Any) -> Any:
            system = """You choose one public environment action at a time. The exact public
action point is available as `action_point`. Use `execute_python(code)` for generic Python
and relative workspace files. Variables and files persist across action points for this run.
Finish with `return_result({"action": integer, "data": null-or-object, "reason": string})`.
Only currently legal action IDs are accepted. Integer event IDs carry no semantic meaning.
Runtime, provider, controller, event, context, trace, IPC, process, and host objects are not
part of the generated-Python interface. Plain text does not finish an action point."""
            action_point = self._python_state.get("action_point")
            if action_point is None:
                raise RuntimeError("public action point missing before model call")
            frame_path = os.path.join(self._workspace, "current_frame.png")
            with open(frame_path, "rb") as frame:
                image_url = "data:image/png;base64," + base64.b64encode(frame.read()).decode(
                    "ascii"
                )
            self._model_round_this_action += 1
            environment_observation = dict(action_point.observation)
            step = environment_observation.pop("step", None)
            notes_path = os.path.join(self._workspace, "NOTES.md")
            notes = ""
            if os.path.isfile(notes_path):
                with open(notes_path, encoding="utf-8", errors="replace") as notes_file:
                    notes = notes_file.read()
            notes_limit = 4_000
            task = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "step": step,
                                "deliberation": {
                                    "model_round": self._model_round_this_action,
                                    "max_model_rounds": self._max_model_rounds_per_action,
                                    "python_calls_this_action": self._python_calls_this_action,
                                    "remaining_rounds": (
                                        self._max_model_rounds_per_action
                                        - self._model_round_this_action
                                        + 1
                                    ),
                                },
                                "environment_observation": environment_observation,
                                "notes": {
                                    "file": "NOTES.md",
                                    "chars": len(notes),
                                    "limit": notes_limit,
                                    "truncated": len(notes) > notes_limit,
                                    "showing": (
                                        f"last {notes_limit} characters"
                                        if len(notes) > notes_limit
                                        else "all"
                                    ),
                                    "content": notes[-notes_limit:],
                                },
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
            cleaned: list[dict[str, Any]] = []
            for message in context.messages:
                content = message.get("content")
                if message.get("role") == "system":
                    cleaned.append({"role": "system", "content": system})
                elif isinstance(content, str) and "Task(prompt=" in content:
                    cleaned.append(task)
                elif isinstance(content, str) and content.lstrip().startswith("<context>"):
                    continue
                else:
                    cleaned.append(message)
            current = [message for message in cleaned if message.get("role") != "system"]
            assembled = [{"role": "system", "content": system}]
            if self._evicted_steps:
                assembled.append(
                    {
                        "role": "user",
                        "content": (
                            "Transcript omission. Conversation turns for action steps "
                            f"{_compact_steps(self._evicted_steps)} are no longer in this "
                            "context and were not summarized. Every public observation for "
                            "those steps remains readable at observations/step-*.json. "
                            "Files you wrote in the workspace are unchanged."
                        ),
                    }
                )
            for _step, messages in self._visible_turns:
                assembled.extend(messages)
            assembled.extend(current)
            context.messages = assembled
            self._current_turn_messages = json.loads(json.dumps(current))
            if (
                self._max_model_calls is not None
                and self._model_calls_total >= self._max_model_calls
            ):
                raise RuntimeError(
                    "model-call budget exhausted "
                    f"({self._model_calls_total}/{self._max_model_calls})"
                )
            if (
                self._max_reported_tokens is not None
                and self._reported_tokens_total >= self._max_reported_tokens
            ):
                raise RuntimeError(
                    "reported-token budget exhausted "
                    f"({self._reported_tokens_total}/{self._max_reported_tokens})"
                )
            processed = await next_handler(context)
            self._model_calls_total += 1
            response = processed.response
            usage = getattr(response, "usage", None)
            if isinstance(usage, dict):
                reported = usage.get("total_tokens")
                if type(reported) is int and reported >= 0:
                    self._reported_tokens_total += reported
            assistant_message = getattr(response, "assistant_message", None)
            if isinstance(assistant_message, dict):
                assistant_message = dict(assistant_message)
                tool_calls = getattr(response, "tool_calls", None) or ()
                if tool_calls and "tool_calls" not in assistant_message:
                    assistant_message["tool_calls"] = [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                            },
                        }
                        for tool_call in tool_calls
                    ]
                self._current_turn_messages.append(
                    json.loads(json.dumps(assistant_message))
                )
            return processed

        self.event_manager.intercept("llm_call", expose_only_the_control_interface)

    async def choose_action(self, action_point: PublicActionPoint) -> ActionDecision:
        """Run one bounded model/Python action point and validate its decision."""

        # Persist model-created names, but the host's live parameter always wins over
        # the previous action point left in that same REPL namespace.
        self._python_state["action_point"] = action_point
        self._model_round_this_action = 0
        self._python_calls_this_action = 0
        decision = await self._deliberate(
            action_point,
            _session_locals=self._python_state,
        )
        validated = validate_decision(action_point, decision)
        step = action_point.observation.get("step")
        if type(step) is not int:
            raise ValueError("public action point observation requires integer step")
        retained = [_without_frame_image(message) for message in self._current_turn_messages]
        self._visible_turns.append((step, retained))
        while len(self._visible_turns) > self._max_context_turns:
            evicted_step, _messages = self._visible_turns.pop(0)
            self._evicted_steps.append(evicted_step)
        return validated

    async def aclose(self) -> None:
        """Close the episode-scoped generated-Python worker."""

        await self._control_strategy.aclose()


def _compact_steps(steps: list[int]) -> str:
    ordered = sorted(set(steps))
    ranges: list[tuple[int, int]] = []
    for step in ordered:
        if ranges and step == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], step)
        else:
            ranges.append((step, step))
    return ", ".join(
        str(start) if start == end else f"{start}-{end}"
        for start, end in ranges
    )


def _bounded_stream(value: str, *, limit: int) -> str:
    if len(value) <= limit:
        return value
    head = limit // 2
    tail = limit - head
    omitted = len(value) - limit
    return (
        value[:head]
        + f"\n<truncated>{omitted} characters omitted</truncated>\n"
        + value[-tail:]
    )


def _without_frame_image(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if not isinstance(content, list):
        return message
    rebuilt = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            rebuilt.append(
                {
                    "type": "text",
                    "text": (
                        "[the frame image for this earlier step is omitted; every frame "
                        "remains on disk as observations/step-NNNNN.png, and the current "
                        "frame is shown with the newest observation]"
                    ),
                }
            )
        else:
            rebuilt.append(part)
    return {**message, "content": rebuilt}
