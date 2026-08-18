"""A fresh, bounded NOOA builder over Tycho's exact workspace contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from nooa import strategy
from nooa.agent import Agent
from nooa.agentdoc import hidden as hidden_field
from nooa.context_blocks import ResultStatus, ToolCallEvent, ToolResult
from nooa.events import Error, Task
from nooa.strategies.base import GenerationStrategy, RuntimeServices
from nooa.unifiedllm import UnifiedLLM
from tycho.agent.builder import REPORT_FILE, _verify_state
from tycho.agent.colors import COLORMAP
from tycho.prompts.render import render_prompt
from tycho.workspace.agent_tools import ToolExecutor

from tycho_nooa.models import GameTurn
from tycho_nooa.tools import BUILDER_TOOL_PROFILE, WorkspaceToolSurface, tool_by_name

if TYPE_CHECKING:
    from nooa.strategies.current_call import CurrentCall


ARTIFACT_PATHS = (
    "world_model.py",
    "notes/world_model.md",
    REPORT_FILE,
)
FINAL_REPORT_REQUEST = (
    "Step budget reached. Do NOT call any more tools. Right now, in your reply, give "
    "the compact report in the required format (confidence/model/dynamics/outcome/"
    "outcome_verified/plan/recommended_action+subgoal/note) summarizing what you "
    "learned and the state of world_model.py — even if it's 'still exploring, model "
    "not yet coded'. Be concrete about what you DID figure out — especially any open "
    "outcome hypotheses and the action that would disambiguate them — so the actor can "
    "use it."
)
FALLBACK_REPORT = (
    "confidence: none\n"
    "model: none — builder did not finish a model in its step budget\n"
    "dynamics: n/a\n"
    "outcome: unknown\n"
    "outcome_verified: no-terminal-yet\n"
    "plan: start=current; validated=no; length=none; assumptions=unknown\n"
    "recommended_action: none   subgoal: none\n"
    "note: keep exploring; re-invoke with more specific mechanics once you have them"
)


@dataclass(frozen=True)
class BuilderConfig:
    max_tool_steps: int = 40
    max_tokens: int = 16_000
    temperature: float | None = None


@dataclass(frozen=True)
class BuilderRequest:
    turn: GameTurn
    reason: str


@dataclass(frozen=True)
class BuilderResult:
    report: str
    report_fresh: bool
    report_forced: bool
    artifact_hashes: dict[str, dict[str, str | bool | None]]
    tool_calls: list[dict[str, Any]]


def verify_state(executor: ToolExecutor) -> tuple[str, tuple[int, int] | None]:
    """Call Tycho's pinned, auto-attached builder verifier unchanged."""
    return _verify_state(executor)


class BuilderStrategy(GenerationStrategy):
    """Run only Tycho workspace tools, then guarantee a fresh compact report."""

    @property
    def name(self) -> str:
        return "TYCHO_TRIGGER_BUILDER"

    def get_block_overrides(self) -> dict[str, str]:
        return {}

    async def execute(
        self,
        runtime: RuntimeServices,
        call: CurrentCall,
    ) -> tuple[str, bool]:
        builder = self._builder(runtime)
        request = call.bound_parameters().get("request")
        if not isinstance(request, BuilderRequest):
            raise TypeError("BuilderStrategy requires a BuilderRequest named 'request'")

        surface = builder._tool_surface
        config = builder._builder_config
        tools = surface.native_tools()
        tools_by_name = tool_by_name(tools)
        kickoff = render_prompt(
            "builder.user",
            level=request.turn.level,
            hint=request.reason,
            available_actions=request.turn.available_actions,
            report_file=REPORT_FILE,
        )
        verifier_text, _ = verify_state(builder._executor)
        runtime.event_manager.add(Task(prompt=kickoff))
        runtime.event_manager.add(Task(prompt=verifier_text))

        for _ in range(config.max_tool_steps):
            response, event_id = await runtime.generate(
                tools=tools,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                tool_choice="auto",
            )
            if not response.tool_calls:
                break
            runtime.event_manager.remove(event_id)
            for index, tool_call in enumerate(response.tool_calls):
                self._execute_tool_call(
                    runtime,
                    surface,
                    tools_by_name,
                    tool_call,
                    reasoning=(response.reasoning or str(response.content) or None)
                    if index == 0
                    else None,
                )

        report = _read_report(builder._executor.ws.dir)
        if report is not None and report != builder._report_before:
            return report, False

        runtime.event_manager.add(Task(prompt=FINAL_REPORT_REQUEST))
        # Tycho's own forced-report call (builder.py's post-loop fallback) still
        # passes its tools_spec — it never hard-blocks the API from offering
        # tools — but its handling of that final reply only reads reply["text"]
        # and never inspects reply["tool_calls"]. Match both halves: keep tools
        # nominally available, and still only read `content` below, so any
        # tool-call attempt on this turn is silently unexecuted rather than
        # structurally disallowed.
        response, _ = await runtime.generate(
            tools=tools,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            tool_choice="auto",
        )
        text = str(response.content or "").strip() or FALLBACK_REPORT
        builder._executor.ws.write_file(REPORT_FILE, text)
        if _read_report(builder._executor.ws.dir) != text:
            raise RuntimeError(f"failed to persist mandatory fresh {REPORT_FILE}")
        return text, True

    @staticmethod
    def _execute_tool_call(
        runtime: RuntimeServices,
        surface: WorkspaceToolSurface,
        tools_by_name: dict[str, Any],
        tool_call: Any,
        *,
        reasoning: str | None,
    ) -> None:
        try:
            arguments = json.loads(tool_call.arguments)
        except json.JSONDecodeError as error:
            message = f"Invalid builder tool call {tool_call.name!r}: {error}"
            surface.record_rejected_call(
                tool_call.name,
                {"raw_arguments": tool_call.arguments},
                message,
            )
            runtime.event_manager.add(Error(content=message))
            return
        try:
            tool = tools_by_name[tool_call.name]
        except KeyError as error:
            message = f"Invalid builder tool call {tool_call.name!r}: {error}"
            surface.record_rejected_call(tool_call.name, arguments, message)
            runtime.event_manager.add(Error(content=message))
            return

        event_id = runtime.event_manager.add(
            ToolCallEvent(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                arguments=arguments,
                reasoning=reasoning,
            )
        )
        try:
            result = surface.dispatch(tool, arguments)
        except Exception as error:  # noqa: BLE001 - tool failures are builder feedback
            runtime.event_manager.update(
                event_id,
                result=ToolResult(
                    tool_call_id=tool_call.id,
                    content=f"{type(error).__name__}: {error}",
                    result_status=ResultStatus.ERROR,
                ),
            )
            return
        runtime.event_manager.update(
            event_id,
            result=ToolResult(
                tool_call_id=tool_call.id,
                content=str(result),
                result_status=ResultStatus.COMPLETE,
            ),
        )

    @staticmethod
    def _builder(runtime: RuntimeServices) -> FocusedBuilder:
        builder = runtime.agent
        if not isinstance(builder, FocusedBuilder):
            raise TypeError("BuilderStrategy requires a FocusedBuilder")
        return builder


class FocusedBuilder(Agent, context={"self": None, "state": None}):
    """One invocation of Tycho's builder prompt through a fresh NOOA history."""

    _executor: Annotated[ToolExecutor, hidden_field]
    _tool_surface: Annotated[WorkspaceToolSurface, hidden_field]
    _builder_config: Annotated[BuilderConfig, hidden_field]
    _report_before: Annotated[str | None, hidden_field]
    _builder_system_prompt: Annotated[str, hidden_field]

    def __init__(
        self,
        *,
        llm: UnifiedLLM,
        executor: ToolExecutor,
        trace_path: Path | None,
        config: BuilderConfig | None = None,
    ) -> None:
        # Reflection is deliberately disabled: this experiment adds only Tycho's
        # construction loop, not its separate meta-reflection mechanism.
        system_prompt = render_prompt(
            "builder.system",
            colormap=COLORMAP,
            report_file=REPORT_FILE,
            meta_reflect=False,
        )
        super().__init__(llm=llm)
        self._builder_system_prompt = system_prompt
        self._executor = executor
        self._tool_surface = WorkspaceToolSurface(
            executor,
            profile=BUILDER_TOOL_PROFILE,
            trace_path=trace_path,
            planner_trace_path=(
                trace_path.with_name("builder-planner-trace.jsonl")
                if trace_path is not None
                else None
            ),
        )
        self._builder_config = config or BuilderConfig()
        self._report_before = None

    def _resolve_system_prompt(self) -> str:
        """Return Tycho's rendered builder.system prompt byte-for-byte."""
        return self._builder_system_prompt

    async def build(self, request: BuilderRequest) -> BuilderResult:
        """Construct against the shared workspace and return fresh artifact evidence."""
        self._tool_surface.begin_turn(request.turn)
        before = _workspace_hashes(self._executor.ws.dir)
        self._report_before = _read_report(self._executor.ws.dir)
        report, forced = await self._construct(request)
        after = _workspace_hashes(self._executor.ws.dir)
        artifact_paths = set(ARTIFACT_PATHS)
        artifact_paths.update(
            path
            for path in before.keys() | after.keys()
            if before.get(path) != after.get(path)
        )
        hashes = {
            path: {
                "before": before.get(path),
                "after": after.get(path),
                "changed": before.get(path) != after.get(path),
            }
            for path in sorted(artifact_paths)
        }
        return BuilderResult(
            report=report,
            # Freshness is a claim about the report's bytes, not about which code
            # path produced them: a forced final-report request can still emit the
            # same content that was already on disk (DIAGNOSIS.md turns 11 and 19),
            # and that must not read as a genuine update.
            report_fresh=hashes[REPORT_FILE]["changed"] is True,
            report_forced=forced,
            artifact_hashes=hashes,
            tool_calls=self._tool_surface.turn_tool_calls(),
        )

    @strategy(BuilderStrategy())
    async def _construct(self, request: BuilderRequest) -> tuple[str, bool]: ...


def _workspace_hashes(workspace_dir: Path) -> dict[str, str]:
    """Hash all stable files so unexpected builder side effects remain visible."""
    return {
        path.relative_to(workspace_dir).as_posix(): digest
        for path in workspace_dir.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and (digest := _file_hash(path)) is not None
    }


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_report(workspace_dir: Path) -> str | None:
    path = workspace_dir / REPORT_FILE
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")
