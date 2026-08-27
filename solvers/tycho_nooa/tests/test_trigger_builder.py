from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

import pytest
from nooa.context_blocks import ToolCallEvent
from nooa.events import Task
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from tycho.agent.builder import REPORT_FILE
from tycho.agent.colors import COLORMAP
from tycho.prompts.render import render_prompt
from tycho.workspace.agent_tools import ToolExecutor
from tycho.workspace.workspace import GameWorkspace

from tycho_nooa import ActionDecision, GameTurn, TychoNooaAgent
from tycho_nooa.builder import (
    FALLBACK_REPORT,
    RETAINED_CANDIDATE_BRIEF,
    BuilderConfig,
    BuilderRequest,
    BuilderResult,
    FocusedBuilder,
    _builder_budget_notice,
)
from tycho_nooa.deadlines import DecisionDeadlineConfig
from tycho_nooa.epistemic import (
    CandidateGoalStatus,
    RequestBuilderArguments,
)
from tycho_nooa.optionality import OPTIONALITY_PROMPT_BRIEF
from tycho_nooa.runtime_notice import PROMPT_BRIEF
from tycho_nooa.theory_evaluation import (
    EvaluationRunStatus,
    EvidenceFrontier,
    ObligationKind,
    ObligationResult,
    ObligationStatus,
    PublicationDecision,
    PublicationDisposition,
    TheoryEvaluation,
)
from tycho_nooa.theory_transaction import TheoryTransaction
from tycho_nooa.trigger_builder import (
    BuilderExposure,
    TriggerDecision,
    _contemporaneous_score,
    _publication_feedback,
    _recommended_action,
)
from tycho_nooa.wm_probe import AST_UNCHANGED_NOTICE


def tool_response(tool: str, arguments: dict, call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=call_id, name=tool, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def malformed_tool_response(tool: str, arguments: str, call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=call_id, name=tool, arguments=arguments)],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def tool_batch_response(calls: list[tuple[str, dict, str]]) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(id=call_id, name=tool, arguments=json.dumps(arguments))
            for tool, arguments, call_id in calls
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def prose_response(content: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def truncated_reasoning_response(content: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[],
        finish_reason="length",
        assistant_message={"role": "assistant", "content": None},
        reasoning=content,
        usage={"prompt_tokens": 1, "completion_tokens": 4096, "total_tokens": 4097},
    )


class RecordingFakeLLMClient(FakeLLMClient):
    """A FakeLLMClient that also records the tools offered on every call."""

    def __init__(self, scripted_responses: list[LLMResponse]) -> None:
        super().__init__(scripted_responses=scripted_responses)
        self.tools_per_call: list[list[str]] = []

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        self.tools_per_call.append([tool.name for tool in tools] if tools else [])
        return await super().acall(
            messages, tools=tools, output_model=output_model, **kwargs
        )


class MessageRecordingFakeLLMClient(FakeLLMClient):
    def __init__(self, scripted_responses: list[LLMResponse]) -> None:
        super().__init__(scripted_responses=scripted_responses)
        self.messages_per_call: list[list[dict]] = []

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        self.messages_per_call.append(messages)
        return await super().acall(
            messages, tools=tools, output_model=output_model, **kwargs
        )


class EditAfterTruncationLLM(FakeLLMClient):
    """Replay the live failure, then edit only if construction continues."""

    def __init__(self) -> None:
        super().__init__(scripted_responses=[])
        self.call_count = 0

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        self.call_count += 1
        if self.call_count == 1:
            return truncated_reasoning_response("I am ready to edit, but")
        latest = str(messages[-1].get("content", ""))
        if "BUILDER BUDGET" in latest:
            return tool_batch_response(
                [
                    (
                        "write_file",
                        {"path": "world_model.py", "content": "candidate"},
                        "candidate-edit",
                    ),
                    (
                        "write_file",
                        {"path": "notes/world_model.md", "content": "hypothesis"},
                        "notes-edit",
                    ),
                    (
                        "write_file",
                        {
                            "path": "notes/world_model_report.md",
                            "content": "recommended_action: none",
                        },
                        "report-edit",
                    ),
                ]
            )
        return prose_response("recommended_action: none")


class EditWhenInvestigationClosesLLM(FakeLLMClient):
    """Keep investigating while run_python remains available."""

    def __init__(self) -> None:
        super().__init__(scripted_responses=[])
        self.call_count = 0
        self.tool_choices: list[str] = []

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        self.call_count += 1
        self.tool_choices.append(kwargs["tool_choice"])
        names = {tool.name for tool in tools or []}
        if self.call_count == 1 or "run_python" in names:
            return tool_response("run_python", {"code": "print('probe')"}, "probe")
        return tool_batch_response(
            [
                (
                    "write_file",
                    {"path": "world_model.py", "content": "candidate"},
                    "candidate-edit",
                ),
                (
                    "write_file",
                    {"path": "notes/world_model.md", "content": "hypothesis"},
                    "notes-edit",
                ),
                (
                    "write_file",
                    {
                        "path": "notes/world_model_report.md",
                        "content": "recommended_action: none",
                    },
                    "report-edit",
                ),
            ]
        )


class StaleInvestigationToolLLM(FakeLLMClient):
    """Emit a remembered investigation tool after its schema disappears."""

    def __init__(self) -> None:
        super().__init__(scripted_responses=[])
        self.call_count = 0

    async def acall(self, messages, tools=None, output_model=None, **kwargs):
        self.call_count += 1
        if self.call_count <= 2:
            return tool_response(
                "run_python",
                {"code": "print('probe')"},
                "initial-probe" if self.call_count == 1 else "stale-probe",
            )
        if self.call_count == 3:
            return tool_batch_response(
                [
                    (
                        "write_file",
                        {"path": "world_model.py", "content": "candidate"},
                        "candidate-edit",
                    ),
                    (
                        "write_file",
                        {"path": "notes/world_model.md", "content": "hypothesis"},
                        "notes-edit",
                    ),
                    (
                        "write_file",
                        {
                            "path": "notes/world_model_report.md",
                            "content": "recommended_action: none",
                        },
                        "report-edit",
                    ),
                ]
            )
        return prose_response("recommended_action: none")


def turn(number: int, previous_action: str = "") -> GameTurn:
    return GameTurn(
        grid=[[0, number], [2, 3]],
        available_actions=["UP", "DOWN"],
        level=0,
        turn_in_level=number,
        previous_action=previous_action,
    )


@pytest.mark.asyncio
async def test_builder_does_not_treat_token_truncation_as_completed_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reasoning response cut off before its tool call is not a prose exit."""
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] first divergence remains", None),
    )
    workspace = GameWorkspace(
        "game", root=str(tmp_path), available_actions=["UP", "DOWN"]
    )
    (workspace.dir / "world_model.py").write_text("parent")
    llm = EditAfterTruncationLLM()
    builder = FocusedBuilder(
        llm=llm,
        executor=ToolExecutor(workspace, wm_feedback_enabled=False),
        trace_path=tmp_path / "builder-tools.jsonl",
        config=BuilderConfig(max_tool_steps=2, show_budget_telemetry=True),
    )

    await builder.build(BuilderRequest(turn=turn(0), reason="repair divergence"))

    assert (workspace.dir / "world_model.py").read_text() == "candidate"


@pytest.mark.asyncio
async def test_builder_commit_mode_mechanically_closes_investigation_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Commit mode changes capability, rather than merely adding more prose."""
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] first divergence remains", None),
    )
    workspace = GameWorkspace(
        "game", root=str(tmp_path), available_actions=["UP", "DOWN"]
    )
    (workspace.dir / "world_model.py").write_text("parent")
    llm = EditWhenInvestigationClosesLLM()
    builder = FocusedBuilder(
        llm=llm,
        executor=ToolExecutor(workspace, wm_feedback_enabled=False),
        trace_path=tmp_path / "builder-tools.jsonl",
        config=BuilderConfig(max_tool_steps=4, show_budget_telemetry=True),
    )

    await builder.build(BuilderRequest(turn=turn(0), reason="repair divergence"))

    assert (workspace.dir / "world_model.py").read_text() == "candidate"
    assert llm.tool_choices[:2] == ["auto", "required"]


@pytest.mark.asyncio
async def test_builder_rejects_remembered_tool_that_commit_mode_did_not_offer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dispatch authority is the current call's tools, not the original map."""
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] first divergence remains", None),
    )
    workspace = GameWorkspace(
        "game", root=str(tmp_path), available_actions=["UP", "DOWN"]
    )
    (workspace.dir / "world_model.py").write_text("parent")
    builder = FocusedBuilder(
        llm=StaleInvestigationToolLLM(),
        executor=ToolExecutor(workspace, wm_feedback_enabled=False),
        trace_path=tmp_path / "builder-tools.jsonl",
        config=BuilderConfig(max_tool_steps=4, show_budget_telemetry=True),
    )

    result = await builder.build(
        BuilderRequest(turn=turn(0), reason="repair divergence")
    )

    stale = next(
        row for row in result.tool_calls if row["tool_call_id"] == "stale-probe"
    )
    assert stale["status"] == "error"
    assert "Invalid builder tool call 'run_python'" in stale["result"]
    assert (workspace.dir / "world_model.py").read_text() == "candidate"


@pytest.mark.asyncio
async def test_transactional_builder_second_edit_sees_restored_incumbent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[typed evaluator owns publication]", None),
    )
    workspace = GameWorkspace(
        "game", root=str(tmp_path), available_actions=["UP", "DOWN"]
    )
    (workspace.dir / "world_model.py").write_text("parent")
    frontier = EvidenceFrontier.freeze(workspace.dir)

    class Evaluator:
        evaluator_sha256 = "fixture"

        def evaluate(self, model: bytes, frozen: EvidenceFrontier) -> TheoryEvaluation:
            statuses = {
                b"parent": (ObligationStatus.PASSED, ObligationStatus.FAILED),
                b"regression": (ObligationStatus.FAILED, ObligationStatus.FAILED),
                b"next": (ObligationStatus.PASSED, ObligationStatus.PASSED),
            }[model]
            return TheoryEvaluation(
                hashlib.sha256(model).hexdigest(),
                frozen.sha256,
                self.evaluator_sha256,
                EvaluationRunStatus.COMPLETE,
                tuple(
                    ObligationResult(
                        f"T{index}",
                        ObligationKind.TRANSITION,
                        status,
                        f"transition {index}",
                    )
                    for index, status in enumerate(statuses, start=1)
                ),
            )

    transaction = TheoryTransaction.begin(
        workspace=workspace.dir,
        evaluator=Evaluator(),
        frontier=frontier,
        cause="preserved-boundary integration",
        store_root=tmp_path / "store",
    )
    llm = FakeLLMClient(
        scripted_responses=[
            tool_response(
                "write_file",
                {"path": "world_model.py", "content": "regression"},
                "broad-regression",
            ),
            tool_response(
                "edit_file",
                {"path": "world_model.py", "old": "parent", "new": "next"},
                "edit-restored-parent",
            ),
            tool_response(
                "write_file",
                {
                    "path": "notes/world_model.md",
                    "content": "broad rewrite regressed; narrow edit survived",
                },
                "notes",
            ),
            tool_response(
                "write_file",
                {
                    "path": "notes/world_model_report.md",
                    "content": "recommended_action: none\nnote: transaction exercised",
                },
                "report",
            ),
            prose_response("complete"),
        ]
    )
    builder = FocusedBuilder(
        llm=llm,
        executor=ToolExecutor(workspace, wm_feedback_enabled=False),
        trace_path=tmp_path / "builder-tools.jsonl",
        config=BuilderConfig(max_tool_steps=5),
        theory_transaction=transaction,
    )

    result = await builder.build(
        BuilderRequest(turn=turn(0), reason="repair a divergence")
    )

    assert "rejected_and_restored" in result.tool_calls[0]["result"]
    assert result.tool_calls[1]["result"].startswith("edited world_model.py")
    assert (workspace.dir / "world_model.py").read_bytes() == b"next"


@pytest.mark.asyncio
async def test_orchestrator_actor_pulls_a_fresh_builder_before_committing_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The builder report must be available before the actor's one commitment."""
    monkeypatch.setenv("TYCHO_MODE", "orchestrator")
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] no recorded transitions yet", None),
    )
    llm = MessageRecordingFakeLLMClient(
        scripted_responses=[
            tool_batch_response(
                [
                    (
                        "invoke_builder",
                        {"reason": "infer the objective from level_0/turn_000.txt"},
                        "actor-build",
                    ),
                    ("take_action", {"action": "UP"}, "stale-action"),
                ]
            ),
            tool_response(
                "write_file",
                {
                    "path": "notes/world_model.md",
                    "content": "Objective may be contact with blue.",
                },
                "builder-notes",
            ),
            tool_response(
                "write_file",
                {
                    "path": "notes/world_model_report.md",
                    "content": "recommended_action: DOWN\nnote: test contact",
                },
                "builder-report",
            ),
            prose_response("builder complete"),
            tool_response("take_action", {"action": "DOWN"}, "fresh-action"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )

    decision = await agent.act(turn(0))

    assert decision == ActionDecision(action="DOWN")
    assert agent._trigger_builder is not None
    assert agent._trigger_builder.builder is not None
    trace = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    request = next(record for record in trace if record["event"] == "actor_request")
    assert request["reason"] == "infer the objective from level_0/turn_000.txt"
    assert any(
        "recommended_action: DOWN" in str(message)
        for message in llm.messages_per_call[-1]
    )


@pytest.mark.asyncio
async def test_typed_goal_request_preserves_hypothesis_and_builder_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYCHO_MODE", "orchestrator")
    monkeypatch.setenv("TYCHO_NOOA_TREATMENT", "typed_candidate_goals")
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] no recorded transitions yet", None),
    )
    llm = MessageRecordingFakeLLMClient(
        scripted_responses=[
            tool_response(
                "propose_goal",
                {
                    "predicate": "contact between colours 4 and 11 completes the level",
                    "evidence_refs": ["level_0/turn_000.txt"],
                    "falsifier": "contact occurs while the level remains incomplete",
                },
                "goal-proposal",
            ),
            tool_response(
                "take_action",
                {"action": "UP"},
                "blocked-before-review",
            ),
            tool_batch_response(
                [
                    (
                        "request_builder",
                        {
                            "focus": "objective",
                            "evidence_refs": ["level_0/turn_000.txt"],
                            "candidate_goal_id": "goal-0001",
                        },
                        "typed-build",
                    ),
                    (
                        "update_goal",
                        {
                            "goal_id": "goal-0001",
                            "status": "selected",
                            "evidence_refs": ["level_0/turn_000.txt"],
                            "note": "premature batched resolution",
                        },
                        "blocked-batched-resolution",
                    ),
                ]
            ),
            tool_response(
                "write_file",
                {
                    "path": "world_model.py",
                    "content": "# premature objective promotion\n",
                },
                "blocked-objective-promotion",
            ),
            tool_response(
                "write_file",
                {
                    "path": "notes/world_model.md",
                    "content": "Candidate goal is plausible but not yet verified.",
                },
                "builder-notes",
            ),
            tool_response(
                "write_file",
                {
                    "path": "notes/world_model_report.md",
                    "content": (
                        "candidate_goal_assessment: supported\n"
                        "candidate_goal_reason: contact is the only congruent target\n"
                        "candidate_goal_route: DOWN x2\n"
                        "recommended_action: DOWN\n"
                    ),
                },
                "builder-report",
            ),
            prose_response("builder complete"),
            tool_response(
                "take_action",
                {"action": "UP"},
                "blocked-before-resolution",
            ),
            tool_response(
                "update_goal",
                {
                    "goal_id": "goal-0001",
                    "status": "selected",
                    "evidence_refs": ["level_0/turn_000.txt"],
                    "note": "builder evidence supports a direct contact test",
                },
                "goal-selection",
            ),
            tool_response("take_action", {"action": "DOWN"}, "fresh-action"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )
    world_model_before = (agent.workspace_dir / "world_model.py").read_bytes()

    decision = await agent.act(turn(0))

    assert decision == ActionDecision(action="DOWN")
    assert (agent.workspace_dir / "world_model.py").read_bytes() == world_model_before
    assert agent._goal_ledger is not None
    goal = agent._goal_ledger.get("goal-0001")
    assert goal.status.value == "selected"
    assert goal.resolved_review_id == "review-0001"
    request = next(
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "actor_request"
    )
    payload = json.loads(request["reason"])
    assert payload["focus"] == "objective"
    assert payload["candidate_goal"]["id"] == "goal-0001"
    assert payload["evidence_refs"] == ["level_0/turn_000.txt"]
    assert request["focus"] == "objective"
    tool_records = [
        json.loads(line)
        for line in (tmp_path / "tool-trace.jsonl").read_text().splitlines()
    ]
    blocked = [
        record
        for record in tool_records
        if record["name"] == "take_action" and record["status"] == "error"
    ]
    assert len(blocked) == 2
    assert "request_builder" in blocked[0]["result"]
    assert "update_goal" in blocked[1]["result"]
    batched_resolution = next(
        record
        for record in tool_records
        if record["tool_call_id"] == "blocked-batched-resolution"
    )
    assert batched_resolution["status"] == "error"
    assert "cannot share a tool batch" in batched_resolution["result"]
    review = json.loads(
        (
            agent.workspace_dir
            / "epistemic"
            / "objective-builder-reviews.jsonl"
        ).read_text()
    )
    assert review["assessment"] == "supported"
    assert review["recommended_route"] == "DOWN x2"
    builder_records = [
        json.loads(line)
        for line in (tmp_path / "builder-tool-trace.jsonl").read_text().splitlines()
    ]
    rejected_promotion = next(
        record
        for record in builder_records
        if record["tool_call_id"] == "blocked-objective-promotion"
    )
    assert rejected_promotion["status"] == "error"
    assert "cannot edit world_model.py" in rejected_promotion["result"]


def test_typed_goal_ledger_is_inlined_on_the_next_actor_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYCHO_MODE", "orchestrator")
    monkeypatch.setenv("TYCHO_NOOA_TREATMENT", "typed_candidate_goals")
    agent = TychoNooaAgent(
        llm=FakeLLMClient(scripted_responses=[]),
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )
    evidence = agent.workspace_dir / "level_0" / "turn_000.txt"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("host-authored frame")
    assert agent._goal_ledger is not None
    agent._goal_ledger.register_authoritative_evidence("level_0/turn_000.txt")
    surface = agent._tool_surface.goal_tools
    assert surface is not None
    surface.begin_turn(turn(0))
    proposed = json.loads(
        surface.propose_goal(
            predicate="contact between colours 4 and 11 completes the level",
            evidence_refs=["level_0/turn_000.txt"],
            falsifier="contact occurs while the level remains incomplete",
        )
    )
    request = RequestBuilderArguments(
        focus="objective",
        evidence_refs=["level_0/turn_000.txt"],
        candidate_goal_id=proposed["id"],
    )
    surface.builder_request(request)
    surface.record_builder_review(
        request,
        report=(
            "candidate_goal_assessment: supported\n"
            "candidate_goal_reason: contact is the only congruent target\n"
            "candidate_goal_route: DOWN x2\n"
            "recommended_action: DOWN\n"
        ),
    )
    surface.update_goal(
        proposed["id"],
        CandidateGoalStatus.SELECTED,
        ["level_0/turn_000.txt"],
        "builder evidence supports a direct contact test",
    )

    next_turn = turn(1, previous_action="DOWN")
    agent._tool_surface.begin_turn(next_turn)
    observation = agent._tool_surface.observation(next_turn)
    assert "typed candidate goals" in observation.prompt
    assert "typed candidate goals" in observation.elided
    assert proposed["id"] in observation.prompt
    assert "status=selected" in observation.prompt
    assert "recommended_action=DOWN" in observation.prompt
    assert "route=DOWN x2" in observation.prompt


@pytest.mark.asyncio
async def test_typed_objective_timeout_returns_a_resolvable_unavailable_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingObjectiveBuilderLLM(FakeLLMClient):
        def __init__(self) -> None:
            super().__init__(
                scripted_responses=[
                    tool_response(
                        "propose_goal",
                        {
                            "predicate": "contact completes the level",
                            "evidence_refs": ["level_0/turn_000.txt"],
                            "falsifier": "contact is nonterminal",
                        },
                        "goal-proposal",
                    ),
                    tool_response(
                        "request_builder",
                        {
                            "focus": "objective",
                            "evidence_refs": ["level_0/turn_000.txt"],
                            "candidate_goal_id": "goal-0001",
                        },
                        "typed-build",
                    ),
                    tool_response(
                        "update_goal",
                        {
                            "goal_id": "goal-0001",
                            "status": "proposed",
                            "evidence_refs": ["level_0/turn_000.txt"],
                            "note": "builder unavailable; retain without selecting",
                        },
                        "acknowledge-unavailable",
                    ),
                    tool_response("take_action", {"action": "UP"}, "fresh-action"),
                ]
            )
            self.calls = 0

        async def acall(self, messages, tools=None, output_model=None, **kwargs):
            self.calls += 1
            if self.calls == 3:
                await asyncio.Event().wait()
            return await super().acall(
                messages, tools=tools, output_model=output_model, **kwargs
            )

    monkeypatch.setenv("TYCHO_MODE", "orchestrator")
    monkeypatch.setenv("TYCHO_NOOA_TREATMENT", "typed_candidate_goals")
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] objective timeout fixture", None),
    )
    agent = TychoNooaAgent(
        llm=BlockingObjectiveBuilderLLM(),
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
        deadline_config=DecisionDeadlineConfig(
            builder_cap_seconds=0.01,
            actor_reserve_seconds=0.02,
            objective_builder_cap_seconds=0.01,
            objective_actor_reserve_seconds=0.02,
            host_return_reserve_seconds=0.005,
        ),
    )

    decision = await agent.act(turn(0), decision_timeout_seconds=0.1)

    assert decision == ActionDecision(action="UP")
    assert agent._goal_ledger is not None
    goal = agent._goal_ledger.get("goal-0001")
    assert goal.status.value == "proposed"
    assert goal.resolved_review_id == "review-0001"
    review = json.loads(
        (
            agent.workspace_dir
            / "epistemic"
            / "objective-builder-reviews.jsonl"
        ).read_text()
    )
    assert review["assessment"] == "unavailable"
    request = next(
        record
        for record in (
            json.loads(line)
            for line in (tmp_path / "tool-trace.jsonl").read_text().splitlines()
        )
        if record["name"] == "request_builder"
    )
    result = json.loads(request["result"])
    assert result["builder"]["status"] == "unavailable"
    assert "update_goal" in result["required_next_step"]


@pytest.mark.asyncio
async def test_builder_deadline_preserves_the_actor_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An advisory builder must yield while the host can still receive an action."""
    cancelled = asyncio.Event()

    class BlockingBuilderLLM(FakeLLMClient):
        def __init__(self) -> None:
            super().__init__(
                scripted_responses=[
                    tool_response("take_action", {"action": "UP"}, "actor-0"),
                    tool_response("take_action", {"action": "DOWN"}, "actor-1"),
                ]
            )
            self.calls = 0

        async def acall(self, messages, tools=None, output_model=None, **kwargs):
            self.calls += 1
            if self.calls == 2:
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return await super().acall(
                messages, tools=tools, output_model=output_model, **kwargs
            )

    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: TriggerDecision(
            fired=True, cause="dynamics", reason="exact trigger reason"
        ),
    )
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] deadline fixture", None),
    )
    llm = BlockingBuilderLLM()
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
        deadline_config=DecisionDeadlineConfig(
            builder_cap_seconds=0.01,
            actor_reserve_seconds=0.02,
            host_return_reserve_seconds=0.005,
        ),
    )

    await agent.act(turn(0), decision_timeout_seconds=0.1)
    decision = await agent.act(
        turn(1, previous_action="UP"), decision_timeout_seconds=0.1
    )

    assert decision == ActionDecision(action="DOWN")
    assert cancelled.is_set()
    records = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    exhausted = next(
        record for record in records if record["event"] == "builder_deadline_exceeded"
    )
    assert exhausted["actor_reserve_seconds"] == 0.02
    assert exhausted["host_return_reserve_seconds"] == 0.005


@pytest.mark.asyncio
async def test_actor_requested_builder_timeout_is_truthful_and_transactional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled advisory pass must not leave an unverified model current."""

    class PartiallyWritingBuilderLLM(FakeLLMClient):
        def __init__(self) -> None:
            super().__init__(
                scripted_responses=[
                    tool_response(
                        "invoke_builder", {"reason": "model movement"}, "request"
                    ),
                    tool_response(
                        "write_file",
                        {
                            "path": "world_model.py",
                            "content": "# incomplete timed-out model\n",
                        },
                        "partial-write",
                    ),
                    tool_response("take_action", {"action": "UP"}, "commit"),
                ]
            )
            self.calls = 0

        async def acall(self, messages, tools=None, output_model=None, **kwargs):
            self.calls += 1
            if self.calls == 3:
                await asyncio.Event().wait()
            return await super().acall(
                messages, tools=tools, output_model=output_model, **kwargs
            )

    monkeypatch.setenv("TYCHO_MODE", "orchestrator")
    llm = PartiallyWritingBuilderLLM()
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
        deadline_config=DecisionDeadlineConfig(
            builder_cap_seconds=0.02,
            actor_reserve_seconds=0.02,
            host_return_reserve_seconds=0.005,
        ),
    )
    world_model_before = (agent.workspace_dir / "world_model.py").read_bytes()

    decision = await agent.act(turn(0), decision_timeout_seconds=0.1)

    assert decision == ActionDecision(action="UP")
    assert (agent.workspace_dir / "world_model.py").read_bytes() == world_model_before
    tool_records = [
        json.loads(line)
        for line in (tmp_path / "tool-trace.jsonl").read_text().splitlines()
    ]
    request = next(record for record in tool_records if record["name"] == "invoke_builder")
    assert request["status"] == "error"
    assert "deadline exceeded after 1 completed tool call" in request["result"]
    assert "rolled back" in request["result"]
    assert "skipped" not in request["result"]
    trace_records = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    timeout = next(
        record for record in trace_records
        if record["event"] == "builder_deadline_exceeded"
    )
    assert timeout["partial_tool_calls"] == 1
    assert "world_model.py" in timeout["rolled_back_paths"]
    assert timeout["candidate_retention_error"] is None
    retained = timeout["retained_candidate"]
    assert retained is not None
    assert retained["publication_allowed"] is False
    assert retained["planner_eligible"] is False
    assert retained["contemporaneous_score"] == {
        "status": "assessed_without_transition_score",
        "reason": "host verifier ran but no graded transition score was available",
        "source": "host auto world-model feedback",
        "load_status": "loaded",
        "tool_call_index": 1,
        "accepted_transition_accuracy": None,
        "matched_transitions": None,
        "evaluated_transitions": None,
        "observed_transitions": None,
        "first_divergence": None,
        "accepted_level_start_accuracy": 0.0,
        "current_level_initial_render_status": "error",
        "all_levels_initial_render_status": "error",
        "current_level_dynamics_status": "error",
        "all_levels_dynamics_status": "error",
        "current_level_outcome_status": "unobserved",
        "current_level_terminal_render_status": "unobserved",
        "planner_status": "not_run_current_level_initial_render_failed",
    }

    candidate_index = json.loads(
        (agent.workspace_dir / "candidates/world_model/index.json").read_text()
    )
    assert candidate_index["schema_version"] == 1
    assert len(candidate_index["records"]) == 1
    indexed = candidate_index["records"][0]
    assert indexed["record_id"] == retained["record_id"]
    assert indexed["publication_allowed"] is False
    assert indexed["planner_eligible"] is False
    assert (agent.workspace_dir / indexed["model_path"]).read_text() == (
        "# incomplete timed-out model\n"
    )

    candidate_record = json.loads(
        (agent.workspace_dir / indexed["record_path"]).read_text()
    )
    assert candidate_record["authority"] == "retained_noncanonical_candidate"
    assert candidate_record["model_sha256"] == indexed["model_sha256"]
    assert candidate_record["evidence_frontier"] == [
        {
            "path": "level_0/turn_000.json",
            "sha256": hashlib.sha256(
                (agent.workspace_dir / "level_0/turn_000.json").read_bytes()
            ).hexdigest(),
        }
    ]


@pytest.mark.asyncio
async def test_the_llm_trace_covers_the_builder_as_well_as_the_actor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run's model leg includes the builder's calls, on its own event stream.

    The trigger loop creates a fresh agent per fire, so a trace attached only to
    the actor would silently omit every builder round-trip — and the builder is
    where the expensive passes happen.
    """
    llm = FakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            tool_response(
                "write_file",
                {"path": "notes/world_model.md", "content": "Observed: UP moves it."},
                "builder-notes",
            ),
            tool_response(
                "write_file",
                {"path": "notes/world_model_report.md", "content": "note: test UP"},
                "builder-report",
            ),
            prose_response("done"),
            tool_response("take_action", {"action": "UP"}, "actor-1"),
            prose_response("actor checkpoint after turn 1"),
        ]
    )
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: TriggerDecision(
            fired=True, cause="dynamics", reason="exact trigger reason"
        ),
    )
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] test divergence", None),
    )
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )

    await agent.act(turn(0))
    await agent.act(turn(1, previous_action="UP"))

    records = [
        json.loads(line)
        for line in (tmp_path / "llm-trace.jsonl").read_text().splitlines()
    ]
    assert {record["source"] for record in records} == {"actor", "builder"}
    # One shared counter, so the trace orders the whole run's calls, not each
    # agent's separately.
    assert [record["call_index"] for record in records] == list(
        range(1, len(records) + 1)
    )
    checkpoints = sorted((tmp_path / "decision-checkpoints" / "calls").iterdir())
    checkpoint_sources = {
        json.loads((checkpoint / "request.json").read_text())["source"]
        for checkpoint in checkpoints
    }
    assert checkpoint_sources == {"actor", "builder"}


@pytest.mark.asyncio
async def test_trigger_builds_fresh_report_before_actor_and_traces_uptake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = (
        "confidence: medium\n"
        "model: UP changes the marker\n"
        "dynamics: one observed transition\n"
        "outcome: unknown\n"
        "outcome_verified: no-terminal-yet\n"
        "plan: start=current L0 turn 1; validated=no; length=none; assumptions=unknown\n"
        "recommended_action: UP   subgoal: test the encoded transition\n"
        "note: read this report, then test UP once"
    )
    llm = FakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            tool_response(
                "write_file",
                {
                    "path": "notes/world_model.md",
                    "content": "Observed: UP changes the marker.",
                },
                "builder-notes",
            ),
            tool_response(
                "write_file",
                {"path": "notes/world_model_report.md", "content": report},
                "builder-report",
            ),
            prose_response("done"),
            tool_response(
                "read_file",
                {"path": "notes/world_model_report.md"},
                "actor-read-report",
            ),
            tool_response("take_action", {"action": "UP"}, "actor-1"),
        ]
    )
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: TriggerDecision(
            fired=True,
            cause="dynamics",
            reason="exact trigger reason",
        ),
    )
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] test divergence", None),
    )
    trace_path = tmp_path / "tool-trace.jsonl"
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=trace_path,
    )

    assert await agent.act(turn(0)) == ActionDecision(action="UP")
    assert await agent.act(turn(1, previous_action="UP")) == ActionDecision(action="UP")

    assert (agent.workspace_dir / "notes/world_model.md").read_text() == (
        "Observed: UP changes the marker."
    )
    assert (agent.workspace_dir / "notes/world_model_report.md").read_text() == report

    builder_tools = agent._trigger_builder.builder._tool_surface.native_tools()
    assert "take_action" not in {tool.name for tool in builder_tools}
    assert "edit_function" in {tool.name for tool in builder_tools}
    assert agent._trigger_builder.builder.agent_id != agent.agent_id
    # Upstream's prompt, unchanged, plus our local capability briefs. The retained
    # candidate notice makes rolled-back work discoverable without granting it authority.
    assert agent._trigger_builder.builder._resolve_system_prompt() == (
        render_prompt(
            "builder.system",
            colormap=COLORMAP,
            report_file=REPORT_FILE,
            meta_reflect=False,
        )
        + f"\n\n{RETAINED_CANDIDATE_BRIEF}\n\n{OPTIONALITY_PROMPT_BRIEF}\n\n{PROMPT_BRIEF}"
    )
    builder_events = agent._trigger_builder.builder.event_manager.values()
    builder_tasks = [event for event in builder_events if isinstance(event, Task)]
    assert builder_tasks[0].prompt == render_prompt(
        "builder.user",
        level=0,
        hint="exact trigger reason",
        available_actions=["UP", "DOWN"],
        report_file=REPORT_FILE,
    )
    assert builder_tasks[1].prompt == "[verify state] test divergence"
    assert [
        event.name for event in builder_events if isinstance(event, ToolCallEvent)
    ] == ["write_file", "write_file"]
    assert [
        event.name
        for event in agent.event_manager.values()
        if isinstance(event, ToolCallEvent)
    ] == ["take_action", "read_file", "take_action"]
    actor_tasks = [
        event for event in agent.event_manager.values() if isinstance(event, Task)
    ]
    assert report in actor_tasks[-1].prompt

    builder_trace = [
        json.loads(line)
        for line in (tmp_path / "builder-tool-trace.jsonl").read_text().splitlines()
    ]
    assert [entry["name"] for entry in builder_trace] == [
        "write_file",
        "write_file",
    ]
    trigger_trace = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    assert [entry["event"] for entry in trigger_trace] == [
        "mechanical_trigger",
        "builder_result",
        "actor_uptake",
    ]
    assert trigger_trace[0]["fired"] is True
    assert trigger_trace[1]["report_fresh"] is True
    assert trigger_trace[1]["artifact_hashes"]["notes/world_model.md"]["changed"]
    assert trigger_trace[1]["artifact_hashes"]["notes/world_model_report.md"]["changed"]
    assert trigger_trace[2]["followed_recommended_action"] is True
    assert trigger_trace[2]["tested_report"] is True


@pytest.mark.asyncio
async def test_malformed_builder_click_recommendation_does_not_end_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A builder recommendation is advisory and cannot become a host action.

    The first credited ar25 replication wrote a bare ``CLICK`` while putting
    coordinates only in its subgoal prose. Parsing that report constructed an
    invalid ActionDecision before the actor could run and ended the episode.
    """
    report = (
        "confidence: low\n"
        "recommended_action: CLICK | subgoal: try CLICK(16,22) then UP"
    )
    llm = FakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            tool_response(
                "write_file",
                {"path": "notes/world_model.md", "content": "UP changed the grid."},
                "builder-notes",
            ),
            tool_response(
                "write_file",
                {"path": "notes/world_model_report.md", "content": report},
                "builder-report",
            ),
            prose_response("done"),
            tool_response("take_action", {"action": "DOWN"}, "actor-1"),
        ]
    )
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: TriggerDecision(
            fired=True, cause="dynamics", reason="exact trigger reason"
        ),
    )
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] test divergence", None),
    )
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN", "CLICK"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )
    first = turn(0).model_copy(update={"available_actions": ["UP", "DOWN", "CLICK"]})
    second = turn(1, previous_action="UP").model_copy(
        update={"available_actions": ["UP", "DOWN", "CLICK"]}
    )

    assert await agent.act(first) == ActionDecision(action="UP")
    assert await agent.act(second) == ActionDecision(action="DOWN")

    trace = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    assert trace[1]["recommended_action"] is None
    assert "CLICK requires row and col" in trace[1]["recommendation_parse_error"]


def test_recommended_action_parses_click_coordinates_on_the_same_field() -> None:
    available = ["UP", "DOWN", "CLICK"]
    parsed = _recommended_action(
        "candidate_goal_assessment: inconclusive\nrecommended_action: CLICK row 19 col 22\n",
        available,
    )
    assert parsed.error is None
    assert parsed.action == ActionDecision(action="CLICK", row=19, col=22)
    assert (
        _recommended_action(
            "recommended_action: CLICK(row=16, col=22)\n",
            available,
        ).action
        == ActionDecision(action="CLICK", row=16, col=22)
    )
    bare = _recommended_action(
        "recommended_action: CLICK | subgoal: try CLICK(16,22) then UP",
        available,
    )
    assert bare.action is None
    assert bare.error is not None
    assert "CLICK requires row and col" in bare.error


@pytest.mark.asyncio
async def test_stale_or_missing_report_is_replaced_by_tool_free_final_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = (
        "confidence: low\nmodel: none — insufficient evidence\n"
        "dynamics: n/a\noutcome: unknown\noutcome_verified: no-terminal-yet\n"
        "plan: start=current; validated=no; length=none; assumptions=unknown\n"
        "recommended_action: none   subgoal: gather one more transition\n"
        "note: test DOWN"
    )
    llm = FakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            prose_response("done"),
            prose_response("nothing new yet"),
            prose_response(fallback),
            tool_response("take_action", {"action": "DOWN"}, "actor-1"),
        ]
    )
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: TriggerDecision(
            fired=True,
            cause="dynamics",
            reason="exact trigger reason",
        ),
    )
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] test divergence", None),
    )
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )
    stale_path = agent.workspace_dir / "notes/world_model_report.md"
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text("confidence: high\nnote: stale prior report")

    await agent.act(turn(0))
    assert await agent.act(turn(1, previous_action="UP")) == ActionDecision(
        action="DOWN"
    )

    assert (agent.workspace_dir / "notes/world_model_report.md").read_text() == fallback
    trace = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    assert trace[1]["report_fresh"] is True
    assert trace[1]["report_forced"] is True
    assert trace[1]["world_model_checkpoint_forced"] is True
    assert trace[1]["artifact_hashes"]["notes/world_model.md"]["changed"] is True


@pytest.mark.asyncio
async def test_forced_report_identical_to_prior_report_is_not_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for DIAGNOSIS.md: turns 11/19 forced-reported `report_fresh=true`
    while the report bytes were byte-identical to the prior pass. Forcing a final
    report is a claim about which code path ran, not about whether anything new
    was learned; only an actual content change should count as fresh."""
    llm = FakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            prose_response("done"),
            prose_response("nothing new yet"),
            prose_response("done"),
            tool_response("take_action", {"action": "DOWN"}, "actor-1"),
        ]
    )
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: TriggerDecision(
            fired=True,
            cause="dynamics",
            reason="exact trigger reason",
        ),
    )
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] test divergence", None),
    )
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )
    report_path = agent.workspace_dir / "notes/world_model_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("done")

    await agent.act(turn(0))
    assert await agent.act(turn(1, previous_action="UP")) == ActionDecision(
        action="DOWN"
    )

    assert report_path.read_text() == "done"
    trace = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    assert trace[1]["report_forced"] is True
    assert trace[1]["report_fresh"] is False
    assert (
        trace[1]["artifact_hashes"]["notes/world_model_report.md"]["changed"] is False
    )


@pytest.mark.asyncio
async def test_forced_report_call_offers_tools_but_ignores_attempted_tool_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matches Tycho's builder.py forced-report fallback: the final generate
    call still offers tools (it is not hard-blocked via tools=[]/tool_choice=
    "none"), but Tycho's handling of that reply only reads reply["text"] and
    never executes reply["tool_calls"] — a tool-call attempt there is rejected
    rather than structurally disallowed. If the model tries to call a tool anyway
    instead of replying in prose, the forced pass must fall back to the fixed
    report template while retaining a terminal error row for the attempt."""
    llm = RecordingFakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            prose_response("still exploring"),
            prose_response("nothing new yet"),
            tool_response(
                "write_file",
                {"path": "notes/world_model_report.md", "content": "sneaky"},
                "forced-attempt",
            ),
            tool_response("take_action", {"action": "DOWN"}, "actor-1"),
        ]
    )
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: TriggerDecision(
            fired=True,
            cause="dynamics",
            reason="exact trigger reason",
        ),
    )
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] test divergence", None),
    )
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )

    await agent.act(turn(0))
    assert await agent.act(turn(1, previous_action="UP")) == ActionDecision(
        action="DOWN"
    )

    # Call order: actor turn 0, builder main loop, builder world-model
    # checkpoint enforcement, builder forced final call, actor turn 1.
    checkpoint_call_tools = llm.tools_per_call[2]
    assert checkpoint_call_tools, "checkpoint call must offer tools too"
    assert "take_action" not in checkpoint_call_tools

    forced_call_tools = llm.tools_per_call[3]
    assert forced_call_tools, "forced call must still offer tools, like Tycho does"
    assert "take_action" not in forced_call_tools

    report_path = agent.workspace_dir / "notes/world_model_report.md"
    assert report_path.read_text() == FALLBACK_REPORT

    # The checkpoint step offered no tool calls, so the fallback checkpoint text
    # (not the "sneaky" write) is what advances notes/world_model.md. The forced
    # final call's tool-call attempt is never executed regardless.
    world_model_notes = agent.workspace_dir / "notes/world_model.md"
    assert world_model_notes.read_text() != "sneaky"

    builder_trace = [
        json.loads(line)
        for line in (tmp_path / "builder-tool-trace.jsonl").read_text().splitlines()
    ]
    assert builder_trace == [
        {
            "schema_version": 2,
            "call_index": 1,
            "tool_call_id": "forced-attempt",
            "source": "builder",
            "phase": "decision",
            "sequence": None,
            "level": 0,
            "turn_in_level": 1,
            "name": "write_file",
            "raw_arguments": json.dumps(
                {
                    "path": "notes/world_model_report.md",
                    "content": "sneaky",
                }
            ),
            "arguments": {
                "path": "notes/world_model_report.md",
                "content": "sneaky",
            },
            "parse_status": "complete",
            "validation_status": "not_attempted",
            "validated_arguments": None,
            "execution_status": "not_attempted",
            "error_stage": "policy",
            "status": "error",
            "result": (
                "Builder final-report phase accepts prose only; tool call was not "
                "executed"
            ),
        }
    ]

    trace = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    assert trace[1]["report_forced"] is True
    assert trace[1]["world_model_checkpoint_forced"] is True


@pytest.mark.asyncio
async def test_world_model_checkpoint_is_forced_when_pass_leaves_it_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for DIAGNOSIS.md: notes/world_model.md was read on all 12
    post-turn-8 builder firings in the first live run but never updated, so every
    later pass repeated the same stale turn-1 scratchpad. A pass that leaves this
    file untouched must not be allowed to end quietly."""
    fallback_report = (
        "confidence: low\nmodel: none — insufficient evidence\n"
        "dynamics: n/a\noutcome: unknown\noutcome_verified: no-terminal-yet\n"
        "plan: start=current; validated=no; length=none; assumptions=unknown\n"
        "recommended_action: none   subgoal: gather one more transition\n"
        "note: test DOWN"
    )
    llm = FakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            prose_response("done"),
            prose_response("nothing new yet"),
            prose_response(fallback_report),
            tool_response("take_action", {"action": "DOWN"}, "actor-1"),
        ]
    )
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: TriggerDecision(
            fired=True, cause="dynamics", reason="exact trigger reason"
        ),
    )
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] test divergence", None),
    )
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )
    notes_path = agent.workspace_dir / "notes/world_model.md"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text("turn 1: UP moved the marker\n")

    await agent.act(turn(0))
    assert await agent.act(turn(1, previous_action="UP")) == ActionDecision(
        action="DOWN"
    )

    updated = notes_path.read_text()
    assert updated.startswith("turn 1: UP moved the marker")
    assert "[checkpoint: pass ended with no model-authored update]" in updated

    builder_tasks = [
        event
        for event in agent._trigger_builder.builder.event_manager.values()
        if isinstance(event, Task)
    ]
    assert any(
        "Before this pass ends, checkpoint your current understanding" in task.prompt
        for task in builder_tasks
    )

    trace = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    assert trace[1]["world_model_checkpoint_forced"] is True
    assert trace[1]["artifact_hashes"]["notes/world_model.md"]["changed"] is True


@pytest.mark.asyncio
async def test_world_model_checkpoint_executes_a_genuine_model_edit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unlike the forced-report call, the checkpoint call's tool calls are real:
    the model gets one genuine chance to write before the fallback breadcrumb
    would apply, and that write must actually land."""
    llm = FakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            prose_response("done"),
            tool_response(
                "write_file",
                {
                    "path": "notes/world_model.md",
                    "content": "genuine checkpoint: UP moves the marker up",
                },
                "checkpoint-write",
            ),
            prose_response("done"),
            tool_response("take_action", {"action": "DOWN"}, "actor-1"),
        ]
    )
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: TriggerDecision(
            fired=True, cause="dynamics", reason="exact trigger reason"
        ),
    )
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] test divergence", None),
    )
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )

    await agent.act(turn(0))
    assert await agent.act(turn(1, previous_action="UP")) == ActionDecision(
        action="DOWN"
    )

    notes_path = agent.workspace_dir / "notes/world_model.md"
    content = notes_path.read_text()
    assert content == "genuine checkpoint: UP moves the marker up"
    assert "[checkpoint: pass ended" not in content

    trace = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    assert trace[1]["world_model_checkpoint_forced"] is True


@pytest.mark.asyncio
async def test_level_start_fires_only_on_genuine_level_advance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the same-level `full_reset` ambiguity: arc-host resets
    turn_in_level to 0 WITHOUT changing level (runner.py), so turn_in_level ==
    0 alone cannot distinguish a genuine level advance from a same-level
    restart. Only a level that actually differs from the immediately
    preceding turn's level (`_last_level`) may fire the unconditional
    LEVEL_START cause."""
    calls: list[str] = []
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate_level_start",
        lambda self, level: (
            calls.append(f"level_start:{level}")
            or TriggerDecision(fired=False, cause="level_start", reason="entered level")
        ),
    )
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: (
            calls.append(f"evaluate:{level}")
            or TriggerDecision(fired=False, cause=None, reason=None)
        ),
    )
    llm = FakeLLMClient(scripted_responses=[])
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )
    loop = agent._trigger_builder
    assert loop is not None

    def level_turn(level: int, turn_in_level: int) -> GameTurn:
        return GameTurn(
            grid=[[0, 1], [2, 3]],
            available_actions=["UP", "DOWN"],
            level=level,
            turn_in_level=turn_in_level,
            previous_action="UP" if turn_in_level > 0 else "",
        )

    # Level 0's own turn_in_level == 0 is guarded out unconditionally (level > 0
    # is required), so it neither fires nor consumes a scripted response.
    assert await loop.before_actor(level_turn(0, 0)) is None
    # A genuine advance into level 1 fires level_start exactly once.
    assert await loop.before_actor(level_turn(1, 0)) is None
    # arc-host's full_reset recurs turn_in_level == 0 at the SAME level — this
    # must not be mistaken for a second level-start.
    assert await loop.before_actor(level_turn(1, 0)) is None

    assert calls == ["level_start:1"]

    trace = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    assert [entry["mechanism"] for entry in trace] == [
        "tycho TriggerDispatcher.level_start_reason (unconditional)"
    ]
    assert trace[0]["level"] == 1
    assert trace[0]["turn_in_level"] == 0


@pytest.mark.asyncio
async def test_level_start_passes_authoritative_completion_evidence_to_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[BuilderRequest] = []

    class CapturingBuilder:
        agent_id = "capturing-builder"

        def __init__(self, **kwargs):
            self.event_manager = None

        async def build(self, request: BuilderRequest) -> BuilderResult:
            captured.append(request)
            resolved = len(captured) == 2
            return BuilderResult(
                report="recommended_action: none",
                report_fresh=False,
                report_forced=False,
                world_model_checkpoint_forced=False,
                outcome_checkpoint_required=request.outcome_evidence is not None,
                outcome_checkpoint_resolved=resolved,
                artifact_hashes={},
                tool_calls=[],
            )

    monkeypatch.setattr("tycho_nooa.trigger_builder.FocusedBuilder", CapturingBuilder)
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: TriggerDecision(
            fired=True, cause="dynamics", reason="transition mismatch"
        ),
    )
    llm = FakeLLMClient(scripted_responses=[])
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )
    evidence_path = agent.workspace_dir / "level_0/level_complete.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(
            {
                "source": "arc-host action_result",
                "outcome": "level_complete",
                "action": "UP",
                "levels_completed_before": 0,
                "levels_completed_after": 1,
            }
        )
    )
    loop = agent._trigger_builder
    assert loop is not None
    loop._llm_trace = None
    assert await loop.before_actor(turn(0)) is None
    level_one = GameTurn(
        grid=[[1, 2], [3, 4]],
        available_actions=["UP", "DOWN"],
        level=1,
        turn_in_level=0,
    )

    await loop.before_actor(level_one)

    assert len(captured) == 1
    assert captured[0].outcome_evidence is not None
    assert '"outcome":"level_complete"' in captured[0].outcome_evidence

    # An unresolved checkpoint persists across later dynamics-triggered passes.
    later = GameTurn(
        grid=[[1, 2], [3, 5]],
        available_actions=["UP", "DOWN"],
        level=1,
        turn_in_level=1,
        previous_action="UP",
    )
    await loop.before_actor(later)
    assert captured[1].outcome_evidence == captured[0].outcome_evidence

    # Once a builder reports verification success, following passes return to
    # ordinary dynamics work without carrying the completed checkpoint forever.
    latest = later.model_copy(update={"turn_in_level": 2})
    await loop.before_actor(latest)
    assert captured[2].outcome_evidence is None


@pytest.mark.asyncio
async def test_host_completion_runs_checkpoint_without_requiring_another_actor_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[BuilderRequest] = []

    class CapturingBuilder:
        agent_id = "completion-builder"

        def __init__(self, **kwargs):
            self.event_manager = None

        async def build(self, request: BuilderRequest) -> BuilderResult:
            captured.append(request)
            return BuilderResult(
                report="recommended_action: none",
                report_fresh=True,
                report_forced=False,
                world_model_checkpoint_forced=False,
                outcome_checkpoint_required=True,
                outcome_checkpoint_resolved=True,
                artifact_hashes={},
                tool_calls=[],
            )

    monkeypatch.setattr("tycho_nooa.trigger_builder.FocusedBuilder", CapturingBuilder)
    agent = TychoNooaAgent(
        llm=FakeLLMClient(scripted_responses=[]),
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )
    loop = agent._trigger_builder
    assert loop is not None
    loop._llm_trace = None
    completion_turn = GameTurn(
        grid=[[1, 2], [3, 4]],
        available_actions=["UP", "DOWN"],
        level=0,
        turn_in_level=20,
        levels_completed=1,
        previous_action="LEFT",
    )

    await loop.after_level_complete(
        completion_turn,
        '{"source":"arc-host action_result","outcome":"level_complete"}',
    )

    assert len(captured) == 1
    assert captured[0].turn.turn_in_level == 20
    assert captured[0].outcome_evidence is not None
    assert captured[0].phase == "post_action"
    assert loop._pending_outcome_evidence is None
    # The subsequent level-start pass is ordinary dynamics work, not a second
    # repair of completion evidence that the immediate checkpoint resolved.
    await loop.before_actor(
        GameTurn(
            grid=[[1, 2], [3, 4]],
            available_actions=["UP", "DOWN"],
            level=1,
            turn_in_level=0,
            levels_completed=1,
        )
    )
    assert len(captured) == 2
    assert captured[1].outcome_evidence is None
    assert captured[1].phase == "decision"


@pytest.mark.asyncio
async def test_unresolved_completion_is_the_latest_builder_priority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = MessageRecordingFakeLLMClient(
        scripted_responses=[
            prose_response("I would rather inspect current dynamics."),
            prose_response("Still no tool call."),
            prose_response("nothing new yet"),
            prose_response("recommended_action: none"),
        ]
    )
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: (
            (
                "OUTCOME_STATUS: fail\npost-winning-action missed terminal: "
                "outcome() did not say level_complete\nREPAIR_FOCUS: current_level_dynamics"
            ),
            None,
        ),
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )
    builder = FocusedBuilder(
        llm=llm,
        executor=agent._tool_executor,
        trace_path=tmp_path / "builder-tool-trace.jsonl",
        config=BuilderConfig(max_tool_steps=2),
    )
    level_one = GameTurn(
        grid=[[1, 2], [3, 4]],
        available_actions=["UP", "DOWN"],
        level=1,
        turn_in_level=0,
    )

    result = await builder.build(
        BuilderRequest(
            turn=level_one,
            reason="investigate the new level",
            outcome_evidence='{"outcome":"level_complete","action":"UP"}',
        )
    )

    first_prompts = [
        message["content"]
        for message in llm.messages_per_call[0]
        if message.get("role") == "user"
    ]
    assert "HOST-CONFIRMED OUTCOME CHECKPOINT" in first_prompts[-1]
    assert "Do not investigate current-level dynamics first" in first_prompts[-1]
    assert all(
        "BUILDER BUDGET" not in str(message.get("content"))
        for messages in llm.messages_per_call
        for message in messages
    )
    # A prose refusal cannot silently end a mandatory repair prefix; both
    # configured construction rounds must be spent before fallback handling.
    assert len(llm.messages_per_call) == 4
    assert result.outcome_checkpoint_required is True
    assert result.outcome_checkpoint_resolved is False


@pytest.mark.asyncio
async def test_builder_malformed_arguments_preserve_parse_failure(
    tmp_path: Path,
) -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            malformed_tool_response("write_file", "{broken", "builder-bad-json"),
            prose_response("recommended_action: none"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )
    builder = FocusedBuilder(
        llm=llm,
        executor=agent._tool_executor,
        trace_path=tmp_path / "builder-tool-trace.jsonl",
        config=BuilderConfig(max_tool_steps=1),
    )

    await builder.build(
        BuilderRequest(
            turn=GameTurn(
                grid=[[1, 2], [3, 4]],
                available_actions=["UP", "DOWN"],
                level=0,
                turn_in_level=0,
            ),
            reason="test malformed arguments",
        )
    )

    trace = json.loads(
        (tmp_path / "builder-tool-trace.jsonl").read_text().splitlines()[0]
    )
    assert trace["source"] == "builder"
    assert trace["raw_arguments"] == "{broken"
    assert trace["parse_status"] == "error"
    assert trace["validation_status"] == "not_attempted"
    assert trace["execution_status"] == "not_attempted"
    assert trace["error_stage"] == "parse"


def test_builder_budget_commit_threshold_is_runtime_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tycho_nooa.builder.monotonic", lambda: 100.0)
    notice = _builder_budget_notice(
        BuilderRequest(
            turn=turn(0),
            reason="repair",
            deadline_at=250.0,
        ),
        repair_calls_remaining=10,
        commit_seconds_remaining=180,
    )

    assert "Time remaining: 150 seconds" in notice
    assert "Mode: commit" in notice


@pytest.mark.asyncio
async def test_builder_sees_fresh_time_and_call_budget_before_every_repair_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = MessageRecordingFakeLLMClient(
        scripted_responses=[
            tool_response(
                "edit_function",
                {
                    "path": "world_model.py",
                    "name": "transition",
                    "code": "def transition(state, action):\n    return state\n",
                },
                "minimal-edit",
            ),
            tool_response("run_python", {"code": "print('probe 2')"}, "probe-2"),
            tool_response("run_python", {"code": "print('probe 3')"}, "probe-3"),
            tool_response("run_python", {"code": "print('probe 4')"}, "probe-4"),
            prose_response("recommended_action: none"),
        ]
    )
    monotonic = iter([100.0, 120.0, 150.0, 180.0])
    monkeypatch.setattr("tycho_nooa.builder.monotonic", lambda: next(monotonic))
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: ("[verify state] first divergence remains", None),
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )
    builder = FocusedBuilder(
        llm=llm,
        executor=agent._tool_executor,
        trace_path=tmp_path / "builder-tool-trace.jsonl",
        config=BuilderConfig(max_tool_steps=4, show_budget_telemetry=True),
    )

    await builder.build(
        BuilderRequest(
            turn=turn(0),
            reason="repair the first divergence",
            deadline_at=240.0,
        )
    )

    notices = []
    for messages in llm.messages_per_call[:4]:
        matching = [
            message["content"]
            for message in messages
            if message.get("role") == "user"
            and "BUILDER BUDGET" in str(message.get("content"))
        ]
        notices.append(matching[-1])

    assert "Time remaining: 140 seconds" in notices[0]
    assert "Repair model calls remaining (including this call): 4" in notices[0]
    assert "Mode: investigate" in notices[0]
    assert "Time remaining: 120 seconds" in notices[1]
    assert "Repair model calls remaining (including this call): 3" in notices[1]
    assert "Mode: commit" in notices[1]
    assert "Make the smallest reversible edit now" in notices[1]
    assert "Time remaining: 60 seconds" in notices[3]
    assert "Repair model calls remaining (including this call): 1" in notices[3]


@pytest.mark.asyncio
async def test_host_builder_allocation_reaches_the_model_visible_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[BuilderRequest] = []

    class CapturingBuilder:
        agent_id = "budget-capturing-builder"

        def __init__(self, **kwargs):
            self.event_manager = None

        async def build(self, request: BuilderRequest) -> BuilderResult:
            captured.append(request)
            return BuilderResult(
                report="recommended_action: none",
                report_fresh=True,
                report_forced=False,
                world_model_checkpoint_forced=False,
                outcome_checkpoint_required=False,
                outcome_checkpoint_resolved=True,
                artifact_hashes={},
                tool_calls=[],
            )

    monkeypatch.setattr("tycho_nooa.trigger_builder.FocusedBuilder", CapturingBuilder)
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: TriggerDecision(
            fired=True,
            cause="dynamics",
            reason="repair the first divergence",
        ),
    )
    agent = TychoNooaAgent(
        llm=FakeLLMClient(
            scripted_responses=[
                tool_response("take_action", {"action": "UP"}, "commit")
            ]
        ),
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        deadline_config=DecisionDeadlineConfig(
            builder_cap_seconds=0.2,
            actor_reserve_seconds=0.2,
            host_return_reserve_seconds=0.01,
        ),
    )
    assert agent._trigger_builder is not None
    agent._trigger_builder._llm_trace = None
    started = time.monotonic()

    await agent.act(turn(1, previous_action="UP"), decision_timeout_seconds=1.0)

    assert len(captured) == 1
    assert captured[0].deadline_at is not None
    assert started < captured[0].deadline_at <= started + 0.25


@pytest.mark.asyncio
async def test_world_model_edit_rechecks_and_resolves_the_outcome_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            tool_batch_response(
                [
                    (
                        "write_file",
                        {
                            "path": "world_model.py",
                            "content": 'def outcome(state):\n    return "level_complete"\n',
                        },
                        "repair-outcome",
                    ),
                    (
                        "write_file",
                        {
                            "path": "notes/world_model.md",
                            "content": "Host evidence now has a terminal predicate.",
                        },
                        "checkpoint-notes",
                    ),
                ]
            ),
            prose_response("recommended_action: none"),
        ]
    )
    verifier_results = iter(
        [
            "post-winning-action missed terminal",
            "OUTCOME_STATUS: pass",
        ]
    )
    monkeypatch.setattr(
        "tycho_nooa.builder.verify_state",
        lambda executor: (next(verifier_results), None),
    )
    agent = TychoNooaAgent(
        llm=llm,
        trigger_builder=False,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
    )
    builder = FocusedBuilder(
        llm=llm,
        executor=agent._tool_executor,
        trace_path=tmp_path / "builder-tool-trace.jsonl",
        config=BuilderConfig(max_tool_steps=1),
    )

    result = await builder.build(
        BuilderRequest(
            turn=GameTurn(
                grid=[[1, 2], [3, 4]],
                available_actions=["UP", "DOWN"],
                level=1,
                turn_in_level=0,
            ),
            reason="repair the host-confirmed terminal",
            outcome_evidence='{"outcome":"level_complete","action":"UP"}',
        )
    )

    assert result.outcome_checkpoint_required is True
    assert result.outcome_checkpoint_resolved is True
    assert (
        (agent.workspace_dir / "world_model.py")
        .read_text()
        .endswith('return "level_complete"\n')
    )


def test_actor_error_after_builder_exposure_is_preserved_in_the_trace(
    tmp_path: Path,
) -> None:
    agent = TychoNooaAgent(
        llm=FakeLLMClient(scripted_responses=[]),
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )
    loop = agent._trigger_builder
    assert loop is not None
    exposure = BuilderExposure(
        result=BuilderResult(
            report="recommended_action: UP",
            report_fresh=True,
            report_forced=False,
            world_model_checkpoint_forced=False,
            outcome_checkpoint_required=False,
            outcome_checkpoint_resolved=True,
            artifact_hashes={},
            tool_calls=[],
        ),
        recommended_action=ActionDecision(action="UP"),
    )

    loop.after_actor_error(turn(0), None, RuntimeError("ignored"), [])
    loop.after_actor_error(
        turn(1), exposure, RuntimeError("actor failed"), tool_calls=[]
    )

    trace = json.loads((tmp_path / "trigger-builder-trace.jsonl").read_text())
    assert trace["event"] == "actor_uptake"
    assert trace["actor_action"] is None
    assert trace["recommended_action"]["action"] == "UP"
    assert trace["error_type"] == "RuntimeError"
    assert trace["message"] == "actor failed"


@pytest.mark.asyncio
async def test_unavailable_sandbox_degrades_the_trigger_instead_of_ending_the_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The divergence probe runs verify.py in Tycho's sandbox. When no container
    runtime exists it raises on the actor's path, where the same fault is only a
    tool result — that killed a whole live run at action 1. An unanswerable probe
    must be traced as absence of signal and let the actor's turn proceed."""

    def unavailable(self, level, executor):
        raise RuntimeError("no usable container runtime found")

    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate", unavailable
    )
    llm = FakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            tool_response("take_action", {"action": "DOWN"}, "actor-1"),
            prose_response("actor checkpoint after turn 1"),
        ]
    )
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )

    await agent.act(turn(0))
    assert await agent.act(turn(1, previous_action="UP")) == ActionDecision(
        action="DOWN"
    )

    trace = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    assert [entry["event"] for entry in trace] == ["trigger_error"]
    assert trace[0]["error_type"] == "RuntimeError"
    assert "no usable container runtime" in trace[0]["message"]
    assert trace[0]["turn_in_level"] == 1
    assert not (agent.workspace_dir / "notes/world_model_report.md").exists()


@pytest.mark.asyncio
async def test_clean_signal_does_not_dispatch_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            tool_response("take_action", {"action": "UP"}, "actor-0"),
            tool_response("take_action", {"action": "DOWN"}, "actor-1"),
        ]
    )
    monkeypatch.setattr(
        "tycho_nooa.trigger_builder.TychoTransitionTrigger.evaluate",
        lambda self, level, executor: TriggerDecision(
            fired=False,
            cause=None,
            reason=None,
        ),
    )
    agent = TychoNooaAgent(
        llm=llm,
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        tool_trace_path=tmp_path / "tool-trace.jsonl",
    )

    await agent.act(turn(0))
    assert await agent.act(turn(1, previous_action="UP")) == ActionDecision(
        action="DOWN"
    )

    assert not (agent.workspace_dir / "notes/world_model_report.md").exists()
    trace = [
        json.loads(line)
        for line in (tmp_path / "trigger-builder-trace.jsonl").read_text().splitlines()
    ]
    assert trace == [
        {
            "event": "mechanical_trigger",
            "mechanism": "tycho TriggerDispatcher.should_fire -> wm_signal",
            "level": 0,
            "turn_in_level": 1,
            "fired": False,
            "cause": None,
            "reason": None,
        }
    ]


def _edit(call_index: int, result: str) -> dict:
    return {
        "call_index": call_index,
        "status": "complete",
        "name": "edit_function",
        "arguments": {"path": "world_model.py", "name": "transition"},
        "result": result,
    }


GRADED_FEEDBACK = (
    "edit_function: replaced `transition` in world_model.py (lines 62-87).\n\n"
    "[auto world-model feedback]\n"
    "DYNAMICS_METRICS:\n"
    "  exact_transition_accuracy (simulation_accuracy) = 0.50 <- exact match rate\n"
    "  transition_replay = matched 8 / evaluated 16 / observed 18 transitions\n"
    "  first divergence: L0 turn 4 RIGHT\n"
)

SKIPPED_FEEDBACK = (
    "edit_function: replaced `transition` in world_model.py (lines 62-86).\n\n"
    "[auto world-model feedback]\n"
    f"{AST_UNCHANGED_NOTICE} (comments/formatting only)"
)


def test_a_cosmetic_final_edit_does_not_erase_the_candidate_score() -> None:
    """The verifier is skipped on an AST-preserving edit, not run and found empty.

    Reading the score off the last edit rather than the last *graded* edit lost the
    only positive score in a preserved pass, which is exactly the selection
    inversion candidate retention exists to prevent.
    """
    score = _contemporaneous_score(
        [_edit(1, GRADED_FEEDBACK), _edit(2, SKIPPED_FEEDBACK)]
    )

    assert score["status"] == "measured"
    assert score["matched_transitions"] == 8
    assert score["evaluated_transitions"] == 16
    assert score["first_divergence"] == "L0 turn 4 RIGHT"
    # The retained bytes are not the graded bytes; the record has to say so.
    assert score["tool_call_index"] == 1
    assert score["ast_preserving_edits_since_grade"] == 1


def test_a_graded_final_edit_claims_no_ast_preserving_gap() -> None:
    score = _contemporaneous_score([_edit(1, SKIPPED_FEEDBACK), _edit(2, GRADED_FEEDBACK)])

    assert score["status"] == "measured"
    assert score["tool_call_index"] == 2
    assert "ast_preserving_edits_since_grade" not in score


def test_an_entirely_cosmetic_pass_reports_an_ungraded_body() -> None:
    score = _contemporaneous_score([_edit(1, SKIPPED_FEEDBACK), _edit(2, SKIPPED_FEEDBACK)])

    assert score["status"] == "unavailable"
    assert "never graded" in score["reason"]
    assert score["ast_preserving_edits_since_grade"] == 2


@pytest.mark.parametrize(
    ("field", "heading"),
    [
        ("gained", "gained obligations:"),
        ("lost", "lost obligations:"),
        ("inconclusive", "inconclusive obligations:"),
    ],
)
def test_publication_feedback_names_every_typed_delta(
    field: str,
    heading: str,
) -> None:
    values = {field: ("transition:L0:T4:RIGHT",)}
    decision = PublicationDecision(
        PublicationDisposition.WITHHELD_INCONCLUSIVE,
        reason="fixture decision",
        **values,
    )

    feedback = _publication_feedback(decision)

    assert "[host theory publication decision — authoritative]" in feedback
    assert "disposition: withheld_inconclusive" in feedback
    assert heading in feedback
    assert "- transition:L0:T4:RIGHT" in feedback
    assert "reason: fixture decision" in feedback
