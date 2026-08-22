"""The direct-reasoning baseline: what a run carries when no world model exists."""

from __future__ import annotations

from pathlib import Path

import pytest
from nooa.unifiedllm import FakeLLMClient
from tycho.agent.modes import ModeSpec

from tycho_nooa import TychoNooaAgent
from tycho_nooa.modes import DEFAULT_MODE, MODE_ENV, mode_from_env, served_mode
from tycho_nooa.planner import actor_world_model_paragraph
from tycho_nooa.treatments import TREATMENT_ENV


def test_an_unset_environment_runs_the_topology_every_run_so_far_used() -> None:
    assert mode_from_env({}).name == DEFAULT_MODE == "trigger"
    assert mode_from_env({MODE_ENV: ""}).name == "trigger"


def test_the_baseline_is_served_and_fires_no_builder() -> None:
    spec = mode_from_env({MODE_ENV: "no_world_model"})

    assert spec.wm_variant == "none"
    assert spec.needs_builder is False


def test_orchestrator_is_an_actor_pull_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MODE_ENV, "orchestrator")

    agent = _agent(tmp_path)
    policy = served_mode("orchestrator")

    assert policy.include_builder is True
    assert policy.harness_fires_builder is False
    assert policy.actor_writes_wm is False
    assert "invoke_builder" in {
        tool.name for tool in agent._tool_surface.native_tools()
    }
    assert "you do not write `world_model.py`" in actor_world_model_paragraph("orch")


def test_typed_goals_are_named_as_a_treatment_not_an_orchestrator_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MODE_ENV, "orchestrator")
    exact = _agent(tmp_path / "exact")
    exact_names = {tool.name for tool in exact._tool_surface.native_tools()}

    monkeypatch.setenv(TREATMENT_ENV, "typed_candidate_goals")
    treatment = _agent(tmp_path / "treatment")
    treatment_names = {tool.name for tool in treatment._tool_surface.native_tools()}

    assert exact.treatment_name == "control"
    assert "invoke_builder" in exact_names
    assert "propose_goal" not in exact_names
    assert treatment.treatment_name == "typed_candidate_goals"
    assert "invoke_builder" not in treatment_names
    assert {
        "propose_goal",
        "update_goal",
        "list_candidate_goals",
        "request_builder",
    } <= treatment_names
    assert "propose a falsifiable candidate goal" in treatment._resolve_system_prompt()
    assert treatment._trigger_builder is not None
    assert treatment._trigger_builder._builder_config.max_tool_steps == 10
    assert exact._trigger_builder is not None
    assert exact._trigger_builder._builder_config.max_tool_steps == 40


def test_typed_goals_cannot_silently_change_another_source_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MODE_ENV, "trigger")
    monkeypatch.setenv(TREATMENT_ENV, "typed_candidate_goals")

    with pytest.raises(ValueError, match="requires TYCHO_MODE='orchestrator'"):
        _agent(tmp_path)


def test_transactional_builder_is_a_named_treatment_not_the_trigger_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MODE_ENV, "trigger")
    control = _agent(tmp_path / "control")

    monkeypatch.setenv(TREATMENT_ENV, "transactional_builder")
    treatment = _agent(tmp_path / "treatment")

    assert control.treatment_name == "control"
    assert treatment.treatment_name == "transactional_builder"
    assert treatment._trigger_builder is not None
    assert treatment._trigger_builder._theory_evaluator is not None
    assert treatment._trigger_builder._builder_config.max_tool_steps == 12
    assert control._trigger_builder is not None
    assert control._trigger_builder._builder_config.max_tool_steps == 40


def test_transactional_builder_requires_a_builder_owned_source_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MODE_ENV, "single")
    monkeypatch.setenv(TREATMENT_ENV, "transactional_builder")

    with pytest.raises(ValueError, match="requires a Builder-owned source mode"):
        _agent(tmp_path)


@pytest.mark.parametrize("mode", ["trigger", "no_world_model"])
def test_builder_owned_modes_block_actor_world_model_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv(MODE_ENV, mode)
    agent = _agent(tmp_path)

    with pytest.raises(PermissionError, match="world_model.py"):
        agent._tool_surface.write_file("world_model.py", "actor-owned contamination")


def test_single_mode_allows_actor_world_model_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MODE_ENV, "single")
    agent = _agent(tmp_path)

    agent._tool_surface.write_file("world_model.py", "actor-owned model")

    assert (agent.workspace_dir / "world_model.py").read_text() == "actor-owned model"
    assert "you own `world_model.py`" in actor_world_model_paragraph("single")
    assert "edit_function" in {
        tool.name for tool in agent._tool_surface.native_tools()
    }


def test_an_unknown_mode_is_still_upstreams_error() -> None:
    with pytest.raises(ValueError, match="not a recognized mode"):
        served_mode("no_builder")


def test_a_future_upstream_mode_does_not_become_locally_served_by_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future = ModeSpec(
        "future",
        wm_variant="none",
        include_builder=False,
        actor_writes_wm=False,
    )
    monkeypatch.setattr("tycho_nooa.modes.resolve_mode", lambda _name: future)

    with pytest.raises(NotImplementedError, match="ownership contract is unaudited"):
        served_mode("future")


def test_the_baseline_actor_is_never_told_a_builder_exists() -> None:
    baseline = actor_world_model_paragraph("none")
    trigger = actor_world_model_paragraph("trigger")

    assert "builder" not in baseline.lower()
    assert "world_model.py" not in baseline
    assert "builder" in trigger.lower()
    # It is not the trigger prompt minus a sentence: the baseline keeps its own
    # durable-notes channel and its own hypothesis-distinguishing instruction.
    assert "notes/actor_beliefs.md" in baseline
    assert "distinguish competing hypotheses" in baseline


def test_no_paragraph_is_invented_for_a_variant_we_cannot_render() -> None:
    with pytest.raises(NotImplementedError, match="future_mode"):
        actor_world_model_paragraph("future_mode")


def test_the_baseline_workspace_seeds_no_model_and_no_probe_to_run_on_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole arm, at the one place it could silently not happen.

    Seeding and probing are one switch upstream (`wm_variant != "none"`): a
    workspace with no world_model.py whose edits still triggered the feedback
    probe would report a missing model as a defect every turn.
    """
    monkeypatch.setenv(MODE_ENV, "no_world_model")

    agent = _agent(tmp_path)

    workspace = agent.workspace_dir
    for absent in ("world_model.py", "verify.py", "plan.py", "predict.py"):
        assert not (workspace / absent).exists(), absent
    # wmlib.py is ours, not agent-authored: wm_none.j2 tells the actor to use it
    # for observed-frame inspection, so the baseline still gets it.
    assert (workspace / "wmlib.py").is_file()
    assert agent._tool_executor.wm_feedback_enabled is False
    assert agent._trigger_builder is None


def test_the_trigger_arm_is_unchanged_by_the_mode_plumbing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MODE_ENV, "trigger")

    agent = _agent(tmp_path)

    assert (agent.workspace_dir / "world_model.py").is_file()
    assert agent._tool_executor.wm_feedback_enabled is True
    assert agent._trigger_builder is not None


def test_an_explicit_trigger_builder_argument_still_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests construct agents without a builder; the mode default must not
    take that switch away from them."""
    monkeypatch.setenv(MODE_ENV, "trigger")

    assert _agent(tmp_path, trigger_builder=False)._trigger_builder is None


def _agent(tmp_path: Path, **kwargs: object) -> TychoNooaAgent:
    return TychoNooaAgent(
        llm=FakeLLMClient(scripted_responses=[]),
        game_id="test-game",
        workspace_root=tmp_path,
        available_actions=["UP", "DOWN"],
        **kwargs,
    )
