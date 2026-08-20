"""The direct-reasoning baseline: what a run carries when no world model exists."""

from __future__ import annotations

from pathlib import Path

import pytest
from nooa.unifiedllm import FakeLLMClient

from tycho_nooa import TychoNooaAgent
from tycho_nooa.modes import DEFAULT_MODE, MODE_ENV, mode_from_env, served_mode
from tycho_nooa.planner import actor_world_model_paragraph


def test_an_unset_environment_runs_the_topology_every_run_so_far_used() -> None:
    assert mode_from_env({}).name == DEFAULT_MODE == "trigger"
    assert mode_from_env({MODE_ENV: ""}).name == "trigger"


def test_the_baseline_is_served_and_fires_no_builder() -> None:
    spec = mode_from_env({MODE_ENV: "no_world_model"})

    assert spec.wm_variant == "none"
    assert spec.needs_builder is False


@pytest.mark.parametrize(
    ("mode", "missing"),
    [("single", "actor_writes_wm"), ("orchestrator", "include_builder")],
)
def test_a_real_mode_we_do_not_wire_fails_loudly_naming_what_is_missing(
    mode: str, missing: str
) -> None:
    """The dangerous failure is silence: an unwired mode running as `trigger`
    would produce a comparison arm that is secretly the control."""
    with pytest.raises(NotImplementedError, match=missing):
        served_mode(mode)


def test_an_unknown_mode_is_still_upstreams_error() -> None:
    with pytest.raises(ValueError, match="not a recognized mode"):
        served_mode("no_builder")


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
    with pytest.raises(NotImplementedError, match="orch"):
        actor_world_model_paragraph("orch")


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
