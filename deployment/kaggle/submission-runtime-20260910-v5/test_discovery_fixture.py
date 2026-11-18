import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "chassis"))


@pytest.mark.asyncio
async def test_two_decision_fixture_uses_real_tools_and_protocol_boundary(tmp_path):
    from nooa.unifiedllm import FakeLLMClient
    from solver_agent import MdArcSolverAgent
    from evidence_renderer import (
        measured_evidence,
        render_discovery_demonstration,
        run_two_decision_fixture,
    )

    agent = MdArcSolverAgent(llm=FakeLLMClient(), run_dir=tmp_path, game_id="opaque", alias="game")
    try:
        trace = run_two_decision_fixture(agent)
        assert trace["protocol"] == "submit then host transition then measure"
        assert trace["api_signatures"]["set_experiment"] == (
            "(question: 'str', hypotheses: 'list[str]', action: 'str', "
            "outcomes: 'dict') -> 'dict'"
        )
        assert trace["code_cells"][0]["source"].startswith("grid = self.model_grid()")
        assert "self.set_experiment(question, hypotheses, action, outcomes)" in trace["code_cells"][0]["source"]
        assert "self.submit_actions([action]" in trace["code_cells"][0]["source"]
        assert trace["code_cells"][0]["captured"]["summary"]["status"] == "measured"
        assert trace["code_cells"][0]["terminal"]["kind"] == "WAIT"
        assert trace["host_transition"]["action"] == "UP"
        assert trace["code_cells"][1]["source"].startswith("transition = self.last_transition(True)")
        assert trace["code_cells"][1]["captured"]["transition"]["changed_cells"] == 2
        assert trace["code_cells"][1]["captured"]["region"]["region_bbox"] is not None
        assert trace["code_cells"][1]["terminal"]["kind"] == "WAIT"
        assert trace["second_action"] == "DOWN"
        assert "does not establish" in trace["negative_boundary"]
        demo = render_discovery_demonstration(trace)
        assert demo.count("```python") == 2
        assert demo.count("```output") == 2
        assert "self.set_experiment(question, hypotheses, action, outcomes)" in demo
        assert "Host transition" in demo
        assert "negative_boundary" in demo
        evidence = measured_evidence(agent)
        assert evidence["previous_transition_status"] == "measured"
        assert evidence["current"]["step"] == 1
        assert "grid_rows" not in evidence["current"]
        assert "before_grid_rows" not in evidence["previous_transition"]
        assert "after_grid_rows" not in evidence["previous_transition"]
        assert (agent.ipc_dir / "evidence_capsules.jsonl").exists()
        # Rendering unchanged evidence does not duplicate the external record.
        before = (agent.ipc_dir / "evidence_capsules.jsonl").read_text().splitlines()
        measured_evidence(agent)
        after = (agent.ipc_dir / "evidence_capsules.jsonl").read_text().splitlines()
        assert after == before
    finally:
        await agent.queue_manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case, expected_update, expected_action",
    [
        ("translated_variant", "translation_observed", "DOWN"),
        ("changed_action_mapping", "unexpected_region", "LEFT"),
        ("reshaped_target", "shape_changed", "DOWN"),
        ("salient_noop", "no_effect", "LEFT"),
        ("alternative_region", "alternative_region", "RIGHT"),
    ],
)
async def test_transfer_cases_change_second_decision_from_returned_evidence(
    tmp_path, case, expected_update, expected_action
):
    from nooa.unifiedllm import FakeLLMClient
    from solver_agent import MdArcSolverAgent
    from evidence_renderer import run_two_decision_fixture

    agent = MdArcSolverAgent(llm=FakeLLMClient(), run_dir=tmp_path, game_id="opaque", alias="game")
    try:
        trace = run_two_decision_fixture(agent, case=case)
        captured = trace["code_cells"][1]["captured"]
        assert captured["update"] == expected_update
        assert trace["second_action"] == expected_action
        assert captured["transition"] == agent.last_transition(True)
        assert not (agent.helpers_dir / "world_model.py").exists()
        assert not (agent.ipc_dir / "procedures.jsonl").exists()
    finally:
        await agent.queue_manager.shutdown()


@pytest.mark.asyncio
async def test_first_observation_is_not_a_measured_noop(tmp_path):
    from nooa.unifiedllm import FakeLLMClient
    from solver_agent import MdArcSolverAgent
    from evidence_renderer import render_measured_evidence

    agent = MdArcSolverAgent(llm=FakeLLMClient(), run_dir=tmp_path, game_id="opaque", alias="game")
    try:
        state = {
            "turn": 0, "step": 0, "state": "NOT_FINISHED",
            "available_actions": ["UP"], "grid_rows": ["0"], "grid": [[0]],
        }
        agent._states_path.write_text(json.dumps(state) + "\n")
        rendered = render_measured_evidence(agent)
        assert "none_executed" in rendered
        assert "measured no-op" not in rendered
    finally:
        await agent.queue_manager.shutdown()


def test_candidate_freeze_and_factor_separation():
    import json
    import subprocess

    root = HERE
    subprocess.run([sys.executable, str(root / "freeze_candidate.py"), "--check"], check=True)
    headless = (root / "headless.py").read_text()
    assert "ARC_DISCOVERY_ARM" in headless
    assert "with_evidence = discovery_arm in ('b', 'd')" in headless
    assert "with_demo = discovery_arm in ('c', 'd')" in headless
    assert "max_context_tokens=20_000" in headless
    assert "prefix=True" in headless
    assert "enable_summarizer=not procedural_diagnostic" in headless
    assert "Do not create or edit helpers" in headless
    assert "if not procedural_diagnostic:\n        agent.context['decision_workflow']" in headless
    assert (root / "chassis/discovery-demo.md").read_text().startswith("<discovery_demonstration>")
    manifest = json.loads((root / "CHASSIS.json").read_text())
    assert "chassis/evidence_renderer.py" in manifest["files"]
    assert "chassis/discovery-demo.md" in manifest["files"]


def test_procedural_diagnostic_flag_crosses_worker_boundary():
    import importlib.util

    spec = importlib.util.spec_from_file_location("diagnostic_shared_solver", HERE / "shared_local_solver.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.worker_environment({"ARC_PROCEDURAL_DIAGNOSTIC": "1", "SECRET": "no"}) == {
        "ARC_PROCEDURAL_DIAGNOSTIC": "1"
    }
