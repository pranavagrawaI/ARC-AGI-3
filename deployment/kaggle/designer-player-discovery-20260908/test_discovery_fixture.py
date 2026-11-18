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
        assert trace["first_decision"][2]["value"] is None
        assert trace["host_transition"]["action"] == "UP"
        second = {item["name"]: item["value"] for item in trace["second_decision"]}
        assert second["last_transition"]["changed_cells"] == 2
        assert second["changed_region"]["region_bbox"] is not None
        assert "does not establish" in trace["negative_boundary"]
        demo = render_discovery_demonstration(trace)
        assert "submit then host transition" not in demo
        assert "Host transition" in demo
        assert "negative_boundary" in demo
        evidence = measured_evidence(agent)
        assert evidence["previous_transition_status"] == "measured"
        assert evidence["current"]["step"] == 1
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
    assert (root / "chassis/discovery-demo.md").read_text().startswith("<discovery_demonstration>")
    manifest = json.loads((root / "CHASSIS.json").read_text())
    assert "chassis/evidence_renderer.py" in manifest["files"]
    assert "chassis/discovery-demo.md" in manifest["files"]
