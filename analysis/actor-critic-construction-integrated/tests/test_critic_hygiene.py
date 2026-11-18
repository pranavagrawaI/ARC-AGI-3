from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CRITIC = ROOT / "deployment/kaggle/submission-runtime-20260912-v11/chassis/epistemic_critic.py"
SOLVER = ROOT / "deployment/kaggle/submission-runtime-20260912-v11/chassis/solver_agent.py"


def load_critic():
    spec = importlib.util.spec_from_file_location("v11_epistemic_critic", CRITIC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_critic_packet_has_no_constructor_diagnosis() -> None:
    c = load_critic()
    packet = c.critic_packet(
        trigger="eight_actor_calls",
        evidence=[{"claim_ref": "actor-message-1-content", "kind": "assistant_content", "text": "move right"}],
        commitment={"turn": 1, "step": 1, "actions": ["UP"], "rationale": "probe"},
        transitions=[{"step": 1, "action": "UP", "changed_cells": 1, "shape_changed": False,
                      "changed_bbox": [0, 0, 0, 0], "state": "NOT_FINISHED", "levels_completed": 0}],
        budget=c.budget_view(wall_remaining=100.0, actions_executed=1, wall_elapsed=10.0),
        ledger={"pending": None, "reminders": 0, "retired_claims": [], "closed": []},
        state={"turn": 1, "step": 1, "level": 0, "state": "NOT_FINISHED",
               "levels_completed": 0, "available_actions": ["UP"]},
    )
    rendered = json.dumps(packet)
    assert "constructor" not in rendered
    assert "FROZEN.json" not in rendered
    assert "collision" not in rendered
    assert "candidate_source" not in rendered
    assert set(packet) >= {"commitment", "actor_claims", "measured_transitions", "budget", "objections"}


def test_incumbent_critic_invocation_block_is_unedited() -> None:
    source = SOLVER.read_text()
    assert "critic_request_messages = _critic_messages(" in source
    marker = source.index("max_tokens=192")
    window = source[max(0, marker - 400) : marker + 80]
    assert "tools=None" in window
    assert "submit_candidate" not in source[source.index("def _guard") : source.index("class MemArcSolverAgent")]
    assert "query_frozen_predict" not in source
