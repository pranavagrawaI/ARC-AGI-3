from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from construction_messages import construction_messages
from freeze import load_frozen_module, write_frozen
from gameplay_solver import GameplaySolver
from observation_abi import validate_constructor_surface
from prefix_adapter import PrefixError, project_actor_prefix

FIXTURES = HERE / "fixtures" / "actor_ipc"
STATES = FIXTURES / "states.jsonl"
ACTIONS = FIXTURES / "executed_actions.jsonl"
EVIDENCE = HERE / "evidence" / "case.json"
FROZEN = HERE / "FROZEN.json"
CANDIDATE_PATH = HERE / "candidates" / "fixture_candidate.py"

BEFORE = [[12, 4, 4]]
AFTER = [[4, 12, 4]]
AVAILABLE = ["UP", "DOWN", "LEFT", "RIGHT"]


class PrefixAdapterTests(unittest.TestCase):
    def test_executed_prefix_projects_decoded_grids_and_action(self) -> None:
        case = project_actor_prefix(STATES, ACTIONS)
        inspection = case.inspection_view()
        self.assertEqual(inspection["visible_transitions"][0]["action"], "RIGHT")
        self.assertEqual(inspection["visible_transitions"][0]["before"]["grid"], BEFORE)
        self.assertEqual(inspection["visible_transitions"][0]["after"]["grid"], AFTER)
        self.assertEqual(inspection["initial_observation"]["grid"], BEFORE)
        self.assertEqual(inspection["current_observation"]["grid"], AFTER)
        self.assertEqual(inspection["frontier"], 1)

    def test_length_mismatch_is_rejected(self) -> None:
        with self.assertRaises(PrefixError):
            project_actor_prefix(STATES, HERE / "fixtures" / "actor_ipc" / "empty_actions.jsonl")

    def test_unprojected_grid_rows_are_rejected_on_constructor_surface(self) -> None:
        with self.assertRaises(ValueError):
            validate_constructor_surface(
                {"visible_transitions": [{"step": 1, "action": "RIGHT", "before": {"grid_rows": ["c44"]}}]}
            )

    def test_constructor_views_pass_abi_and_omit_forbidden_world_models(self) -> None:
        case = project_actor_prefix(STATES, ACTIONS)
        compact = case.compact_view()
        inspection = case.inspection_view()
        validate_constructor_surface(compact)
        validate_constructor_surface(inspection)
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE.write_text(json.dumps(inspection, separators=(",", ":")))
        dumped = json.dumps(construction_messages(case))
        self.assertNotIn("collision", dumped)
        self.assertNotIn("suffix", dumped)
        self.assertNotIn("knowledge/", dumped)
        self.assertNotIn("helpers/", dumped)
        self.assertNotIn("grid_rows", dumped)

    def test_frozen_candidate_loads_in_gameplay_solver(self) -> None:
        candidate = CANDIDATE_PATH.read_text()
        record = write_frozen(FROZEN, candidate)
        independent = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        self.assertEqual(record["source_digest"], independent)
        on_disk = json.loads(FROZEN.read_text())
        self.assertEqual(on_disk["source_digest"], independent)
        self.assertEqual(on_disk["candidate_source"], candidate)
        module = load_frozen_module(FROZEN)
        solver = GameplaySolver(module)
        observation = SimpleNamespace(grid=BEFORE, available_actions=AVAILABLE)
        solver.start_game(observation)
        decision = solver.decide(observation, AVAILABLE)
        self.assertIsInstance(decision.action, str)
        self.assertIn(decision.action, AVAILABLE)
        self.assertEqual(decision.action, "RIGHT")

    def test_gameplay_solver_does_not_import_solver_agent(self) -> None:
        source = (HERE / "gameplay_solver.py").read_text()
        self.assertNotIn("solver_agent", source)
        self.assertNotIn("solver_agent", sys.modules)


if __name__ == "__main__":
    unittest.main()
