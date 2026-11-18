import json
import types
import unittest

from observation_abi import validate_constructor_surface, validate_observation
from real_trace import translated_prefix_case
from replay import replay_visible

LS20_BLOCK = [45, 34]
LS20_BAR = 13


HARD_CODED = '''
def initialize(observation):
    return {"block": [45, 34], "bar_left": 13, "step": 0}

def predict(state, action):
    return [{"state": state, "commitments": {"block": state["block"], "bar_left": state["bar_left"]}, "declared_dependencies": ["hardcoded"]}]

def assimilate(predicted_state, observation, verdict):
    return dict(predicted_state)
'''

GROUNDED = '''
def initialize(observation):
    grid = observation["grid"]
    cells = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 12]
    bar = next(c for c, v in enumerate(grid[61]) if v == 11)
    return {"block": [min(r for r, _c in cells), min(c for _r, c in cells)], "bar_left": bar}

def predict(state, action):
    dr, dc = {"UP": (-5, 0), "DOWN": (5, 0), "LEFT": (0, -5), "RIGHT": (0, 5)}.get(action, (0, 0))
    nxt = {"block": [state["block"][0] + dr, state["block"][1] + dc], "bar_left": state["bar_left"] + 1}
    return [{"state": nxt, "commitments": dict(nxt), "declared_dependencies": ["block:state", "bar_left:state", "action:driver"]}]

def assimilate(predicted_state, observation, verdict):
    return dict(predicted_state)
'''


def load_module(source: str) -> types.ModuleType:
    module = types.ModuleType("candidate")
    exec(source, module.__dict__)
    return module


class ObservationAbiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.origin = translated_prefix_case(LS20_BLOCK, LS20_BAR, [0, 0], 0)
        self.shifted = translated_prefix_case(LS20_BLOCK, LS20_BAR, [3, -2], 4)

    def test_inspection_and_compact_share_canonical_grid(self) -> None:
        inspection = self.shifted.inspection_view()
        compact = self.shifted.compact_view()
        initial = validate_observation(inspection["initial_observation"])
        transition = validate_observation(inspection["visible_transitions"][0]["before"])
        self.assertEqual(initial, transition)
        self.assertIsInstance(initial["grid"][0][0], int)
        self.assertNotIn("grid_rows", json.dumps(inspection))
        self.assertNotIn("grid_rows", json.dumps(compact))
        self.assertNotIn("suffix", inspection)
        self.assertNotIn("sealed_suffix", inspection)
        self.assertNotIn("suffix", compact)
        self.assertEqual(inspection["frontier"], 1)
        self.assertTrue(all(row["step"] <= inspection["frontier"] for row in inspection["visible_transitions"]))

    def test_hex_string_view_is_rejected(self) -> None:
        live_shaped = {
            "initial_observation": {"grid_rows": ["4" * 64]},
            "visible_transitions": [
                {"step": 1, "action": "RIGHT", "before": {"grid_rows": ["4" * 64]}, "after": {"grid_rows": ["4" * 64]}}
            ],
        }
        with self.assertRaises(ValueError):
            validate_constructor_surface(live_shaped)

    def test_evaluator_suffix_does_not_enter_constructor_views(self) -> None:
        payload = self.shifted.evaluator_payload(include_suffix=True)
        self.assertEqual(len(payload["suffix"]), 1)
        inspection = self.shifted.inspection_view()
        compact = self.shifted.compact_view()
        self.assertNotIn("suffix", inspection)
        self.assertNotIn("suffix", compact)
        dumped = json.dumps(inspection) + json.dumps(compact)
        self.assertNotIn('"step": 6', dumped)

    def test_hard_coded_ls20_initialize_fails_translation(self) -> None:
        module = load_module(HARD_CODED)
        fixture = self.shifted.evaluator_payload(include_suffix=False)
        result = replay_visible(module, fixture)
        self.assertEqual(result["status"], "contradicted")
        origin_state = module.initialize(self.origin.inspection_view()["initial_observation"])
        self.assertEqual(origin_state["block"], LS20_BLOCK)
        self.assertEqual(origin_state["bar_left"], LS20_BAR)

    def test_grounded_initialize_passes_origin_and_translation(self) -> None:
        module = load_module(GROUNDED)
        for case in (self.origin, self.shifted):
            observation = case.inspection_view()["initial_observation"]
            state = module.initialize(observation)
            measured = case.constructor_view()["visible_transition_summaries"][0]["before"]
            self.assertEqual(state["block"], measured["block"])
            self.assertEqual(state["bar_left"], measured["bar_left"])
            self.assertNotEqual(state["block"], LS20_BLOCK if case is self.shifted else [0, 0])
            result = replay_visible(module, case.evaluator_payload(include_suffix=False))
            self.assertEqual(result["status"], "passed")

    def test_initialize_receives_the_inspection_observation(self) -> None:
        seen = {}

        class Probe:
            def initialize(self, observation):
                seen["observation"] = observation
                grid = observation["grid"]
                cells = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 12]
                bar = next(c for c, v in enumerate(grid[61]) if v == 11)
                return {"block": [min(r for r, _c in cells), min(c for _r, c in cells)], "bar_left": bar}

            def predict(self, state, action):
                nxt = {"block": [state["block"][0], state["block"][1] + 5], "bar_left": state["bar_left"] + 1}
                return [{"state": nxt, "commitments": dict(nxt), "declared_dependencies": []}]

            def assimilate(self, predicted_state, observation, verdict):
                seen["assimilate_observation"] = observation
                return dict(predicted_state)

        case = self.shifted
        replay_visible(Probe(), case.evaluator_payload(include_suffix=False))
        self.assertEqual(seen["observation"], case.inspection_view()["initial_observation"])
        self.assertEqual(seen["assimilate_observation"], case.inspection_view()["visible_transitions"][0]["after"])


if __name__ == "__main__":
    unittest.main()
