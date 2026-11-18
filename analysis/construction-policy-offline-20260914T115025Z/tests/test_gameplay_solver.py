import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

from action_selector import select_action
from gameplay_solver import GameplaySolver
from local_env import LocalGridWorld
from observation_abi import observation_from_host
from play_local import run_local


def load(name: str):
    path = HERE / "candidates" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GameplaySolverTests(unittest.TestCase):
    def test_host_tuple_grid_is_canonical(self) -> None:
        observation = observation_from_host(type("Obs", (), {"grid": ((4, 12), (4, 4))})())
        self.assertEqual(observation["grid"], [[4, 12], [4, 4]])
        envelope = {
            "decision_frame": [[4, 12], [4, 4]],
            "available_actions": ["UP"],
        }
        self.assertEqual(observation_from_host(envelope)["grid"], [[4, 12], [4, 4]])
        with self.assertRaises(ValueError):
            observation_from_host({"grid_rows": ["4c"]})

    def test_selector_skips_predicted_wall_and_env_receives_that_action(self) -> None:
        module = load("occupancy_motion.py")
        env = LocalGridWorld(block=[4, 0])
        solver = GameplaySolver(module)
        observation = env.reset()
        solver.start_game(observation)
        decision = solver.decide(observation, observation.available_actions)
        self.assertEqual(decision.action, "RIGHT")
        following = env.step(decision.action)
        solver.observe(following)
        self.assertEqual(env.steps, ["RIGHT"])
        self.assertEqual(env.block, [4, 1])

    def test_exploratory_fallback_still_submits_a_real_action(self) -> None:
        module = load("exploratory.py")
        env = LocalGridWorld()
        solver = GameplaySolver(module)
        observation = env.reset()
        solver.start_game(observation)
        decision = solver.decide(observation, observation.available_actions)
        self.assertEqual(decision.action, "UP")
        env.step(decision.action)
        self.assertEqual(env.steps, ["UP"])

    def test_local_episode_executes_multiple_environment_steps(self) -> None:
        result = run_local(load("occupancy_motion.py"), max_actions=6)
        self.assertGreaterEqual(result["action_count"], 6)
        self.assertEqual(result["actions"][0], "RIGHT")
        self.assertNotEqual(result["final_block"], [4, 0])

    def test_selector_does_not_name_collision(self) -> None:
        module = load("occupancy_motion.py")
        env = LocalGridWorld(block=[4, 0])
        state = module.initialize(observation_from_host(env.reset()))
        decision = select_action(module, state, ("LEFT", "RIGHT"))
        self.assertEqual(decision.action, "RIGHT")


if __name__ == "__main__":
    unittest.main()
