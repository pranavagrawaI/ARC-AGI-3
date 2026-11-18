import json
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
WORKER = HERE / "worker.py"
CANDIDATE = HERE / "candidates" / "exploratory.py"
HOST = "ARC_HOST\t"
AGENT = "ARC_AGENT\t"


def envelope():
    return {
        "decision_frame": [[4, 4, 4], [4, 12, 4], [4, 4, 4]],
        "terminal_frame": None,
        "event_frames": [],
        "provenance": {
            "raw_frame_count": 1,
            "decision_frame_index": 0,
            "terminal_frame_index": None,
            "event_frame_indices": [],
            "terminal_reason": None,
        },
        "available_actions": ["RESET", "UP", "DOWN", "LEFT", "RIGHT"],
        "levels_completed": 0,
        "state": "NOT_FINISHED",
        "full_reset": False,
    }


def host(kind, **fields):
    return HOST + json.dumps({"protocol_version": 3, "type": kind, **fields}) + "\n"


class WorkerProtocolTests(unittest.TestCase):
    def test_worker_emits_a_decision_on_the_arc_host_protocol(self) -> None:
        script = (
            host(
                "start_game",
                game_id="ls20",
                seed=0,
                workspace_root=str(HERE),
                initial_observation=envelope(),
            )
            + host("turn", sequence=0, level=0, turn_in_level=0, observation=envelope(), previous_decision=None)
            + host(
                "action_result",
                sequence=0,
                level=0,
                turn_in_level=0,
                decision={"action": "UP", "row": None, "col": None},
                before=envelope(),
                after=envelope(),
            )
            + host("finish", stop_reason="action_budget", actions=1, levels_completed=0, final_observation=envelope())
            + host("close")
        )
        result = subprocess.run(
            [sys.executable, str(WORKER), "--candidate", str(CANDIDATE)],
            input=script,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [line[len(AGENT) :] for line in result.stdout.splitlines() if line.startswith(AGENT)]
        types = [json.loads(line)["type"] for line in lines]
        self.assertEqual(types, ["ready", "decision", "observed", "finished"])
        decision = json.loads(lines[1])["decision"]
        self.assertEqual(decision["action"], "UP")
        self.assertEqual(json.loads(lines[1])["protocol_version"], 3)


if __name__ == "__main__":
    unittest.main()
