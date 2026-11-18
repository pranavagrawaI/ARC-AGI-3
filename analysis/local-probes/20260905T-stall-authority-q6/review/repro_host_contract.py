"""Compare real chassis submission gates with the observation file from the join.

Run with /home/pranav/projects/arc-agi-3-reference/nooa/.venv/bin/python.
No model or environment calls. Expected output describes current defects.
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
JOIN = ROOT / "deployment/kaggle/s0-nooa-join-candidate"
sys.path.insert(0, str(ROOT / "solvers/tycho_nooa/src"))
sys.path.insert(0, str(JOIN / "chassis"))
# Load the real DTO modules without importing the unrelated Tycho agent runtime.
dto_package = ModuleType("tycho_nooa")
dto_package.__path__ = [str(ROOT / "solvers/tycho_nooa/src/tycho_nooa")]
sys.modules["tycho_nooa"] = dto_package


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


worker = load("review_worker", JOIN / "nooa_worker.py")
chassis = load("review_chassis", JOIN / "chassis/solver_agent.py")
tmp = Path(tempfile.mkdtemp(prefix="q6-host-contract-"))
session = worker.HostSession(tmp)
obs = SimpleNamespace(grid=[[0, 1], [0, 0]], state="NOT_FINISHED",
                      levels_completed=0, available_actions=["UP", "DOWN"])
session.observe(SimpleNamespace(sequence=0, before=obs, after=obs,
                                decision=SimpleNamespace(action="UP")))
record = json.loads(session.observations_path.read_text())
assert record["known_from_here"] == {} and record["last_retrodiction"] is None


class Actor:
    submit_actions = chassis.ArcSolverBase.submit_actions
    _effort_ladder = []

    def __init__(self, evidence):
        self.evidence = evidence
        self._actions_path = tmp / "gate-actions.jsonl"

    def _latest_state(self):
        return {"turn": 1, "levels_completed": 0, "state": "NOT_FINISHED",
                "available_actions": ["UP", "DOWN"]}

    def _last_submitted_turn(self):
        return None

    def observations(self):
        return self.evidence

    def _log_rejection(self, *args):
        pass


def submit(evidence, actions):
    try:
        return Actor(evidence).submit_actions(
            actions, "predict: move", predicted_path=[{"expect": "move"}] * len(actions))
    except chassis._ReturnResultSignal:
        return "ACCEPTED"


assert submit({"known_from_here": {"UP": "no_impact"}}, ["UP"]).startswith("REJECTED")
assert submit(record, ["UP"]) == "ACCEPTED"
assert submit({"last_retrodiction": {"ok": False, "matched": 0, "checked": 2}},
              ["UP", "DOWN"]).startswith("REJECTED")
assert submit(record, ["UP", "DOWN"]) == "ACCEPTED"
print("Known inert UP: populated history REJECTED; join history ACCEPTED")
print("Failed prediction: populated score REJECTED; join score ACCEPTED")

advanced = SimpleNamespace(grid=[[2, 2], [2, 2]], state="NOT_FINISHED",
                           levels_completed=1, available_actions=["UP", "DOWN"])
session.observe(SimpleNamespace(sequence=1, before=obs, after=advanced,
                                decision=SimpleNamespace(action="DOWN")))
assert not (session.ipc / "level_0_terminal.json").exists()
print("Completed level: level_0_terminal.json absent")
win = SimpleNamespace(grid=[[3, 3]], state="WIN", levels_completed=2)
assert worker._outcome(SimpleNamespace(before=advanced, after=win)) == "level_up"
print("WIN with level increment: join labels level_up; native harness labels terminal")
print(f"Artifacts: {tmp}")
