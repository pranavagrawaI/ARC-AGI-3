"""Exercise the unchanged sealed runner inside a local mapped user namespace.

Run: unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python analysis/actor-critic-construction-verification/check_local_seal.py
No real model requests are made. This is a trusted fixture containment check.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/construction-policy-live-20260914T093351Z"))
from integration_boundary import SealedBoundary  # noqa: E402 — donor module requires its runtime import path

SOURCE = '''
def initialize(observation):
    import os
    return {"grid": observation["grid"], "uid": os.getuid(), "gid": os.getgid()}
def predict(state, action):
    return [{"state": state, "commitments": {}, "declared_dependencies": []}]
def assimilate(predicted_state, observation, verdict):
    return predicted_state
'''

if __name__ == "__main__":
    result = SealedBoundary().invoke(SOURCE, "initialize", [{"grid": [[4, 12]]}])
    assert result == {"grid": [[4, 12]], "uid": 65534, "gid": 65534}, result
    print(json.dumps({"status": "passed", "scope": "local mapped namespace, unchanged sealed candidate runner", "result": result}))
