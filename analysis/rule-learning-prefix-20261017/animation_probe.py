"""Separate post-run observation ablation. The frozen prefix experiment remains unchanged."""
import hashlib
import importlib.util
import json
import signal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("prefix_run", HERE / "run.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

card = json.loads((HERE / "ANIMATION-PROBE-CARD.json").read_text())
for path, sha in card["source_sha256"].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == sha
signal.signal(signal.SIGALRM, M.alarm)
signal.alarm(card["cap"]["wall_seconds"])
track = next(t for t in json.loads((HERE/"inputs.json").read_text()) if t["id"] == "tn36-b1-r1")
prefix_rows = track["actions"][:40]
boards = [track["entry_board"]] + [r["board"] for r in prefix_rows]
actions = [r["display"] for r in prefix_rows]
row = track["actions"][40]
receipt = json.loads((ROOT/"analysis/failure-audit-20261013/tn36/engine-replay.json").read_text())
command = next(c for c in receipt["runs"]["b1-r1"]["commands"] if c["local_action"] == 41)
targets = [("settled", row["board"])] + [(f"animation-{i}", command["animation_frames"][i]) for i in card["targets"]]
assert command["animation_frames"][-1] == row["board"]
result = {}
for name, target in targets:
    r = M.evaluate(boards, actions, row["display"], target)
    result[name] = r
assert len({r["frozen_prediction_sha256"] for r in result.values()}) == 1
(HERE/"animation-probe.json").write_text(json.dumps(result, indent=2)+"\n")
for name,r in result.items():
    print(json.dumps(dict(target=name, score=r["score"], events=r["event_descriptions"])))
