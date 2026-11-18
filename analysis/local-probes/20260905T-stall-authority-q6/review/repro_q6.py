"""Reproduce q6 review findings using the existing offline fixture; no model calls."""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    q6 = Path(__file__).resolve().parents[1]
    fixture = load("q6_fixture", q6 / "test_g5_offline.py")
    replay = load("replay", q6.parents[1] / "discrimination-replay/replay.py")
    chassis = fixture.load_chassis()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        def opened(name, rationale="predict: navigate toward spiral"):
            stub = fixture.bind(fixture.Stub(root / name, chassis), chassis)
            fixture.write_batches(stub, [rationale] * 7, ["UP"])
            stub._evaluate_stall(fixture.state(40, 7))
            return stub

        os.environ["ARC_STALL_REVIEW"] = "0"
        control = opened("control")
        print("control flag=0 interlock open:", control._stall_id != control._resolved_stall_id)
        print("probe without predictions:", control.resolve_stall("probe", "H1", "H1"))

        short = opened("short", "go up to x")
        run = replay.Run(
            root,
            [replay.Turn(i * 6, i, ["UP"], "go up to x") for i in range(7)],
            [replay.State(i * 6, i, 0, 0, ["UP", "DOWN"], "NOT_FINISHED") for i in range(7)],
            {}, {}, [], [], [],
        )
        print("short-token detections live/replay:", short._stall_id, len(replay.trigger_stall_repetition(run)))

        truncated = opened("truncated")
        prefix = "x" * 500
        print("switch differing after character 500:", truncated.resolve_stall("switch", prefix + "A", prefix + "B"))
        row = truncated.events()[-1]
        print("recorded different bases became equal:", row["previous_basis"] == row["basis"])

        rejected = opened("rejected")
        responses = [rejected.resolve_stall("continue", "H1", "H1") for _ in range(8)]
        print("rejected resolutions:", sum(r.startswith("REJECTED") for r in responses))
        print("rejection logger calls:", len(rejected.rejections))
        print("ABORT exists:", (rejected.ipc_dir / "ABORT").exists())


if __name__ == "__main__":
    main()
