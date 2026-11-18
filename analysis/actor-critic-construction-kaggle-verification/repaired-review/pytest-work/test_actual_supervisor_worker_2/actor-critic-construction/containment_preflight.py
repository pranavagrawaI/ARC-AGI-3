"""Actual sealed RPC and inspection gates, before server warmup or model calls."""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
from integration_boundary import CandidateRPC, SealedBoundary


def qualify(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema": "pre-model-containment-v1", "status": "failed", "cases": {}}
    try:
        with tempfile.TemporaryDirectory(prefix="pre-model-canary-") as d:
            canary = Path(d) / "sealed-future"
            canary.write_text("synthetic-future-not-an-experiment-trace")
            code = f'''
import os
outside = {str(canary)!r}
def probe():
    try:
        open(outside).read()
        denied = False
    except OSError:
        denied = True
    return {{"outside_denied": denied, "stdin_eof": os.read(0, 32) == b"", "credential_absent": "PREFLIGHT_SYNTHETIC_KEY" not in os.environ}}
def initialize(observation): return probe()
def predict(state, action): return [{{"state": state, "commitments": {{}}}}]
def assimilate(state, observation, verdict): return state
'''
            old = os.environ.get("PREFLIGHT_SYNTHETIC_KEY")
            os.environ["PREFLIGHT_SYNTHETIC_KEY"] = "synthetic-only"
            try:
                rpc = CandidateRPC(SealedBoundary(), code)
                value = rpc.initialize({"grid": [[0]]})
                assert value == {"outside_denied": True, "stdin_eof": True, "credential_absent": True}, value
                assert rpc.assimilate(rpc.predict(value, "UP")[0]["state"], {"grid": [[1]]}, {"status": "unscored"}) == value
                record["cases"]["candidate"] = "passed"
                inspected = SealedBoundary().inspect(code + '\nimport json\nprint(json.dumps(probe()))\nprint(open("/evidence/case.json").read())', {"prefix_canary": "visible-only"})
                assert inspected["status"] == "completed", inspected
                lines = inspected["stdout"].splitlines()
                assert json.loads(lines[0]) == value
                assert json.loads(lines[1]) == {"prefix_canary": "visible-only"}
                record["cases"]["inspection"] = "passed"
            finally:
                if old is None:
                    os.environ.pop("PREFLIGHT_SYNTHETIC_KEY", None)
                else:
                    os.environ["PREFLIGHT_SYNTHETIC_KEY"] = old
        record["status"] = "passed"
        return record
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    import sys
    qualify(Path(sys.argv[1]))
