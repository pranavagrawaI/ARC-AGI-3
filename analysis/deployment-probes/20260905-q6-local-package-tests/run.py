"""CPU-only tests of the frozen notebook's extracted deployment payload."""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PAYLOAD = OUT / "extracted"
NOTEBOOK = ROOT / "deployment/kaggle/q6-control-preflight/preflight.ipynb"
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "matplotlib"))


def extract():
    PAYLOAD.mkdir(exist_ok=True)
    notebook = json.loads(NOTEBOOK.read_text())
    assets = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Attribute)
                or node.func.attr != "write_bytes"
            ):
                continue
            try:
                destination = ast.literal_eval(node.func.value.args[0])
                encoded = ast.literal_eval(node.args[0].args[0])
                if isinstance(encoded, str):
                    assets[destination] = base64.b64decode(encoded)
            except (AttributeError, ValueError, IndexError):
                pass
    with zipfile.ZipFile(io.BytesIO(assets["/tmp/nooa-join.zip"])) as archive:
        for name in archive.namelist():
            path = Path(name)
            assert not path.is_absolute() and ".." not in path.parts
        archive.extractall(PAYLOAD)
    (PAYLOAD / "runtime").mkdir(exist_ok=True)
    (PAYLOAD / "runtime/__init__.py").write_text("")
    (PAYLOAD / "runtime/local_solver.py").write_bytes(
        assets["/tmp/shared_local_solver.py"]
    )
    (PAYLOAD / "my_agent.py").write_bytes(assets["/tmp/my_agent.py"])
    manifest = json.loads((PAYLOAD / "CHASSIS.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((PAYLOAD / name).read_bytes()).hexdigest() == expected
    return {
        "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
        "contract_sha256": manifest["contract_sha256"],
        "verified_files": len(manifest["files"]),
    }


def main():
    report = extract()
    sys.path.insert(0, str(ROOT / "deployment/kaggle/tests"))
    sys.path.insert(0, str(PAYLOAD))
    import pytest
    import test_my_agent
    import test_nooa_join as join
    import test_q6_host_contract as host_tests

    test_my_agent.AGENT_PATH = PAYLOAD / "my_agent.py"
    join.CANDIDATE = PAYLOAD
    host_tests.CANDIDATE = PAYLOAD
    # The fixture prepends deployment/kaggle; importing the extracted runtime first
    # makes the worker factory use the exact notebook adapter dependencies.
    import runtime.local_solver

    assert (
        Path(runtime.local_solver.__file__).resolve()
        == PAYLOAD / "runtime/local_solver.py"
    )
    result = pytest.main(
        [
            str(ROOT / "deployment/kaggle/tests/test_q6_host_contract.py"),
            str(OUT / "test_payload.py"),
            "-q",
            "--basetemp",
            str(OUT / "pytest-artifacts"),
        ]
    )
    report["pytest_exit_code"] = result
    (OUT / "CHECK.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(result)


if __name__ == "__main__":
    main()
