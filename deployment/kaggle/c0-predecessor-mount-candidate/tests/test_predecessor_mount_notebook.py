from __future__ import annotations

import ast
import base64
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
BUILDER = HERE / "build_predecessor_mount_notebook.py"
NOTEBOOK = HERE / "c0_predecessor_mount.ipynb"
METADATA = HERE / "kernel-metadata.json"
PROBE = HERE / "predecessor_mount_probe.py"
FIXTURE = HERE / "checkpoint-fixture.json"


def assignment_bytes(source: str, name: str) -> bytes:
    module = ast.parse(source)
    for statement in module.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == name
        ):
            value = ast.literal_eval(statement.value)
            assert isinstance(value, str)
            return base64.b64decode(value)
    raise AssertionError(f"missing assignment {name}")


def test_generated_notebook_embeds_exact_probe_and_fixture() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    assert len(notebook["cells"]) == 1
    source = notebook["cells"][0]["source"]
    compile(source, "generated-cell", "exec")
    assert assignment_bytes(source, "PROBE_B64") == PROBE.read_bytes()
    assert assignment_bytes(source, "FIXTURE_B64") == FIXTURE.read_bytes()
    assert "--input-root" in source
    assert "/kaggle/input" in source
    assert "/kaggle/working/tycho-nooa-predecessor-mount" in source


def test_kernel_metadata_is_private_offline_cpu_and_version_pinned() -> None:
    metadata = json.loads(METADATA.read_text())
    assert metadata == {
        "id": "pranavagrawai/prototype-tycho-nooa-predecessor-mount",
        "title": "PROTOTYPE Tycho NOOA predecessor mount",
        "code_file": "c0_predecessor_mount.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [
            "pranavagrawai/prototype-tycho-nooa-successor-concurrency/1"
        ],
        "model_sources": [],
    }


def test_builder_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.ipynb"
    second = tmp_path / "second.ipynb"
    for output in (first, second):
        subprocess.run(
            [sys.executable, str(BUILDER), "--output", str(output)],
            check=True,
            capture_output=True,
            text=True,
        )
    assert first.read_bytes() == second.read_bytes() == NOTEBOOK.read_bytes()
