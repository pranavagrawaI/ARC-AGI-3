"""Regress the generated notebook's real default-branch donor lifecycle."""
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
DEFAULT_PACKAGE = REPO / "deployment/kaggle" / HERE.name
PACKAGE = Path(os.environ.get("CONSTRUCTION_PACKAGE", DEFAULT_PACKAGE))
DONOR = REPO / "deployment/kaggle/submission-20260912-v15"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"


def cell_source(notebook, index):
    value = notebook["cells"][index]["source"]
    return "".join(value) if isinstance(value, list) else value


def default_branch_sources(source):
    tree = ast.parse(source)
    branch = next(node for node in tree.body if isinstance(node, ast.If) and node.orelse)
    exec_sources = []
    for node in branch.orelse:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if isinstance(call.func, ast.Name) and call.func.id == "exec" and call.args:
            if isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
                exec_sources.append(call.args[0].value)
    return branch, exec_sources


def localized_runtime_setup(source, root):
    mounts = root / "input"
    working = root / "working"
    patch = mounts / "patch"
    base = mounts / "base"
    solver = mounts / "solver"
    for directory in (patch, base, solver, working):
        directory.mkdir(parents=True)

    patch_wheel = patch / "sglang-0.5.19.dev591g8e04f66a7-cp312-cp312-linux_x86_64.whl"
    base_wheel = base / "sglang-0.5.19.dev591+g8e04f66a7-cp312-cp312-linux_x86_64.whl"
    lock_patch = patch / "requirements.lock"
    lock_base = base / "requirements.lock"
    solver_wheel = solver / "arc_agi_3_tycho_nooa-0-py3-none-any.whl"
    patch_wheel.write_bytes(b"patched-runtime-wheel-fixture")
    base_wheel.write_bytes(b"base-runtime-wheel-fixture")
    lock_patch.write_bytes(b"runtime-lock-fixture")
    lock_base.write_bytes(lock_patch.read_bytes())
    solver_wheel.write_bytes(b"solver-wheel-fixture")

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    dataset_manifest = {
        "schema": "sglang-r8l3b-thin-runtime-dataset-v1",
        "dataset": {"source": "pranavagrawai/sglang-r8l3b-orderly-shutdown-runtime"},
        "files": [{
            "path": "sglang-0.5.19.dev591+g8e04f66a7-cp312-cp312-linux_x86_64.whl",
            "sha256": digest(patch_wheel),
        }],
    }
    dataset_manifest_path = patch / "runtime-dataset-manifest.json"
    dataset_manifest_path.write_text(json.dumps(dataset_manifest))
    solver_manifest = {
        "schema": "tycho-nooa-full-wheelhouse-v1",
        "artifact_revision": 11,
        "wheels": [
            {"name": "arc-agi-3-tycho-nooa", "version": "0", "filename": solver_wheel.name, "sha256": digest(solver_wheel)},
            {"name": "nooa", "version": "0", "filename": "unused-nooa.whl", "sha256": "unused"},
        ],
    }
    solver_manifest_path = solver / "wheelhouse-manifest.json"
    solver_manifest_path.write_text(json.dumps(solver_manifest))
    (solver / "verify_wheelhouse.py").write_text("raise AssertionError('must be subprocess-stubbed')\n")

    replacements = {
        "/kaggle/input": str(mounts),
        "/kaggle/working": str(working),
        "06cc48d1e067c0eb08a1548c5c244bd0f541b114947c29b87042ce3156bb5cc0": digest(patch_wheel),
        "9d07f8bf3b718dc4cd970e72a03415b0b0597da89625acbee7a655d6387e7a8d": digest(base_wheel),
        "d473b6e092e01f66313e12069ac8873f2ff1c6c3b84b51c1b37312229de5475c": digest(lock_patch),
        "f1a50bb224dc26ac48c0fd219c28e4fadc92ede6d2a80a26fd18d9a004a79b94": digest(dataset_manifest_path),
        "518f8407b0ab1234430fc0c167e2ecd62ce70318a62292f8397e1a7ea4ed3862": digest(solver_manifest_path),
        "9ffadfc88ed40667fa14b876b5a17b94ac4992ca277377c887de0db5b0df7faf": digest(solver_wheel),
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    expected = {record["name"].replace("_", "-").lower(): record["version"] for record in solver_manifest["wheels"]}
    return source, expected


class DefaultBranchLifecycle(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads((PACKAGE / "construction.ipynb").read_text())
        self.donor = json.loads((DONOR / "submission.ipynb").read_text())
        self.source = cell_source(self.notebook, 1)

    def test_exact_donor_cells_and_launcher_order(self):
        branch, embedded = default_branch_sources(self.source)
        donor_one = cell_source(self.donor, 1)
        donor_two = cell_source(self.donor, 2)
        donor_three = cell_source(self.donor, 3)
        self.assertEqual(embedded, [donor_one, donor_two])
        self.assertNotIn(donor_three, self.source)
        command_node = next(node for node in branch.orelse if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "command" for target in node.targets))
        command_rendered = ast.unparse(command_node)
        self.assertNotIn("submission_play.py", command_rendered)
        self.assertNotIn("submission.parquet", command_rendered)
        self.assertIn("root / 'remote_entry.py'", command_rendered)
        self.assertIn("'/tmp/solver-venv/bin/python'", command_rendered)
        exec_indices = [index for index, node in enumerate(branch.orelse) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "exec"]
        self.assertEqual(len(exec_indices), 2)
        self.assertLess(max(exec_indices), branch.orelse.index(command_node))

    def test_metadata_is_exact_donor_arc_image_contract(self):
        meta = json.loads((PACKAGE / "kernel-metadata.json").read_text())
        donor = json.loads((DONOR / "kernel-metadata.json").read_text())
        for key in ("language", "kernel_type", "is_private", "enable_gpu", "machine_shape", "enable_internet", "enable_tpu", "dataset_sources", "kernel_sources", "competition_sources", "model_sources"):
            self.assertEqual(meta[key], donor[key], key)
        self.assertEqual(meta["competition_sources"], ["arc-prize-2026-arc-agi-3"])
        self.assertEqual(meta["docker_image"], IMAGE)

    def test_actual_default_branch_runs_setup_before_launcher(self):
        branch, embedded = default_branch_sources(self.source)
        self.assertEqual(len(embedded), 2, "complete donor setup is absent")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            localized_two, installed = localized_runtime_setup(embedded[1], root)
            tree = ast.parse(self.source.replace("/tmp/construction-policy-integration", str(root / "assets")))
            default_if = next(node for node in tree.body if isinstance(node, ast.If) and node.orelse)
            exec_nodes = [node for node in default_if.orelse if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "exec"]
            self.assertEqual(len(exec_nodes), 2)
            runtime_exec = cast(ast.Call, exec_nodes[1].value)
            self.assertIsInstance(runtime_exec, ast.Call)
            runtime_exec.args[0] = ast.Constant(localized_two)
            ast.fix_missing_locations(tree)
            calls = []
            launcher_observation = {}

            def fake_run(command, *args, **kwargs):
                command = [str(value) for value in command]
                calls.append(command)
                if len(command) > 1 and command[1] == "venv":
                    solver_venv = Path(command[-1])
                    (solver_venv / "bin").mkdir(parents=True)
                    (solver_venv / "bin/python").write_text("fixture executable\n")
                if len(command) > 1 and command[1] == "/tmp/official_launcher_runtime.py":
                    launcher_observation.update(
                        provenance=Path(command[command.index("--runtime-provenance") + 1]).is_file(),
                        solver_python=Path(command[command.index("--launcher-python") + 1]).is_file(),
                        remote_entry=Path(command[command.index("--official-launcher-probe") + 1]).name,
                    )
                stdout = kwargs.get("stdout")
                if stdout is not None and hasattr(stdout, "write"):
                    stdout.write("[]")
                return subprocess.CompletedProcess(command, 0, "", "")

            old_fixture = os.environ.pop("CONSTRUCTION_FIXTURE", None)
            old_local = os.environ.pop("CONSTRUCTION_LOCAL_BASE_URL", None)
            module_names = ("accelerator", "constructor", "containment_base", "integration", "integration_boundary", "local_transport", "real_trace", "request_contract")
            saved_modules = {name: sys.modules.get(name) for name in module_names}
            for path in (Path("/tmp/sglang-runtime-provenance.json"), Path("/tmp/solver-venv")):
                if path.is_dir(): shutil.rmtree(path)
                elif path.exists(): path.unlink()
            try:
                with mock.patch("subprocess.run", side_effect=fake_run), mock.patch("subprocess.check_output", return_value=json.dumps(installed)), mock.patch("shutil.which", return_value="/fixture/uv"):
                    exec(compile(tree, str(PACKAGE / "construction.ipynb"), "exec"), {"__name__": "__main__"})
            finally:
                if old_fixture is not None: os.environ["CONSTRUCTION_FIXTURE"] = old_fixture
                if old_local is not None: os.environ["CONSTRUCTION_LOCAL_BASE_URL"] = old_local
                for name, value in saved_modules.items():
                    sys.modules.pop(name, None)
                    if value is not None:
                        sys.modules[name] = value
                for path in (Path("/tmp/sglang-runtime-provenance.json"), Path("/tmp/solver-venv")):
                    if path.is_dir(): shutil.rmtree(path)
                    elif path.exists(): path.unlink()
            self.assertEqual(launcher_observation, {"provenance": True, "solver_python": True, "remote_entry": "remote_entry.py"})
            launcher_index = next(i for i, command in enumerate(calls) if len(command) > 1 and command[1] == "/tmp/official_launcher_runtime.py")
            venv_index = next(i for i, command in enumerate(calls) if len(command) > 1 and command[1] == "venv")
            self.assertLess(venv_index, launcher_index)


if __name__ == "__main__":
    unittest.main()
