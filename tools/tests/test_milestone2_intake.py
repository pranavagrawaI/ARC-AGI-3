"""Exercise the intake CLI and its artifacts; donor code must never run."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "milestone2_intake.py"


class IntakeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.duck = self.root / "duck"
        self.output = self.root / "output with spaces"
        self.sentinel = self.root / "DONOR_EXECUTED"
        self.code = (
            "import subprocess\nimport numpy as np\nfrom pathlib import Path\n"
            f"Path({str(self.sentinel)!r}).touch()\n"
            "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-index', 'example==1.2'])\n"
            "ROOT = '/kaggle/input/author/data/wheels'\n"
            "MODEL = 'https://example.com/model'\n"
            "token = UserSecretsClient().get_secret('SERVICE_TOKEN')\n"
            "setting = os.environ.get('MODEL_NAME')\n"
            "solver = pickle.load(open('solver.pkl', 'rb'))\n"
        )
        self.make_bundle(self.source, "author/winner", self.code)
        self.make_bundle(self.duck, "duck/baseline", "import numpy as np\nprint('baseline')\n")

    def make_bundle(self, folder, ref, code):
        folder.mkdir()
        meta = dict(id=ref, id_no=12345, title="Public winner", code_file="source.ipynb",
                    language="python", kernel_type="notebook", is_private=False,
                    enable_gpu=True, enable_tpu=False, enable_internet=False,
                    machine_shape="NvidiaRtxPro6000", docker_image="image@sha256:abc",
                    dataset_sources=["author/data"], model_sources=["author/model/pyTorch/default/1"],
                    kernel_sources=["author/helper/3"], competition_sources=["arc-competition"])
        notebook = dict(nbformat=4, nbformat_minor=5, metadata={}, cells=[
            dict(cell_type="markdown", metadata={}, source=["Source attribution"]),
            dict(cell_type="code", metadata={}, source=code.splitlines(keepends=True),
                 execution_count=7, outputs=[dict(output_type="stream", name="stdout", text="old output")])])
        self.write(folder / "kernel-metadata.json", meta)
        self.write(folder / "source.ipynb", notebook)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    @staticmethod
    def read(path):
        return json.loads(path.read_text())

    def run_intake(self, *extra, live=False):
        command = [sys.executable, str(SCRIPT), "https://www.kaggle.com/code/author/winner",
                   "--duck", "duck/baseline", "--owner", "myuser", "--slug", "private-winner",
                   "--output", str(self.output)]
        if not live:
            command += ["--source-dir", str(self.source), "--duck-dir", str(self.duck)]
        return subprocess.run(command + list(extra), capture_output=True, text=True, timeout=20)

    def test_private_copy_preserves_code_inputs_and_runtime_with_a_new_identity(self):
        original = {p.name: p.read_bytes() for p in self.source.iterdir()}
        result = self.run_intake()
        self.assertEqual(result.returncode, 0, result.stderr)
        meta = self.read(self.output / "kernel/kernel-metadata.json")
        self.assertEqual(meta["id"], "myuser/private-winner")
        self.assertNotIn("id_no", meta)
        self.assertIs(meta["is_private"], True)
        self.assertIs(meta["enable_internet"], False)
        self.assertEqual(meta["machine_shape"], "NvidiaRtxPro6000")
        self.assertEqual(meta["docker_image_pinning_type"], "original")
        self.assertEqual(meta["docker_image"], "image@sha256:abc")
        self.assertEqual(meta["dataset_sources"], ["author/data"])
        self.assertEqual(meta["model_sources"], ["author/model/pyTorch/default/1"])
        self.assertEqual(meta["kernel_sources"], ["author/helper/3"])
        self.assertEqual(meta["competition_sources"], ["arc-competition"])
        nb = self.read(self.output / "kernel/source.ipynb")
        self.assertEqual(nb["metadata"]["kernelspec"]["name"], "python3")
        self.assertEqual(nb["metadata"]["language_info"]["name"], "python")
        self.assertEqual(len({c["id"] for c in nb["cells"]}), 2)
        self.assertEqual("".join(nb["cells"][1]["source"]), self.code)
        self.assertFalse(self.sentinel.exists())
        for name, data in original.items():
            self.assertEqual((self.source / name).read_bytes(), data)
            self.assertEqual((self.output / "candidate" / name).read_bytes(), data)
        report = self.read(self.output / "intake.json")
        inv = report["candidate_inventory"]
        self.assertIn("numpy", inv["imports"])
        self.assertIn("example==1.2", inv["install_commands"][0]["text"])
        self.assertEqual(inv["secrets"][0]["name"], "SERVICE_TOKEN")
        self.assertEqual(inv["environment_variables"][0]["name"], "MODEL_NAME")
        self.assertEqual(inv["input_paths"][0]["value"], "/kaggle/input/author/data/wheels")
        self.assertTrue(inv["external_code"])
        self.assertIn("subprocess", report["differences"]["imports_added"])
        diff = (self.output / "code.diff").read_text()
        self.assertIn("-print('baseline')", diff)
        self.assertIn("+solver = pickle.load", diff)
        self.assertNotIn("old output", diff)
        for filename, sha in report["sha256"].items():
            self.assertEqual(hashlib.sha256((self.output / filename).read_bytes()).hexdigest(), sha)

    def test_live_cli_path_only_pulls_and_prepares(self):
        # A process boundary fixture fails on any command capable of running a kernel.
        fake = self.root / "kaggle"
        log = self.root / "calls.jsonl"
        fake.write_text(f"#!{sys.executable}\n" +
                        "import json, pathlib, shutil, sys\n" +
                        f"log = pathlib.Path({str(log)!r})\n" +
                        "with log.open('a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n" +
                        "if sys.argv[1:] == ['--version']: print('Kaggle CLI fixture'); sys.exit(0)\n" +
                        "assert sys.argv[1:3] == ['kernels', 'pull'], 'Unexpected remote operation'\n" +
                        f"source = {str(self.source)!r} if sys.argv[3] == 'author/winner' else {str(self.duck)!r}\n" +
                        "assert '--metadata' in sys.argv\n" +
                        "shutil.copytree(source, sys.argv[sys.argv.index('--path') + 1])\n")
        fake.chmod(0o755)
        result = self.run_intake("--kaggle", str(fake), live=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual([c[:2] for c in calls], [["--version"], ["kernels", "pull"], ["kernels", "pull"]])
        self.assertTrue((self.output / "kernel/kernel-metadata.json").is_file())
        self.assertFalse(self.sentinel.exists())

    def test_existing_output_is_never_overwritten_even_on_failure(self):
        self.output.mkdir()
        sentinel = self.output / "ERROR.json"
        sentinel.write_text("preserve me")
        result = self.run_intake()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(sentinel.read_text(), "preserve me")

    def test_missing_accelerator_blocks_command_and_override_resolves_it(self):
        path = self.source / "kernel-metadata.json"
        meta = self.read(path)
        del meta["machine_shape"]
        self.write(path, meta)
        result = self.run_intake()
        self.assertEqual(result.returncode, 2, result.stderr)
        report = self.read(self.output / "intake.json")
        self.assertIsNone(report["commit_command"])
        self.assertIn("machine_shape", report["blockers"][0])
        self.output = self.root / "override"
        result = self.run_intake("--accelerator", "NvidiaRtxPro6000", "--timeout", "7200")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--timeout 7200", self.read(self.output / "intake.json")["commit_command"])

    def test_string_false_is_not_coerced_to_true(self):
        path = self.source / "kernel-metadata.json"
        meta = self.read(path)
        for field in ("is_private", "enable_gpu", "enable_tpu", "enable_internet"):
            meta[field] = str(meta[field]).lower()
        self.write(path, meta)
        result = self.run_intake()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIs(self.read(self.output / "kernel/kernel-metadata.json")["enable_internet"], False)

    def test_metadata_cannot_read_outside_pull_directory(self):
        path = self.source / "kernel-metadata.json"
        meta = self.read(path)
        meta["code_file"] = "../outside.py"
        (self.root / "outside.py").write_text("secret")
        self.write(path, meta)
        result = self.run_intake()
        self.assertEqual(result.returncode, 1)
        self.assertIn("inside the pulled directory", result.stderr)
        self.assertFalse((self.output / "kernel").exists())

    def test_refuses_donor_destination_and_ambiguous_version_url(self):
        result = self.run_intake("--owner", "author", "--slug", "winner")
        self.assertEqual(result.returncode, 1)
        self.assertFalse(self.output.exists())
        result = subprocess.run([sys.executable, str(SCRIPT),
                                 "https://www.kaggle.com/code/author/winner?scriptVersionId=123"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("ambiguous", result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("kaggle"), "requires installed Kaggle CLI Python environment")
    def test_real_kaggle_request_projection_without_network(self):
        from contextlib import nullcontext
        from kaggle.api.kaggle_api_extended import KaggleApi

        result = self.run_intake()
        self.assertEqual(result.returncode, 0, result.stderr)
        captured = []
        client = SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=SimpleNamespace(
            save_kernel=lambda request: captured.append(request))))
        api = KaggleApi()
        # This runs Kaggle's real parser/serializer; only the network boundary is replaced.
        with patch.object(api, "build_kaggle_client", return_value=nullcontext(client)):
            api.kernels_push(str(self.output / "kernel"), timeout=7200, acc="NvidiaRtxPro6000")
        self.assertEqual(len(captured), 1)
        request = captured[0]
        self.assertFalse(request.id)
        self.assertEqual(request.slug, "myuser/private-winner")
        self.assertIs(request.is_private, True)
        self.assertIs(request.enable_internet, False)
        self.assertEqual(request.machine_shape, "NvidiaRtxPro6000")
        self.assertEqual(request.docker_image, "image@sha256:abc")
        self.assertEqual(request.docker_image_pinning_type, "original")
        self.assertEqual(request.session_timeout_seconds, 7200)
        self.assertEqual(list(request.dataset_data_sources), ["author/data"])
        self.assertEqual(list(request.model_data_sources), ["author/model/pyTorch/default/1"])
        self.assertEqual(list(request.kernel_data_sources), ["author/helper/3"])
        self.assertEqual(list(request.competition_data_sources), ["arc-competition"])
        notebook = json.loads(request.text)
        self.assertEqual(notebook["cells"][1]["source"], self.code)
        self.assertFalse(self.sentinel.exists())

    @unittest.skipUnless(importlib.util.find_spec("nbformat"), "requires nbformat for schema validation")
    def test_repaired_notebook_validates_against_jupyter_schema(self):
        import nbformat

        # The live Duck notebooks use an older schema that disallows cell IDs.
        path = self.source / "source.ipynb"
        notebook = self.read(path)
        notebook["nbformat_minor"] = 2
        self.write(path, notebook)
        result = self.run_intake()
        self.assertEqual(result.returncode, 0, result.stderr)
        repaired = nbformat.read(self.output / "kernel/source.ipynb", as_version=4)
        nbformat.validate(repaired)
        self.assertEqual(repaired["nbformat_minor"], 5)


if __name__ == "__main__":
    unittest.main()
