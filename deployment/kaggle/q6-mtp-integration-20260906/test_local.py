"""Execute notebook asset extraction locally; never install or launch a server."""

import ast
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from build import build, DONOR, FLAGS


class Checks(unittest.TestCase):
    def test_actual_asset_cell_and_unchanged_embedded_solver(self):
        outputs = build()
        manifest = json.loads(outputs["manifest.json"])
        notebook = json.loads(outputs["q6_mtp.ipynb"])
        tree = ast.parse("".join(notebook["cells"][2]["source"]))
        # Redirect only Path imports for a safe temporary asset-extraction test.
        tree.body = [
            n
            for n in tree.body
            if not (isinstance(n, ast.ImportFrom) and n.module == "pathlib")
        ]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            def local_path(path):
                result = root / str(path).lstrip("/")
                result.parent.mkdir(parents=True, exist_ok=True)
                return result

            exec(compile(tree, "asset-cell", "exec"), {"Path": local_path})
            runtime = local_path("/tmp/official_launcher_runtime.py").read_bytes()
            self.assertEqual(runtime, outputs["runtime.py"])
            raw = local_path("/tmp/nooa-join.zip").read_bytes()
            self.assertEqual(
                hashlib.sha256(raw).hexdigest(), manifest["join_zip_sha256"]
            )
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                for name, digest in manifest["solver_files"].items():
                    self.assertEqual(
                        hashlib.sha256(archive.read(name)).hexdigest(), digest
                    )
            self.assertEqual(
                json.loads(
                    local_path("/kaggle/working/INTEGRATION-IDENTITY.json").read_text()
                )["runtime_sha256"],
                manifest["runtime_sha256"],
            )

    def test_actual_engine_function_adds_only_selected_flags(self):
        def command(source):
            fn = next(
                n
                for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "sglang_command"
            )
            module = ast.Module(body=[fn], type_ignores=[])
            scope = {"Path": Path, "shutil": shutil}
            exec(compile(module, "engine-command", "exec"), scope)
            with patch.object(shutil, "which", return_value="/fixture/sglang"):
                return scope["sglang_command"](
                    Path("/model"),
                    "127.0.0.1",
                    30000,
                    attention_backend="triton",
                    sampling_backend="pytorch",
                )

        original = command((DONOR / "exact_request_outer_runtime.py").read_text())
        candidate = command(build()["runtime.py"].decode())
        self.assertEqual(candidate, original + FLAGS)

    def test_upload_binding_and_bounds(self):
        outputs = build()
        metadata = json.loads(outputs["kernel-metadata.json"])
        self.assertEqual(metadata["competition_sources"], ["arc-prize-2026-arc-agi-3"])
        self.assertTrue(metadata["is_private"])
        self.assertFalse(metadata["enable_internet"])
        notebook = json.loads(outputs["q6_mtp.ipynb"])
        run = "".join(notebook["cells"][-1]["source"])
        for required in [
            "TYCHO_KAGGLE_MAX_ACTIONS'] = '10'",
            "TYCHO_GAMEPLAY_ADMISSION_SECONDS'] = '1800'",
            "'--launcher-timeout-s', '3000'",
            "TYCHO_NOOA_WORKER_TIMEOUT_SECONDS'] = '900'",
            "ARC_STALL_REVIEW'] = '0'",
            "ARC_LLM_MAX_TOKENS'] = '4096'",
        ]:
            self.assertIn(required, run)


if __name__ == "__main__":
    unittest.main()
