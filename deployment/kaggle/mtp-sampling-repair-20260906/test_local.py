import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from build import build, DONOR
import toolchain
from watchdog import guarded_call


class Checks(unittest.TestCase):
    def test_selects_new_compiler_instead_of_old_default(self):
        distribution = SimpleNamespace(
            files=["nvidia/cu13/bin/nvcc"], locate_file=lambda f: Path("/new") / f
        )

        def inspect(path):
            old = str(path).startswith("/old")
            return {
                "nvcc": str(path),
                "root": "/old" if old else "/new/nvidia/cu13",
                "version": [12, 8] if old else [13, 3],
                "gpu_codes": ["sm_120"],
                "headers_present": True,
            }

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(toolchain.shutil, "which", return_value="/old/bin/nvcc"),
            patch.object(toolchain.metadata, "distribution", return_value=distribution),
            patch.object(toolchain, "inspect_nvcc", side_effect=inspect),
        ):
            result = toolchain.select()
            self.assertEqual(result["selected"]["version"], [13, 3])
            self.assertEqual(os.environ["CUDA_HOME"], "/new/nvidia/cu13")
            self.assertNotIn("FLASHINFER_CUDA_ARCH_LIST", os.environ)

    def test_no_compiler_is_a_hard_failure(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(toolchain.shutil, "which", return_value=None),
            patch.object(
                toolchain.metadata,
                "distribution",
                side_effect=toolchain.metadata.PackageNotFoundError,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "No complete"):
                toolchain.select()

    def test_actual_flashinfer_empty_target_failure_and_recovery(self):
        toolchain.select()
        import flashinfer.jit.core as core
        import flashinfer.jit.cpp_ext as cpp
        from flashinfer.compilation_context import CompilationContext

        with patch.object(cpp, "is_cuda_version_at_least", return_value=False):
            context = CompilationContext()
        self.assertEqual(context.TARGET_CUDA_ARCHS, set())
        with patch.object(core, "current_compilation_context", context):
            with self.assertRaisesRegex(RuntimeError, "sm75"):
                core.check_cuda_arch()
        context = CompilationContext()
        self.assertIn((12, "0f"), context.TARGET_CUDA_ARCHS)
        with patch.object(core, "current_compilation_context", context):
            core.check_cuda_arch()

    def test_engine_death_does_not_wait_900_seconds(self):
        release = threading.Event()
        server = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
            stdin=subprocess.PIPE,
        )

        def hung_request():
            server.stdin.write(b"x")
            server.stdin.flush()
            release.wait(5)

        start = time.monotonic()
        try:
            with self.assertRaisesRegex(RuntimeError, "Engine exited"):
                guarded_call(hung_request, server, timeout=900)
            self.assertLess(time.monotonic() - start, 2)
        finally:
            release.set()
            server.wait(timeout=5)
            server.stdin.close()

    def test_success_and_original_exception_propagate(self):
        server = SimpleNamespace(poll=lambda: None)
        self.assertEqual(guarded_call(lambda: 42, server), 42)

        def failed():
            raise ValueError("original failure")

        with self.assertRaisesRegex(ValueError, "original failure"):
            guarded_call(failed, server)

    def test_notebook_retains_corpus_and_adds_preload_gate(self):
        outputs = build()
        notebook = json.loads(outputs["sampling_repair.ipynb"])
        original = json.loads((DONOR / "actor_mtp.ipynb").read_text())
        self.assertEqual(notebook["cells"][1], original["cells"][1])
        self.assertIn("sampling_preflight.py", notebook["cells"][4]["source"])
        self.assertIn("spec_verify_ct", outputs["runner.py"].decode())
        self.assertIn(
            "stochastic-preflight-request.json", outputs["runner.py"].decode()
        )
        self.assertEqual(
            json.loads(outputs["kernel-metadata.json"])["competition_sources"],
            ["arc-prize-2026-arc-agi-3"],
        )


if __name__ == "__main__":
    unittest.main()
