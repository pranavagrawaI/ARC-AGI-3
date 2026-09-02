from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

PROBE_PATH = Path(__file__).resolve().parents[1] / "PROTOTYPE-qwen3_8_r8_l3b_probe.py"
LAUNCH_PATH = Path(__file__).resolve().parents[1] / "r8l3b-launch.json"
NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "r8l3b-notebook/r8l3b_probe.ipynb"
METADATA_PATH = Path(__file__).resolve().parents[1] / "r8l3b-notebook/kernel-metadata.json"
KAGGLE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = KAGGLE_ROOT.parents[1]
MODEL_MANIFEST_PATH = KAGGLE_ROOT / "PROTOTYPE-qwen3_8_fp8-manifest.json"
NOTEBOOK_BUILDER_PATH = KAGGLE_ROOT / "scripts/build_r8l3b_probe_notebook.py"
DATASET_BUILDER_PATH = KAGGLE_ROOT / "scripts/build_r8l3b_runtime_dataset.py"
DATASET_ROOT = KAGGLE_ROOT / "r8l3b-runtime-dataset"
DATASET_MANIFEST_PATH = DATASET_ROOT / "runtime-dataset-manifest.json"
DATASET_UPLOAD_PATH = KAGGLE_ROOT / "r8l3b-dataset-upload.json"
CANDIDATE_PATH = KAGGLE_ROOT / "r8l3b-candidate.json"
WHEEL_NAME = "sglang-0.5.19.dev591+g8e04f66a7-cp312-cp312-linux_x86_64.whl"
REMOTE_WHEEL_NAME = "sglang-0.5.19.dev591g8e04f66a7-cp312-cp312-linux_x86_64.whl"
REMOTE_DATASET_ROOT = (
    REPOSITORY_ROOT
    / "analysis/deployment-probes/20260831T140711Z-r8l3b-dataset-upload-v1/remote-v1"
)
BASE_WHEEL_PATH = Path("/tmp/sglang-r8l3-runtime") / WHEEL_NAME
SPEC = importlib.util.spec_from_file_location("r8l3b_probe", PROBE_PATH)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_text(path: Path, marker: str, timeout_s: float = 5) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists() and marker in path.read_text(errors="replace"):
            return
        time.sleep(0.02)
    raise AssertionError(f"{marker!r} did not appear in {path}")


class CandidateArtifactTests(unittest.TestCase):
    def test_candidate_manifest_recomputes_every_participant(self) -> None:
        candidate = json.loads(CANDIDATE_PATH.read_text())
        participants = [
            *candidate["dataset_candidate"]["participants"],
            *candidate["notebook_candidate"]["participants"],
        ]
        for participant in participants:
            path = KAGGLE_ROOT.parents[1] / participant["path"]
            with self.subTest(path=path):
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    participant["sha256"],
                )

    def test_dataset_upload_binding_is_private_one_shot_and_non_authorizing(self) -> None:
        binding = json.loads(DATASET_UPLOAD_PATH.read_text())
        self.assertIn("does not grant permission", binding["authority"])
        self.assertEqual(binding["maximum_uploads"], 1)
        self.assertEqual(binding["automatic_retries"], 0)
        self.assertTrue(binding["expected_private"])
        self.assertNotIn("--public", binding["argv"])
        self.assertEqual(
            binding["candidate_binding"]["manifest_sha256"],
            hashlib.sha256(DATASET_MANIFEST_PATH.read_bytes()).hexdigest(),
        )

    def test_dataset_manifest_closes_every_regular_file(self) -> None:
        manifest = json.loads(DATASET_MANIFEST_PATH.read_text())
        records = {record["path"]: record for record in manifest["files"]}
        actual_names = {path.name for path in DATASET_ROOT.iterdir()}
        self.assertIn("dataset-metadata.json", records)
        self.assertNotIn("datasets-metadata.json", actual_names)
        self.assertEqual(actual_names, set(records) | {DATASET_MANIFEST_PATH.name})
        self.assertEqual(
            manifest["dataset"]["source"],
            "pranavagrawai/sglang-r8l3b-orderly-shutdown-runtime",
        )
        self.assertEqual(manifest["dataset"]["expected_initial_version"], 1)
        self.assertTrue(manifest["dataset"]["private"])
        for name, record in records.items():
            path = DATASET_ROOT / name
            with self.subTest(path=path):
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())
                self.assertEqual(path.stat().st_size, record["bytes"])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])

    def test_dataset_builder_reproduces_production_bytes(self) -> None:
        production = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in DATASET_ROOT.iterdir()
        }
        with tempfile.TemporaryDirectory() as temporary:
            roots = [Path(temporary) / "first", Path(temporary) / "second"]
            for root in roots:
                subprocess.run(
                    [sys.executable, str(DATASET_BUILDER_PATH), "--output-dir", str(root)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                observed = {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in root.iterdir()
                }
                self.assertEqual(observed, production)

    def test_notebook_builder_and_embedded_assets_are_exact(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile(cell["source"], f"notebook-cell-{index}", "exec")

        asset_tree = ast.parse(notebook["cells"][1]["source"])
        encoded = [
            node.args[0].value
            for node in ast.walk(asset_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "b64decode"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ]
        self.assertEqual(len(encoded), 2)
        self.assertEqual(base64.b64decode(encoded[0]), PROBE_PATH.read_bytes())
        self.assertEqual(base64.b64decode(encoded[1]), MODEL_MANIFEST_PATH.read_bytes())

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "notebook"
            command = [
                sys.executable,
                str(NOTEBOOK_BUILDER_PATH),
                "--kernel-id",
                "pranavagrawai/prototype-qwen3-8-fp8-r8-b-request-path",
                "--model-dataset-source",
                "jakobbrggen/qwen3-8-27b-fp8-hf-snapshot",
                "--base-runtime-dataset-source",
                "mahdiseddigh/kaggle-t4-stack-2026-08-28",
                "--patch-runtime-dataset-source",
                "pranavagrawai/sglang-r8l3b-orderly-shutdown-runtime",
                "--output-dir",
                str(output),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual((output / NOTEBOOK_PATH.name).read_bytes(), NOTEBOOK_PATH.read_bytes())
            self.assertEqual((output / METADATA_PATH.name).read_bytes(), METADATA_PATH.read_bytes())

    def _execute_install_cell(
        self,
        temporary: Path,
        *,
        mutate_manifest: bool = False,
        use_untransformed_wheel_name: bool = False,
        duplicate_patch_wheel: bool = False,
    ) -> dict:
        notebook = json.loads(NOTEBOOK_PATH.read_text())
        source = notebook["cells"][2]["source"]
        mounts = temporary / "input"
        base_root = mounts / "kaggle-t4-stack-2026-08-28/stack"
        base_wheelhouse = base_root / "wheelhouse"
        patch_root = mounts / "sglang-r8l3b-orderly-shutdown-runtime"
        base_wheelhouse.mkdir(parents=True)
        patch_root.mkdir(parents=True)
        shutil.copyfile(BASE_WHEEL_PATH, base_wheelhouse / WHEEL_NAME)
        shutil.copyfile(Path("/tmp/sglang-r8l3-runtime/requirements.lock"), base_root / "requirements.lock")
        mounted_wheel_name = WHEEL_NAME if use_untransformed_wheel_name else REMOTE_WHEEL_NAME
        shutil.copyfile(REMOTE_DATASET_ROOT / REMOTE_WHEEL_NAME, patch_root / mounted_wheel_name)
        if duplicate_patch_wheel:
            duplicate_root = mounts / "duplicate-patch"
            duplicate_root.mkdir()
            shutil.copyfile(
                REMOTE_DATASET_ROOT / REMOTE_WHEEL_NAME,
                duplicate_root / REMOTE_WHEEL_NAME,
            )
        shutil.copyfile(
            REMOTE_DATASET_ROOT / "requirements.lock",
            patch_root / "requirements.lock",
        )
        shutil.copyfile(
            REMOTE_DATASET_ROOT / DATASET_MANIFEST_PATH.name,
            patch_root / DATASET_MANIFEST_PATH.name,
        )
        if mutate_manifest:
            with (patch_root / DATASET_MANIFEST_PATH.name).open("ab") as handle:
                handle.write(b"\n")

        working = temporary / "working"
        working.mkdir()
        provenance_path = temporary / "runtime-provenance.json"
        overrides_path = temporary / "runtime-overrides.txt"
        source = source.replace("/kaggle/input", str(mounts))
        source = source.replace("/kaggle/working", str(working))
        source = source.replace("/tmp/sglang-runtime-provenance.json", str(provenance_path))
        source = source.replace("/tmp/sglang-runtime-overrides.txt", str(overrides_path))
        with (
            mock.patch("subprocess.run") as run,
            mock.patch("shutil.which", return_value="/fake/uv"),
        ):
            exec(compile(source, "install-cell", "exec"), {})
        self.assertEqual(run.call_count, 3)
        provenance = json.loads(provenance_path.read_text())
        canonical_wheel = Path(provenance["artifact_path"])
        self.assertEqual(canonical_wheel.name, WHEEL_NAME)
        self.assertEqual(
            hashlib.sha256(canonical_wheel.read_bytes()).hexdigest(),
            PROBE.RUNTIME_WHEEL_SHA256,
        )
        runtime_install = run.call_args_list[1].args[0]
        self.assertIn(str(canonical_wheel), runtime_install)
        self.assertNotIn(str(patch_root / REMOTE_WHEEL_NAME), runtime_install)
        return provenance

    def test_install_projection_uses_patch_wheel_and_base_wheelhouse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            provenance = self._execute_install_cell(Path(temporary))
        self.assertEqual(provenance["artifact_sha256"], PROBE.RUNTIME_WHEEL_SHA256)
        self.assertEqual(
            provenance["base_artifact_sha256"], PROBE.BASE_RUNTIME_WHEEL_SHA256
        )
        self.assertEqual(
            provenance["dataset_manifest_sha256"],
            PROBE.RUNTIME_DATASET_MANIFEST_SHA256,
        )
        self.assertEqual(provenance["lock_sha256"], PROBE.RUNTIME_LOCK_SHA256)
        self.assertEqual(provenance["mounted_artifact_name"], REMOTE_WHEEL_NAME)
        self.assertEqual(provenance["canonical_artifact_name"], WHEEL_NAME)
        self.assertEqual(
            provenance["remote_dataset_readback_sha256"],
            "cee470f08217da591b8a89d3a80f1f9b4f03aecffced29f69d1b2a5c510fda9d",
        )

    def test_install_projection_rejects_untransformed_remote_wheel_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "provider-normalized patched wheel"):
                self._execute_install_cell(
                    Path(temporary),
                    use_untransformed_wheel_name=True,
                )

    def test_install_projection_rejects_manifest_drift_before_install(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "patch dataset manifest"):
                self._execute_install_cell(Path(temporary), mutate_manifest=True)

    def test_install_projection_rejects_duplicate_exact_patch_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "expected one exact"):
                self._execute_install_cell(
                    Path(temporary),
                    duplicate_patch_wheel=True,
                )


class LifecycleObserverTests(unittest.TestCase):
    def test_self_test_and_r8l3b_are_mutually_exclusive(self) -> None:
        argv = [
            str(PROBE_PATH),
            "--output-dir",
            "/tmp/r8l3b-arg-test",
            "--self-test",
            "--r8-l3b",
        ]
        with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            PROBE.parse_args()

    def test_r8l3b_requires_exact_timeout_treatment(self) -> None:
        base = [
            str(PROBE_PATH),
            "--output-dir",
            "/tmp/r8l3b-timeout-test",
            "--r8-l3b",
            "--attention-backend",
            "triton",
            "--sampling-backend",
            "pytorch",
        ]
        for flag, value in (
            ("--startup-timeout-s", "1199"),
            ("--startup-timeout-s", "1201"),
            ("--request-timeout-s", "179"),
            ("--request-timeout-s", "181"),
        ):
            with self.subTest(flag=flag, value=value):
                with mock.patch.object(sys, "argv", [*base, flag, value]):
                    with self.assertRaises(SystemExit):
                        PROBE.parse_args()

    def test_delayed_first_quiet_scan_still_measures_full_window(self) -> None:
        class FakeClock:
            value = 0.0

            def monotonic(self) -> float:
                return self.value

            def sleep(self, seconds: float) -> None:
                self.value += seconds

        clock = FakeClock()
        target = {
            "pid": 42,
            "identity": "42:100",
            "start_time": 100,
            "pgid": 42,
            "sid": 42,
            "owner_marker": True,
        }
        observer = {"pid": 7, "pgid": 7, "sid": 7}
        calls = 0

        def scan(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {
                    "monotonic_s": clock.value,
                    "processes": [target],
                    "listeners": {"inodes": []},
                    "gpu_processes": {},
                    "unattributed_gpu_processes": {},
                    "closed": False,
                }
            if calls == 2:
                clock.value += 0.3
            return {
                "monotonic_s": clock.value,
                "processes": [],
                "listeners": {"inodes": []},
                "gpu_processes": {},
                "unattributed_gpu_processes": {},
                "closed": True,
            }

        process = mock.Mock(pid=42, returncode=0)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "server.log").write_bytes(b"")
            with (
                mock.patch.object(PROBE, "process_identity", return_value=target),
                mock.patch.object(PROBE, "observer_identity", return_value=observer),
                mock.patch.object(PROBE, "scan_lifecycle", side_effect=scan),
                mock.patch.object(PROBE, "_proc_stat", return_value={"start_time": 100}),
                mock.patch.object(PROBE.os, "kill"),
                mock.patch.object(PROBE.time, "monotonic", side_effect=clock.monotonic),
                mock.patch.object(PROBE.time, "sleep", side_effect=clock.sleep),
            ):
                result = PROBE.qualify_lifecycle(
                    process,
                    mock.Mock(),
                    output,
                    port=30000,
                    term_grace_s=1,
                    kill_wait_s=1,
                    quiet_window_s=1,
                    deadline_s=5,
                    poll_s=0.05,
                    gpu_query=lambda: {},
                )

        self.assertEqual(result["status"], "passed", result)
        self.assertTrue(result["quiet_window_valid"])
        self.assertGreaterEqual(result["quiet_scan_count"], 2)
        self.assertEqual(result["quiet_window_anchor_phase"], "term_grace")
        self.assertAlmostEqual(result["quiet_window_anchor_monotonic_s"], 0.3)
        self.assertAlmostEqual(result["first_closed_scan_completed_elapsed_s"], 0.3)

    def test_term_scan_completing_after_grace_cannot_pass(self) -> None:
        class FakeClock:
            value = 0.0

            def monotonic(self) -> float:
                return self.value

            def sleep(self, seconds: float) -> None:
                self.value += seconds

        clock = FakeClock()
        target = {
            "pid": 42,
            "identity": "42:100",
            "start_time": 100,
            "pgid": 42,
            "sid": 42,
            "owner_marker": True,
        }
        observer = {"pid": 7, "pgid": 7, "sid": 7}
        calls = 0

        def scan(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {
                    "monotonic_s": clock.value,
                    "processes": [target],
                    "listeners": {"inodes": []},
                    "gpu_processes": {},
                    "unattributed_gpu_processes": {},
                    "closed": False,
                }
            if calls == 2:
                clock.value += 1.1
            return {
                "monotonic_s": clock.value,
                "processes": [],
                "listeners": {"inodes": []},
                "gpu_processes": {},
                "unattributed_gpu_processes": {},
                "closed": True,
            }

        process = mock.Mock(pid=42, returncode=0)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "server.log").write_bytes(b"")
            with (
                mock.patch.object(PROBE, "process_identity", return_value=target),
                mock.patch.object(PROBE, "observer_identity", return_value=observer),
                mock.patch.object(PROBE, "scan_lifecycle", side_effect=scan),
                mock.patch.object(PROBE, "_proc_stat", return_value={"start_time": 100}),
                mock.patch.object(PROBE.os, "kill"),
                mock.patch.object(PROBE.time, "monotonic", side_effect=clock.monotonic),
                mock.patch.object(PROBE.time, "sleep", side_effect=clock.sleep),
            ):
                result = PROBE.qualify_lifecycle(
                    process,
                    mock.Mock(),
                    output,
                    port=30000,
                    term_grace_s=1,
                    kill_wait_s=1,
                    quiet_window_s=1,
                    deadline_s=5,
                    poll_s=0.05,
                    gpu_query=lambda: {},
                )

        self.assertEqual(result["status"], "failed", result)
        self.assertEqual(result["failure_classification"], "kill_required")
        self.assertFalse(result["term_only"])
        self.assertEqual(result["quiet_window_anchor_phase"], "term_grace")
        self.assertAlmostEqual(result["quiet_window_anchor_monotonic_s"], 1.1)
        self.assertAlmostEqual(result["first_closed_scan_completed_elapsed_s"], 1.1)
        self.assertIsNone(result["term_closure_completed_elapsed_s"])

    def test_term_scan_crossing_grace_with_residue_starts_containment(self) -> None:
        class FakeClock:
            value = 0.0

            def monotonic(self) -> float:
                return self.value

            def sleep(self, seconds: float) -> None:
                self.value += seconds

        clock = FakeClock()
        target = {
            "pid": 42,
            "identity": "42:100",
            "start_time": 100,
            "pgid": 42,
            "sid": 42,
            "owner_marker": True,
        }
        observer = {"pid": 7, "pgid": 7, "sid": 7}
        calls = 0
        kill_times: list[float] = []

        def scan(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                clock.value += 1.1
            closed = calls >= 3
            return {
                "monotonic_s": clock.value,
                "processes": [] if closed else [target],
                "listeners": {"inodes": []},
                "gpu_processes": {},
                "unattributed_gpu_processes": {},
                "closed": closed,
            }

        process = mock.Mock(pid=42, returncode=0)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "server.log").write_bytes(b"")
            with (
                mock.patch.object(PROBE, "process_identity", return_value=target),
                mock.patch.object(PROBE, "observer_identity", return_value=observer),
                mock.patch.object(PROBE, "scan_lifecycle", side_effect=scan),
                mock.patch.object(
                    PROBE,
                    "_proc_stat",
                    return_value={"start_time": 100, "pgid": 42},
                ),
                mock.patch.object(PROBE.os, "kill"),
                mock.patch.object(
                    PROBE.os,
                    "killpg",
                    side_effect=lambda *args: kill_times.append(clock.value),
                ),
                mock.patch.object(PROBE.time, "monotonic", side_effect=clock.monotonic),
                mock.patch.object(PROBE.time, "sleep", side_effect=clock.sleep),
            ):
                result = PROBE.qualify_lifecycle(
                    process,
                    mock.Mock(),
                    output,
                    port=30000,
                    term_grace_s=1,
                    kill_wait_s=1,
                    quiet_window_s=1,
                    deadline_s=5,
                    poll_s=0.05,
                    gpu_query=lambda: {},
                )

        self.assertEqual(kill_times, [1.1])
        self.assertEqual(result["status"], "failed", result)
        self.assertEqual(result["failure_classification"], "kill_required")
        self.assertEqual(result["quiet_window_anchor_phase"], "kill_containment")
        self.assertAlmostEqual(result["quiet_window_anchor_monotonic_s"], 1.1)
        self.assertAlmostEqual(result["first_closed_scan_completed_elapsed_s"], 1.1)

    def test_post_term_scan_does_not_admit_reused_numeric_group(self) -> None:
        target = {"identity": "42:100", "pid": 42, "pgid": 42, "sid": 42}
        unrelated = {
            "pid": 77,
            "identity": "77:900",
            "state": "S",
            "ppid": 1,
            "pgid": 42,
            "sid": 42,
            "start_time": 900,
            "cgroup": ["0::/"],
            "command": "unrelated",
            "command_bytes_b64": "",
            "command_bytes_sha256": "",
            "environment_inspected": True,
            "owner_marker": False,
        }
        with (
            mock.patch.object(
                PROBE,
                "_all_process_stats",
                return_value={77: {key: unrelated[key] for key in ("state", "ppid", "pgid", "sid", "start_time")}},
            ),
            mock.patch.object(PROBE, "_same_user_pids", return_value=[77]),
            mock.patch.object(PROBE, "process_identity", return_value=unrelated),
            mock.patch.object(PROBE, "_listening_inodes", return_value=set()),
        ):
            result = PROBE.scan_lifecycle(
                target,
                set(),
                port=30000,
                gpu_query=lambda: {},
                admit_numeric_group=False,
            )

        self.assertEqual(result["processes"], [])
        self.assertTrue(result["closed"])

    def test_reparented_recorded_survivor_remains_owned(self) -> None:
        target = {"identity": "42:100", "pid": 42, "pgid": 42, "sid": 42}
        survivor = {
            "pid": 84,
            "identity": "84:200",
            "state": "S",
            "ppid": 1,
            "pgid": 999,
            "sid": 999,
            "start_time": 200,
            "cgroup": ["0::/"],
            "command": "escaped-child",
            "command_bytes_b64": "",
            "command_bytes_sha256": "",
            "environment_inspected": True,
            "owner_marker": False,
        }
        with (
            mock.patch.object(
                PROBE,
                "_all_process_stats",
                return_value={84: {key: survivor[key] for key in ("state", "ppid", "pgid", "sid", "start_time")}},
            ),
            mock.patch.object(PROBE, "_same_user_pids", return_value=[84]),
            mock.patch.object(PROBE, "process_identity", return_value=survivor),
            mock.patch.object(PROBE, "_listening_inodes", return_value=set()),
        ):
            result = PROBE.scan_lifecycle(
                target,
                {"84:200"},
                port=30000,
                gpu_query=lambda: {},
                admit_numeric_group=False,
            )

        self.assertEqual([item["identity"] for item in result["processes"]], ["84:200"])
        self.assertIn("recorded_identity", result["processes"][0]["ownership_reasons"])

    def test_group_generation_requires_current_preterm_identity(self) -> None:
        target = {"pid": 42, "identity": "42:100", "pgid": 42, "sid": 42}
        trusted = {"42:100", "84:200"}
        unrelated_group = [{"pid": 77, "identity": "77:900", "pgid": 42}]
        self.assertFalse(
            PROBE.owned_group_generation_alive(target, unrelated_group, trusted)
        )

        trusted_member = [{"pid": 42, "identity": "42:100", "pgid": 42}]
        with mock.patch.object(
            PROBE,
            "_proc_stat",
            return_value={"start_time": 100, "pgid": 42},
        ):
            self.assertTrue(
                PROBE.owned_group_generation_alive(target, trusted_member, trusted)
            )
        with mock.patch.object(
            PROBE,
            "_proc_stat",
            return_value={"start_time": 101, "pgid": 42},
        ):
            self.assertFalse(
                PROBE.owned_group_generation_alive(target, trusted_member, trusted)
            )

    def test_emergency_cleanup_does_not_signal_reused_launch_pid(self) -> None:
        launch_target = {
            "pid": 42,
            "identity": "42:100",
            "pgid": 42,
            "sid": 42,
        }
        process = mock.Mock()
        process.poll.return_value = None
        with (
            mock.patch.object(
                PROBE,
                "_proc_stat",
                return_value={"start_time": 101, "pgid": 42},
            ),
            mock.patch.object(PROBE.os, "killpg") as killpg,
        ):
            signals = PROBE.emergency_contain_process_group(process, launch_target)

        killpg.assert_not_called()
        process.wait.assert_not_called()
        self.assertEqual(signals[0]["sent"], False)
        self.assertEqual(signals[0]["result"], "not_sent_launch_identity_unverified")

    def test_timeout_containment_individually_signals_unproven_group_member(
        self,
    ) -> None:
        target = {"pid": 42, "identity": "42:100", "pgid": 42, "sid": 42}
        survivor = {
            "pid": 84,
            "identity": "84:200",
            "pgid": 42,
            "sid": 42,
        }
        with (
            mock.patch.object(
                PROBE,
                "_proc_stat",
                return_value={"start_time": 200, "pgid": 42},
            ),
            mock.patch.object(PROBE.os, "killpg") as killpg,
            mock.patch.object(PROBE.os, "kill") as kill_pid,
        ):
            signals = PROBE.contain_recorded_survivors(
                target,
                [survivor],
                {"42:100"},
            )

        killpg.assert_not_called()
        kill_pid.assert_called_once_with(84, PROBE.signal.SIGKILL)
        self.assertEqual(signals[0]["sent"], False)
        self.assertEqual(signals[1]["sent"], True)
        self.assertEqual(signals[1]["target"], "pid:84")

    def test_launch_binding_enforces_remote_session_deadline(self) -> None:
        launch = json.loads(LAUNCH_PATH.read_text())
        self.assertEqual(launch["kaggle_cli_version"], "2.2.4")
        self.assertEqual(launch["session_timeout_seconds"], 1800)
        self.assertEqual(launch["maximum_pushes"], 1)
        self.assertEqual(launch["automatic_retries"], 0)
        self.assertEqual(
            launch["argv"],
            [
                "uvx",
                "--from",
                "kaggle==2.2.4",
                "kaggle",
                "kernels",
                "push",
                "--path",
                "deployment/kaggle/r8l3b-notebook",
                "--timeout",
                "1800",
                "--accelerator",
                "NvidiaRtxPro6000",
            ],
        )
        self.assertEqual(
            launch["request_projection"],
            {
                "session_timeout_seconds": 1800,
                "machine_shape": "NvidiaRtxPro6000",
            },
        )
        self.assertEqual(
            launch["candidate_binding"],
            {
                "notebook_path": "deployment/kaggle/r8l3b-notebook/r8l3b_probe.ipynb",
                "notebook_sha256": hashlib.sha256(NOTEBOOK_PATH.read_bytes()).hexdigest(),
                "metadata_path": "deployment/kaggle/r8l3b-notebook/kernel-metadata.json",
                "metadata_sha256": hashlib.sha256(METADATA_PATH.read_bytes()).hexdigest(),
                "remote_kernel_id": (
                    "pranavagrawai/prototype-qwen3-8-fp8-r8-b-request-path"
                ),
                "expected_predecessor_version": 13,
                "expected_new_version": 14,
                "patch_dataset_source": (
                    "pranavagrawai/sglang-r8l3b-orderly-shutdown-runtime"
                ),
                "expected_patch_dataset_version": 1,
                "patch_dataset_manifest_sha256": hashlib.sha256(
                    DATASET_MANIFEST_PATH.read_bytes()
                ).hexdigest(),
                "patch_dataset_remote_wheel_name": REMOTE_WHEEL_NAME,
                "patch_dataset_readback_path": (
                    "analysis/deployment-probes/20260831T140711Z-r8l3b-dataset-upload-v1/"
                    "REMOTE-READBACK.json"
                ),
                "patch_dataset_readback_sha256": (
                    "cee470f08217da591b8a89d3a80f1f9b4f03aecffced29f69d1b2a5c510fda9d"
                ),
            },
        )

    def test_root_only_term_allows_parent_coordinated_shutdown(self) -> None:
        port = free_port()
        fixture = """
import signal
import socket
import subprocess
import sys
import time

port = int(sys.argv[1])
listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(('127.0.0.1', port))
listener.listen()
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])

def stop(signum, frame):
    print('SIGTERM received. signum=15 frame=None. Draining requests and shutting down...', flush=True)
    try:
        child.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
        print('Gracefully exiting... Remaining number of requests 0', flush=True)
        listener.close()
        raise SystemExit(0)
    print('Subprocess child crashed with exit code -15. Triggering fatal cleanup...', flush=True)
    listener.close()
    raise SystemExit(9)

signal.signal(signal.SIGTERM, stop)
print('READY', flush=True)
while True:
    time.sleep(1)
"""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            log_path = output / "server.log"
            environment = os.environ.copy()
            environment["R8_L3B_OWNER"] = PROBE.R8_L3B_OWNER
            with log_path.open("w") as server_log:
                process = subprocess.Popen(
                    [sys.executable, "-c", fixture, str(port)],
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=environment,
                    start_new_session=True,
                )
                try:
                    wait_for_text(log_path, "READY")
                    target_pgid = os.getpgid(process.pid)

                    def fixture_pids() -> list[int]:
                        visible: list[int] = []
                        for entry in Path("/proc").iterdir():
                            if not entry.name.isdigit():
                                continue
                            pid = int(entry.name)
                            try:
                                if pid == os.getpid() or os.getpgid(pid) == target_pgid:
                                    visible.append(pid)
                            except (ProcessLookupError, PermissionError):
                                continue
                        return visible

                    with mock.patch.object(PROBE, "_same_user_pids", fixture_pids):
                        result = PROBE.qualify_lifecycle(
                            process,
                            server_log,
                            output,
                            port=port,
                            term_grace_s=3,
                            kill_wait_s=1,
                            quiet_window_s=1.2,
                            deadline_s=5,
                            poll_s=0.05,
                            gpu_query=lambda: {},
                        )
                finally:
                    if process.poll() is None:
                        os.killpg(os.getpgid(process.pid), 9)
                        process.wait(timeout=2)

            self.assertEqual(result["status"], "passed", result)
            self.assertTrue(result["term_only"])
            self.assertTrue(result["quiet_window_valid"])
            self.assertEqual(result["root_returncode"], 0)
            signals = json.loads((output / "lifecycle-signals.json").read_text())
            self.assertEqual(signals[0]["target"], f"pid:{process.pid}")
            before = json.loads((output / "lifecycle-owned-before.json").read_text())
            self.assertGreaterEqual(len(before["processes"]), 2)
            self.assertTrue(before["listeners"]["inodes"])
            after = json.loads((output / "lifecycle-owned-after.json").read_text())
            self.assertTrue(after["closed"])
            self.assertEqual(after["processes"], [])

    def test_kill_containment_cannot_pass_lifecycle(self) -> None:
        port = free_port()
        fixture = """
import signal
import socket
import sys
import time
listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(('127.0.0.1', int(sys.argv[1])))
listener.listen()
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print('READY', flush=True)
while True:
    time.sleep(1)
"""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            log_path = output / "server.log"
            environment = os.environ.copy()
            environment["R8_L3B_OWNER"] = PROBE.R8_L3B_OWNER
            with log_path.open("w") as server_log:
                process = subprocess.Popen(
                    [sys.executable, "-c", fixture, str(port)],
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=environment,
                    start_new_session=True,
                )
                try:
                    wait_for_text(log_path, "READY")
                    target_pgid = os.getpgid(process.pid)

                    def fixture_pids() -> list[int]:
                        visible: list[int] = []
                        for entry in Path("/proc").iterdir():
                            if not entry.name.isdigit():
                                continue
                            pid = int(entry.name)
                            try:
                                if pid == os.getpid() or os.getpgid(pid) == target_pgid:
                                    visible.append(pid)
                            except (ProcessLookupError, PermissionError):
                                continue
                        return visible

                    with mock.patch.object(PROBE, "_same_user_pids", fixture_pids):
                        result = PROBE.qualify_lifecycle(
                            process,
                            server_log,
                            output,
                            port=port,
                            term_grace_s=0.3,
                            kill_wait_s=1,
                            quiet_window_s=1.2,
                            deadline_s=3,
                            poll_s=0.05,
                            gpu_query=lambda: {},
                        )
                finally:
                    if process.poll() is None:
                        os.killpg(os.getpgid(process.pid), 9)
                        process.wait(timeout=2)

            self.assertEqual(result["status"], "failed", result)
            self.assertEqual(result["failure_classification"], "kill_required")
            self.assertFalse(result["term_only"])
            self.assertTrue(result["containment_passed"])
            signals = json.loads((output / "lifecycle-signals.json").read_text())
            self.assertEqual([item["signal"] for item in signals], ["SIGTERM", "SIGKILL"])

    def test_r8b2_request_path_functions_are_ast_identical(self) -> None:
        baseline_path = PROBE_PATH.with_name("PROTOTYPE-qwen3_8_phase_a_probe.py")
        baseline = ast.parse(baseline_path.read_text())
        candidate = ast.parse(PROBE_PATH.read_text())
        names = {
            "arc_like_png",
            "build_request",
            "sglang_command",
            "wait_for_server",
            "wait_for_sglang_warmup",
            "wait_for_chat_completion_count",
            "backend_facts",
            "validate_response",
            "post_bytes",
        }

        def functions(tree: ast.AST) -> dict[str, str]:
            return {
                node.name: ast.dump(node, include_attributes=False)
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in names
            }

        baseline_functions = functions(baseline)
        candidate_functions = functions(candidate)
        self.assertEqual(set(baseline_functions), names)
        self.assertEqual(candidate_functions, baseline_functions)

        def top_level_definitions(tree: ast.Module) -> dict[str, str]:
            return {
                node.name: ast.dump(node, include_attributes=False)
                for node in tree.body
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
            }

        baseline_definitions = top_level_definitions(baseline)
        candidate_definitions = top_level_definitions(candidate)
        changed_shared = {
            name
            for name in baseline_definitions.keys() & candidate_definitions.keys()
            if baseline_definitions[name] != candidate_definitions[name]
        }
        self.assertEqual(changed_shared, {"main", "parse_args", "run_probe"})

        image = PROBE.arc_like_png()
        request = PROBE.build_request(PROBE.EXPECTED_SERVED_MODEL_ID, image)
        request_bytes = json.dumps(request).encode("utf-8")
        self.assertEqual(len(request_bytes), PROBE.EXPECTED_REQUEST_BYTES)
        self.assertEqual(
            hashlib.sha256(request_bytes).hexdigest(), PROBE.EXPECTED_REQUEST_SHA256
        )

    def test_unclassified_warning_is_forbidden(self) -> None:
        value = (
            b"SIGTERM received. signum=None frame=None. Draining requests and shutting down...\n"
            b"Warning: destroy_process_group() was not called before program exit\n"
        )
        result = PROBE.classify_teardown_log(value)
        self.assertFalse(result["allowed"])
        self.assertEqual(result["forbidden_count"], 1)
        self.assertEqual(
            result["entries"][-1]["class"], "forbidden_unclassified_log_line"
        )

    def test_request_failure_cannot_be_overwritten_by_clean_lifecycle(self) -> None:
        request = {
            "status": "failed",
            "failure_classification": "request_serving_failure",
        }
        lifecycle = {
            "status": "passed",
            "failure_classification": None,
            "log_classification_allowed": True,
        }

        with tempfile.TemporaryDirectory() as temporary:
            result = PROBE.write_r8l3b_result(Path(temporary), request, lifecycle)
            persisted = json.loads((Path(temporary) / "result.json").read_text())

        self.assertEqual(result["status"], "failed")
        self.assertEqual(persisted, result)
        self.assertEqual(persisted["status"], "failed")
        self.assertEqual(result["request_path_status"], "failed")
        self.assertEqual(result["failure_classification"], "request_serving_failure")
        self.assertFalse(result["r8_l3b_gate"]["request_path_passed"])
        self.assertTrue(result["r8_l3b_gate"]["lifecycle_passed"])

        request_passed = {"status": "passed"}
        lifecycle_without_log_acceptance = {
            "status": "passed",
            "failure_classification": None,
            "log_classification_allowed": False,
        }
        result = PROBE.combine_r8l3b_result(
            request_passed, lifecycle_without_log_acceptance
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            result["failure_classification"], "teardown_log_unacceptable"
        )

        result = PROBE.combine_r8l3b_result(
            {"status": "passed"},
            {
                "status": "passed",
                "failure_classification": None,
                "log_classification_allowed": True,
            },
        )
        self.assertEqual(result["status"], "passed")
        self.assertTrue(all(result["r8_l3b_gate"].values()))

    def test_unattributed_gpu_process_prevents_closure(self) -> None:
        target = {
            "identity": "41:100",
            "pid": 41,
            "pgid": 41,
            "sid": 41,
        }
        with (
            mock.patch.object(PROBE, "_all_process_stats", return_value={}),
            mock.patch.object(PROBE, "_same_user_pids", return_value=[]),
            mock.patch.object(PROBE, "_listening_inodes", return_value=set()),
        ):
            result = PROBE.scan_lifecycle(
                target,
                {"41:100"},
                port=30000,
                gpu_query=lambda: {41: 256},
            )

        self.assertFalse(result["closed"])
        self.assertEqual(result["gpu_processes"], {})
        self.assertEqual(result["unattributed_gpu_processes"], {"41": 256})

    def test_allowed_prefix_cannot_hide_forbidden_suffix(self) -> None:
        value = (
            b"SIGTERM received. signum=None frame=None. Draining requests and shutting down... "
            b"ERROR: destroy_process_group() was not called\n"
        )
        result = PROBE.classify_teardown_log(value)
        self.assertFalse(result["allowed"])
        self.assertEqual(result["forbidden_count"], 1)
        self.assertEqual(
            result["entries"][0]["class"], "forbidden_unclassified_log_line"
        )

    def test_timestamp_prefix_uses_exact_grammar(self) -> None:
        valid = (
            b"[2026-08-30 08:43:01] SIGTERM received. signum=None frame=None. "
            b"Draining requests and shutting down...\n"
        )
        valid_result = PROBE.classify_teardown_log(valid)
        self.assertTrue(valid_result["allowed"], valid_result)

        rejected = (
            (
                b"[destroy_process_group() was not called] SIGTERM received. "
                b"signum=None frame=None. Draining requests and shutting down...\n"
            ),
            (
                b"[2026-13-99 25:61:61] SIGTERM received. signum=None frame=None. "
                b"Draining requests and shutting down...\n"
            ),
        )
        for value in rejected:
            with self.subTest(value=value):
                result = PROBE.classify_teardown_log(value)
                self.assertFalse(result["allowed"])
                self.assertEqual(
                    result["entries"][0]["rule"],
                    "default_deny_unmatched_nonempty_line",
                )

    def test_unknown_fatal_lines_fail_closed(self) -> None:
        mutants = (
            b"FATAL: worker died unexpectedly\n",
            b"Killed\n",
            b"Out of memory: process terminated\n",
            b"child process 123 exited with signal 9\n",
            b"worker exited with status 2\n",
        )
        for value in mutants:
            with self.subTest(value=value):
                result = PROBE.classify_teardown_log(value)
                self.assertFalse(result["allowed"])
                self.assertEqual(result["forbidden_count"], 1)
                self.assertEqual(
                    result["entries"][0]["rule"],
                    "default_deny_unmatched_nonempty_line",
                )

    def test_every_traceback_and_exception_block_is_forbidden(self) -> None:
        exact_previous_pair = b"""ERROR:    Traceback (most recent call last):
  File \"/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/tokenizer_manager.py\", line 3595, in print_exception_wrapper
    await func()
  File \"/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/tokenizer_manager.py\", line 3170, in sigterm_watchdog
    sys.exit(0)
SystemExit: 0
During handling of the above exception, another exception occurred:
Traceback (most recent call last):
  File \"/usr/local/lib/python3.12/dist-packages/starlette/routing.py\", line 655, in lifespan
  File \"/usr/local/lib/python3.12/dist-packages/uvicorn/lifespan/on.py\", line 137, in receive
asyncio.exceptions.CancelledError
"""
        mutants = (
            exact_previous_pair,
            exact_previous_pair.replace(
                b"SystemExit: 0\n",
                b"SystemExit: 0\nattacker-controlled line: exfiltrating /kaggle/working now\n",
            ),
            exact_previous_pair.replace(
                b"SystemExit: 0\n", b"SystemExit: 1\nSystemExit: 0\n"
            ),
            exact_previous_pair.replace(
                b"SystemExit: 0\n", b"KeyboardInterrupt\nSystemExit: 0\n"
            ),
        )
        for value in mutants:
            with self.subTest(value=value):
                result = PROBE.classify_teardown_log(value)
                self.assertFalse(result["allowed"])
                self.assertGreater(result["forbidden_count"], 0)


if __name__ == "__main__":
    unittest.main()
