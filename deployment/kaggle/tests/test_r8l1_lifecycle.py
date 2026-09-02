from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

PROBE_PATH = Path(__file__).resolve().parents[1] / "PROTOTYPE-qwen3_8_r8_l1_probe.py"
SPEC = importlib.util.spec_from_file_location("r8l1_probe", PROBE_PATH)
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


class LifecycleObserverTests(unittest.TestCase):
    def test_self_test_and_r8l1_are_mutually_exclusive(self) -> None:
        argv = [
            str(PROBE_PATH),
            "--output-dir",
            "/tmp/r8l1-arg-test",
            "--self-test",
            "--r8-l1",
        ]
        with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            PROBE.parse_args()

    def test_term_only_process_group_closes_cleanly(self) -> None:
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
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
    print('Gracefully exiting... Remaining number of requests 0', flush=True)
    listener.close()
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
print('READY', flush=True)
while True:
    time.sleep(1)
"""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            log_path = output / "server.log"
            environment = os.environ.copy()
            environment["R8_L1_OWNER"] = PROBE.R8_L1_OWNER
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
            environment["R8_L1_OWNER"] = PROBE.R8_L1_OWNER
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
            result = PROBE.write_r8l1_result(Path(temporary), request, lifecycle)
            persisted = json.loads((Path(temporary) / "result.json").read_text())

        self.assertEqual(result["status"], "failed")
        self.assertEqual(persisted, result)
        self.assertEqual(persisted["status"], "failed")
        self.assertEqual(result["request_path_status"], "failed")
        self.assertEqual(result["failure_classification"], "request_serving_failure")
        self.assertFalse(result["r8_l1_gate"]["request_path_passed"])
        self.assertTrue(result["r8_l1_gate"]["lifecycle_passed"])

        request_passed = {"status": "passed"}
        lifecycle_without_log_acceptance = {
            "status": "passed",
            "failure_classification": None,
            "log_classification_allowed": False,
        }
        result = PROBE.combine_r8l1_result(
            request_passed, lifecycle_without_log_acceptance
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            result["failure_classification"], "teardown_log_unacceptable"
        )

        result = PROBE.combine_r8l1_result(
            {"status": "passed"},
            {
                "status": "passed",
                "failure_classification": None,
                "log_classification_allowed": True,
            },
        )
        self.assertEqual(result["status"], "passed")
        self.assertTrue(all(result["r8_l1_gate"].values()))

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
