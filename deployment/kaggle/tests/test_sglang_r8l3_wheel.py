from __future__ import annotations

import asyncio
import ast
import base64
import csv
import hashlib
import importlib.util
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_sglang_r8l3_wheel.py"
)
SPEC = importlib.util.spec_from_file_location("build_sglang_r8l3_wheel", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {SCRIPT_PATH}")
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)


TOKENIZER_SOURCE = b'''        # Subprocess liveness watchdog \xe2\x80\x94 set by Engine or http_server after construction
        self._subprocess_watchdog = None

        # Stop the watchdog: child exits are expected during shutdown, not crashes.
        if self._subprocess_watchdog is not None:
            self._subprocess_watchdog.stop()
        # Ask schedulers to release resources in userspace and exit (see
        # ShutdownReq), then wait for them before hard-killing the rest.
        self._dispatch_to_scheduler(ShutdownReq())
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and collect_scheduler_processes():
            time.sleep(0.1)
        kill_process_tree(os.getpid(), include_parent=False, wait_timeout=60)
        sys.exit(0)
'''

SCHEDULER_SOURCE = b'''from sglang.srt.distributed.parallel_state import get_tp_group

    scheduler = None
    try:
        scheduler.run_event_loop()

    except Exception:
        pass
    finally:
        if scheduler is not None:
            # FPM has a background ZMQ publisher thread that needs explicit
            # teardown to flush queued metrics and close the socket cleanly.
            scheduler.metrics_reporter._shutdown_fpm()
            # Graceful path only: on the exception path the GPU may be wedged
            # and the synchronize() in destroy() could itself hang.
            if scheduler.gracefully_exit:
                scheduler.release_host_resources()
'''

HTTP_SERVER_SOURCE = b'''            else:
                # Default case, one tokenizer process
                uvicorn.run(
                    app,
                    host=get_serving().host,
                    port=get_serving().port,
                    root_path=server_args.fastapi_root_path,
                    log_level=get_observability().log_level_http
                    or get_observability().log_level,
                    timeout_keep_alive=envs.SGLANG_TIMEOUT_KEEP_ALIVE.get(),
                    loop="uvloop",
                    ssl_keyfile=server_args.ssl_keyfile,
                    ssl_certfile=server_args.ssl_certfile,
                    ssl_ca_certs=server_args.ssl_ca_certs,
                    ssl_keyfile_password=server_args.ssl_keyfile_password,
                )
'''

EXECUTABLE_TOKENIZER_SOURCE = b'''class TokenizerManager:
    def init_running_status(self):
        # Subprocess liveness watchdog \xe2\x80\x94 set by Engine or http_server after construction
        self._subprocess_watchdog = None

    async def sigterm_watchdog(self):
        while not self.gracefully_exit:
            await asyncio.sleep(5)

        # Drain requests
        while True:
            remain_num_req = len(self.rid_to_state)
            remaining_rids = list(self.rid_to_state.keys())

            if self.server_status == ServerStatus.UnHealthy:
                logger.error(
                    "Signal SIGTERM received while health check failed. Force exiting."
                )
                self.dump_requests_before_crash()
                self.force_exit_handler()
                break

            elif get_bool_env_var("SGL_FORCE_SHUTDOWN"):
                logger.error(
                    "Signal SIGTERM received while force shutdown flag set. Force exiting."
                )
                self.force_exit_handler()
                break

            logger.info(
                f"Gracefully exiting... Remaining number of requests {remain_num_req}. Remaining requests {remaining_rids=}."
            )
            if remain_num_req > 0:
                await asyncio.sleep(5)
            else:
                break

        # Stop the watchdog: child exits are expected during shutdown, not crashes.
        if self._subprocess_watchdog is not None:
            self._subprocess_watchdog.stop()
        # Ask schedulers to release resources in userspace and exit (see
        # ShutdownReq), then wait for them before hard-killing the rest.
        self._dispatch_to_scheduler(ShutdownReq())
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and collect_scheduler_processes():
            time.sleep(0.1)
        kill_process_tree(os.getpid(), include_parent=False, wait_timeout=60)
        sys.exit(0)
'''

EXECUTABLE_SCHEDULER_SOURCE = b'''from sglang.srt.distributed.parallel_state import get_tp_group

def run_scheduler_process(scheduler_factory):
    scheduler = None
    try:
        scheduler = scheduler_factory()
        scheduler.run_event_loop()

    except Exception:
        pass
    finally:
        if scheduler is not None:
            scheduler.metrics_reporter._shutdown_fpm()
            # Graceful path only: on the exception path the GPU may be wedged
            # and the synchronize() in destroy() could itself hang.
            if scheduler.gracefully_exit:
                scheduler.release_host_resources()
'''


class ShutdownSourceTreatmentTests(unittest.TestCase):
    def test_treatment_returns_through_bound_server_and_destroys_distributed_state(self):
        patched = PATCHER.patch_runtime_sources(
            {
                PATCHER.TOKENIZER_MANAGER_PATH: TOKENIZER_SOURCE,
                PATCHER.SCHEDULER_PATH: SCHEDULER_SOURCE,
                PATCHER.HTTP_SERVER_PATH: HTTP_SERVER_SOURCE,
            }
        )

        tokenizer = patched[PATCHER.TOKENIZER_MANAGER_PATH]
        scheduler = patched[PATCHER.SCHEDULER_PATH]
        http_server = patched[PATCHER.HTTP_SERVER_PATH]

        self.assertNotIn(b"sys.exit(0)", tokenizer)
        self.assertIn(b"self._http_server_shutdown_callback()", tokenizer)
        self.assertIn(b"self._http_server_shutdown_callback = None", tokenizer)
        self.assertIn(
            b"""            if scheduler.gracefully_exit and scheduler_loop_returned_normally:
                try:
                    scheduler.release_host_resources()
                finally:
                    cleanup_dist_env_and_memory()
""",
            scheduler,
        )
        self.assertIn(b"scheduler_loop_returned_normally = False", scheduler)
        self.assertIn(b"scheduler_loop_returned_normally = True", scheduler)
        self.assertIn(
            b"if scheduler.gracefully_exit and scheduler_loop_returned_normally:",
            scheduler,
        )
        self.assertIn(b"server = uvicorn.Server(config)", http_server)
        self.assertIn(
            b"tokenizer_manager._http_server_shutdown_callback = request_http_server_shutdown",
            http_server,
        )
        self.assertIn(b"server.run()", http_server)

    def test_original_watchdog_raises_but_treatment_requests_server_exit_and_returns(self):
        files = {
            PATCHER.TOKENIZER_MANAGER_PATH: EXECUTABLE_TOKENIZER_SOURCE,
            PATCHER.SCHEDULER_PATH: SCHEDULER_SOURCE,
            PATCHER.HTTP_SERVER_PATH: HTTP_SERVER_SOURCE,
        }
        patched = PATCHER.patch_runtime_sources(files)[PATCHER.TOKENIZER_MANAGER_PATH]

        def run(source: bytes) -> list[str]:
            events: list[str] = []

            class Status:
                UnHealthy = object()
                Up = object()

            class Logger:
                def info(self, message):
                    events.append("log_info")

                def error(self, message):
                    events.append("log_error")

            class Watchdog:
                def stop(self):
                    events.append("watchdog_stop")

            class Shutdown:
                pass

            namespace = {
                "asyncio": asyncio,
                "collect_scheduler_processes": lambda: [],
                "get_bool_env_var": lambda name: False,
                "kill_process_tree": lambda pid, **kwargs: events.append("kill_children"),
                "logger": Logger(),
                "os": __import__("os"),
                "ServerStatus": Status,
                "ShutdownReq": Shutdown,
                "sys": __import__("sys"),
                "time": __import__("time"),
            }
            exec(compile(source, "tokenizer_manager.py", "exec"), namespace)
            manager = namespace["TokenizerManager"]()
            manager.init_running_status()
            manager.gracefully_exit = True
            manager.rid_to_state = {}
            manager.server_status = Status.Up
            manager._subprocess_watchdog = Watchdog()
            manager._dispatch_to_scheduler = lambda request: events.append("dispatch_shutdown")
            manager._http_server_shutdown_callback = lambda: events.append("server_exit")
            asyncio.run(manager.sigterm_watchdog())
            return events

        with self.assertRaises(SystemExit):
            run(EXECUTABLE_TOKENIZER_SOURCE)
        self.assertEqual(
            run(patched),
            ["log_info", "watchdog_stop", "dispatch_shutdown", "kill_children", "server_exit"],
        )

    def test_distributed_cleanup_requires_normal_scheduler_loop_return(self):
        patched = PATCHER.patch_runtime_sources(
            {
                PATCHER.TOKENIZER_MANAGER_PATH: TOKENIZER_SOURCE,
                PATCHER.SCHEDULER_PATH: EXECUTABLE_SCHEDULER_SOURCE,
                PATCHER.HTTP_SERVER_PATH: HTTP_SERVER_SOURCE,
            }
        )[PATCHER.SCHEDULER_PATH]
        tree = ast.parse(patched)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_scheduler_process"
        )
        namespace: dict[str, Any] = {}
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), "scheduler.py", "exec"),
            namespace,
        )

        def exercise(*, raises: bool) -> list[str]:
            events: list[str] = []

            class Reporter:
                def _shutdown_fpm(self):
                    events.append("metrics_shutdown")

            class Scheduler:
                gracefully_exit = True
                metrics_reporter = Reporter()

                def run_event_loop(self):
                    events.append("event_loop")
                    if raises:
                        raise RuntimeError("after shutdown request")

                def release_host_resources(self):
                    events.append("host_cleanup")

            namespace["cleanup_dist_env_and_memory"] = lambda: events.append(
                "distributed_cleanup"
            )
            namespace["run_scheduler_process"](Scheduler)
            return events

        self.assertEqual(
            exercise(raises=False),
            ["event_loop", "metrics_shutdown", "host_cleanup", "distributed_cleanup"],
        )
        self.assertEqual(
            exercise(raises=True),
            ["event_loop", "metrics_shutdown"],
        )

    def test_wheel_rewrite_is_deterministic_and_record_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_wheel = root / "source.whl"
            first_wheel = root / "first.whl"
            second_wheel = root / "second.whl"
            record_path = "sglang-0.0.dist-info/RECORD"
            entries = {
                PATCHER.TOKENIZER_MANAGER_PATH: TOKENIZER_SOURCE,
                PATCHER.SCHEDULER_PATH: SCHEDULER_SOURCE,
                PATCHER.HTTP_SERVER_PATH: HTTP_SERVER_SOURCE,
                "sglang-0.0.dist-info/METADATA": b"Metadata-Version: 2.1\nName: sglang\nVersion: 0.0\n",
                "sglang-0.0.dist-info/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
                record_path: b"",
            }
            with zipfile.ZipFile(source_wheel, "w", compression=zipfile.ZIP_STORED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            replacements = PATCHER.patch_runtime_sources(entries)
            PATCHER.rewrite_wheel(source_wheel, first_wheel, replacements)
            PATCHER.rewrite_wheel(source_wheel, second_wheel, replacements)

            self.assertEqual(first_wheel.read_bytes(), second_wheel.read_bytes())
            with zipfile.ZipFile(first_wheel) as archive:
                record_rows = list(
                    csv.reader(io.StringIO(archive.read(record_path).decode("utf-8")))
                )
                by_name = {row[0]: row[1:] for row in record_rows}
                self.assertEqual(set(by_name), set(entries))
                for name in entries:
                    if name == record_path:
                        self.assertEqual(by_name[name], ["", ""])
                        continue
                    data = archive.read(name)
                    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(
                        b"="
                    )
                    self.assertEqual(by_name[name], [f"sha256={digest.decode()}", str(len(data))])

    def test_production_builder_rejects_any_other_upstream_wheel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_wheel = root / "untrusted.whl"
            output_wheel = root / "patched.whl"
            with zipfile.ZipFile(source_wheel, "w") as archive:
                archive.writestr("sglang-0.0.dist-info/RECORD", b"")

            with self.assertRaisesRegex(ValueError, "upstream wheel SHA-256"):
                PATCHER.build_patched_wheel(source_wheel, output_wheel)
            self.assertFalse(output_wheel.exists())


if __name__ == "__main__":
    unittest.main()
