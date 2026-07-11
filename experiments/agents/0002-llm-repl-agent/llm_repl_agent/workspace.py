"""Fresh, persistent, filesystem-isolated Python workspace for one run."""

from __future__ import annotations

from dataclasses import dataclass
import errno
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import selectors
import shutil
import struct
import subprocess
import tempfile
from typing import Any


WORKER_SOURCE = r'''
import contextlib
import io
import json
import resource
import traceback

resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024, 32 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))

namespace = {"__name__": "__workspace__"}

class CappedBuffer(io.StringIO):
    limit = 100_000

    def write(self, value):
        remaining = self.limit - self.tell()
        if remaining > 0:
            super().write(value[:remaining])
        return len(value)

for line in __import__("sys").stdin:
    try:
        request = json.loads(line)
        stdout = CappedBuffer()
        stderr = CappedBuffer()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(request["code"], "<workspace>", "exec"), namespace, namespace)
        response = {"ok": True, "stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
    except BaseException:
        response = {
            "ok": False,
            "stdout": locals().get("stdout", io.StringIO()).getvalue(),
            "stderr": locals().get("stderr", io.StringIO()).getvalue(),
            "traceback": traceback.format_exc(),
        }
    print(json.dumps(response, ensure_ascii=False), flush=True)
'''


def _network_seccomp_fd() -> int:
    """Return an x86_64 seccomp BPF program that rejects socket creation/use."""

    if platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError("the Python workspace seccomp filter currently requires x86_64")
    # Linux x86_64: socket=41, connect=42, socketpair=53. No sockets are inherited.
    load_syscall = (0x20, 0, 0, 0)
    jump_socket = (0x15, 3, 0, 41)
    jump_connect = (0x15, 2, 0, 42)
    jump_socketpair = (0x15, 1, 0, 53)
    allow = (0x06, 0, 0, 0x7FFF0000)
    deny = (0x06, 0, 0, 0x00050000 | errno.EPERM)
    program = b"".join(
        struct.pack("=HBBI", *instruction)
        for instruction in (
            load_syscall,
            jump_socket,
            jump_connect,
            jump_socketpair,
            allow,
            deny,
        )
    )
    file_descriptor, path = tempfile.mkstemp(prefix="llm-workspace-seccomp-")
    os.unlink(path)
    os.write(file_descriptor, program)
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    return file_descriptor


@dataclass(frozen=True)
class PythonResult:
    ok: bool
    stdout: str
    stderr: str
    traceback: str | None = None
    timed_out: bool = False
    workspace_restarted: bool = False

    def model_text(self, *, limit: int = 20_000) -> str:
        record = {
            "ok": self.ok,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "traceback": self.traceback,
            "timed_out": self.timed_out,
            "workspace_restarted": self.workspace_restarted,
        }
        rendered = json.dumps(record, ensure_ascii=False)
        if len(rendered) <= limit:
            return rendered
        return rendered[:limit] + "...[tool output truncated]"


class PythonWorkspace:
    """Run arbitrary Python with only stdlib, scratch files, and no network or repo."""

    def __init__(self, *, timeout_seconds: float = 20.0) -> None:
        if shutil.which("bwrap") is None:
            raise RuntimeError("bubblewrap (bwrap) is required for the LLM Python workspace")
        self.timeout_seconds = timeout_seconds
        self.root = Path(tempfile.mkdtemp(prefix="arc-llm-workspace-"))
        self._process: subprocess.Popen[str] | None = None
        self._start()

    def _command(self, *, seccomp_fd: int) -> list[str]:
        command = [
            "bwrap",
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--share-net",
            "--unshare-user",
            "--disable-userns",
            "--seccomp",
            str(seccomp_fd),
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/bin",
            "--setenv",
            "HOME",
            "/workspace",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--dir",
            "/usr",
            "--dir",
            "/usr/bin",
            "--dir",
            "/usr/lib",
            "--ro-bind",
            "/usr/bin/python3.12",
            "/usr/bin/python3.12",
            "--ro-bind",
            "/usr/lib/python3.12",
            "/usr/lib/python3.12",
        ]
        for path in ("/lib", "/lib64"):
            if Path(path).exists():
                command.extend(("--ro-bind", path, path))
        command.extend((
            "--dev", "/dev",
            "--proc", "/proc",
            "--tmpfs", "/tmp",
            "--bind", str(self.root), "/workspace",
            "--chdir", "/workspace",
            "/usr/bin/python3.12", "-I", "-S", "-u", "-c", WORKER_SOURCE,
        ))
        return command

    def _start(self) -> None:
        seccomp_fd = _network_seccomp_fd()
        try:
            self._process = subprocess.Popen(
                self._command(seccomp_fd=seccomp_fd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                pass_fds=(seccomp_fd,),
            )
        finally:
            os.close(seccomp_fd)

    def execute(self, code: str) -> PythonResult:
        process = self._require_process()
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(json.dumps({"code": code}, ensure_ascii=False) + "\n")
        process.stdin.flush()
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        ready = selector.select(self.timeout_seconds)
        selector.close()
        if not ready:
            self._stop()
            self._start()
            return PythonResult(
                ok=False,
                stdout="",
                stderr=f"execution exceeded {self.timeout_seconds:.1f} seconds",
                timed_out=True,
                workspace_restarted=True,
            )
        line = process.stdout.readline()
        if not line:
            detail = self._stderr_text(process)
            self._stop()
            self._start()
            return PythonResult(
                ok=False,
                stdout="",
                stderr=f"Python workspace exited unexpectedly: {detail}",
                workspace_restarted=True,
            )
        response = json.loads(line)
        return PythonResult(
            ok=bool(response.get("ok")),
            stdout=str(response.get("stdout", "")),
            stderr=str(response.get("stderr", "")),
            traceback=(
                str(response["traceback"])
                if response.get("traceback") is not None
                else None
            ),
        )

    def snapshot(self, destination: Path) -> None:
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copytree(self.root, destination, symlinks=True)

    def evidence_state(self) -> dict[str, dict[str, Any]]:
        """Capture model-created files without following model-created symlinks."""

        reserved_names = {
            "WORKSPACE.md",
            "current_frame.png",
            "current_observation.json",
            "public_history.jsonl",
        }
        state: dict[str, dict[str, Any]] = {}
        for path in sorted(self.root.rglob("*")):
            relative = path.relative_to(self.root)
            if relative.parts[0] == "observations" or relative.name in reserved_names:
                continue
            key = relative.as_posix()
            if path.is_symlink():
                state[key] = {"kind": "symlink", "target": os.readlink(path)}
                continue
            if not path.is_file():
                continue
            payload = path.read_bytes()
            record: dict[str, Any] = {
                "kind": "file",
                "size": len(payload),
                "sha256": sha256(payload).hexdigest(),
            }
            if len(payload) <= 100_000:
                try:
                    record["text"] = payload.decode("utf-8")
                except UnicodeDecodeError:
                    pass
            state[key] = record
        return state

    def runtime_info(self) -> dict[str, Any]:
        result = self.execute(
            "import json, pkgutil, platform, sys\n"
            "print(json.dumps({\n"
            "    'implementation': platform.python_implementation(),\n"
            "    'version': sys.version,\n"
            "    'executable': sys.executable,\n"
            "    'path': sys.path,\n"
            "    'top_level_modules': sorted(item.name for item in pkgutil.iter_modules()),\n"
            "}))\n"
        )
        if not result.ok:
            raise RuntimeError(f"could not inspect Python workspace runtime: {result.stderr}")
        info = dict(json.loads(result.stdout))
        info.update({
            "bubblewrap_version": subprocess.check_output(
                ("bwrap", "--version"), text=True
            ).strip(),
            "network_policy": "seccomp denies socket, connect, and socketpair",
            "mounted_runtime": ["/usr/bin/python3.12", "/usr/lib/python3.12"],
        })
        return info

    def close(self) -> None:
        self._stop()
        shutil.rmtree(self.root, ignore_errors=True)

    def _require_process(self) -> subprocess.Popen[str]:
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("Python workspace is closed")
        return self._process

    def _stop(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    @staticmethod
    def _stderr_text(process: subprocess.Popen[str]) -> str:
        if process.stderr is None:
            return ""
        return process.stderr.read()[-4000:]

    def __enter__(self) -> "PythonWorkspace":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
