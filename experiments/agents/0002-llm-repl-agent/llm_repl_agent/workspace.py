"""Fresh, persistent, filesystem-isolated Python workspace for one run."""

from __future__ import annotations

from dataclasses import dataclass
import errno
from hashlib import sha256
from importlib import metadata
from importlib.util import find_spec
from functools import lru_cache
import json
import os
from pathlib import Path
import platform
import re
import selectors
import shutil
import stat
import struct
import subprocess
import sys
import sysconfig
import tempfile
import time
from typing import Any, Mapping
from uuid import uuid4


WORKER_SOURCE = r'''
import contextlib
import io
import json
import resource
import sys
import traceback

sys.path.insert(0, "/packages")

def deny_process_creation(event, args):
    blocked = {
        "ctypes.dlopen",
        "os.exec",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.spawn",
        "os.system",
        "subprocess.Popen",
    }
    if event in blocked:
        raise PermissionError(1, "process creation and native-library loading are blocked")

sys.addaudithook(deny_process_creation)

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
        response = {
            "request_id": request.get("request_id"),
            "ok": True,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
        }
    except BaseException:
        response = {
            "request_id": locals().get("request", {}).get("request_id"),
            "ok": False,
            "stdout": locals().get("stdout", io.StringIO()).getvalue(),
            "stderr": locals().get("stderr", io.StringIO()).getvalue(),
            "traceback": traceback.format_exc(),
        }
    print(json.dumps(response, ensure_ascii=False), flush=True)
'''

MODEL_OUTPUT_CHAR_LIMIT = 4_000

# General computation only. Keep this list deliberately small: packages added here
# become part of the policy's experimental substrate, not an ARC-specific toolkit.
MODEL_RUNTIME_DISTRIBUTIONS = {
    "numpy": ("numpy", "numpy.libs"),
    "pillow": ("PIL", "pillow.libs"),
}

LDD_PATH = re.compile(r"(?:=>\s+)?(/[^\s()]+)")


@lru_cache(maxsize=1)
def _shared_library_mounts() -> tuple[Path, ...]:
    """Resolve native dependencies without mounting the host's whole /lib tree."""

    executable, stdlib, package_mounts, _versions = _model_runtime_mounts()
    candidates = [executable, *stdlib.glob("lib-dynload/*.so")]
    for source in package_mounts.values():
        if source.is_dir():
            candidates.extend(source.rglob("*.so*"))
    package_roots = tuple(path.resolve() for path in package_mounts.values())
    libraries: set[Path] = set()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        completed = subprocess.run(
            ("ldd", str(candidate)),
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode not in {0, 1}:
            raise RuntimeError(f"could not inspect native dependencies for {candidate}")
        for match in LDD_PATH.finditer(completed.stdout + completed.stderr):
            dependency = Path(match.group(1))
            try:
                resolved = dependency.resolve(strict=True)
            except FileNotFoundError:
                continue
            if any(resolved.is_relative_to(root) for root in package_roots):
                continue
            libraries.add(dependency)
    return tuple(sorted(libraries))


def _synthetic_proc_root() -> Path:
    """Create the small non-host-identifying /proc surface admitted to contestants."""

    root = Path(tempfile.mkdtemp(prefix="arc-llm-proc-"))
    (root / "self").mkdir()
    (root / "cpuinfo").write_text("processor\t: 0\n", encoding="utf-8")
    (root / "meminfo").write_text("MemTotal:       1048576 kB\n", encoding="utf-8")
    (root / "version").write_text("Linux sandbox\n", encoding="utf-8")
    (root / "self" / "status").write_text(
        "Name:\tpython3\nState:\tR (running)\nNoNewPrivs:\t1\n",
        encoding="utf-8",
    )
    return root


def _model_runtime_mounts() -> tuple[Path, Path, dict[str, Path], dict[str, str]]:
    """Resolve the interpreter and explicitly allowed packages used by the model."""

    executable = Path(sys.base_prefix) / "bin" / "python3"
    stdlib = Path(sysconfig.get_path("stdlib"))
    package_mounts: dict[str, Path] = {}
    versions: dict[str, str] = {}
    for distribution_name, import_roots in MODEL_RUNTIME_DISTRIBUTIONS.items():
        distribution = metadata.distribution(distribution_name)
        versions[distribution_name] = distribution.version
        top_level_spec = find_spec(import_roots[0])
        if top_level_spec is None or top_level_spec.origin is None:
            raise RuntimeError(
                f"model runtime dependency is not importable: {distribution_name}"
            )
        site_packages = Path(top_level_spec.origin).parent.parent
        for import_root in import_roots:
            source = site_packages / import_root
            if not source.exists():
                raise RuntimeError(
                    f"model runtime dependency is incomplete: {source} does not exist"
                )
            package_mounts[import_root] = source

    if not executable.is_file():
        raise RuntimeError(f"model Python interpreter does not exist: {executable}")
    if not stdlib.is_dir():
        raise RuntimeError(f"model Python standard library does not exist: {stdlib}")
    return executable, stdlib, package_mounts, versions


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

    def model_text(self, *, limit: int = MODEL_OUTPUT_CHAR_LIMIT) -> str:
        def render(value: Mapping[str, Any]) -> str:
            return json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )

        record = {
            "ok": self.ok,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "traceback": self.traceback,
            "timed_out": self.timed_out,
            "workspace_restarted": self.workspace_restarted,
        }
        original = render(record)
        payload = {
            **record,
            "truncated": False,
            "original_chars": len(original),
            "limit": limit,
        }
        rendered = render(payload)
        if len(rendered) <= limit:
            return rendered

        payload["truncated"] = True
        text_fields = (
            ("stdout", "stderr", "traceback")
            if self.ok
            else ("traceback", "stdout", "stderr")
        )
        originals = {
            field: payload[field]
            for field in text_fields
            if isinstance(payload[field], str)
        }
        for field in originals:
            payload[field] = ""
        if len(render(payload)) > limit:
            raise ValueError(f"model output limit {limit} is too small for metadata")

        for field in text_fields:
            value = originals.get(field)
            if value is None:
                continue
            low, high = 0, len(value)
            while low < high:
                middle = (low + high + 1) // 2
                payload[field] = value[:middle]
                if len(render(payload)) <= limit:
                    low = middle
                else:
                    high = middle - 1
            payload[field] = value[:low]
        return render(payload)


class PythonWorkspace:
    """Run arbitrary Python with a small allowlist, scratch files, and no network or repo."""

    def __init__(self, *, timeout_seconds: float = 20.0) -> None:
        if shutil.which("bwrap") is None:
            raise RuntimeError("bubblewrap (bwrap) is required for the LLM Python workspace")
        self.timeout_seconds = timeout_seconds
        (
            self._runtime_executable,
            self._runtime_stdlib,
            self._package_mounts,
            self._package_versions,
        ) = _model_runtime_mounts()
        self.root = Path(tempfile.mkdtemp(prefix="arc-llm-workspace-"))
        self._proc_root = _synthetic_proc_root()
        self._shared_libraries = _shared_library_mounts()
        self._process: subprocess.Popen[bytes] | None = None
        self._launch_command: list[str] = []
        self._stdout_pending = b""
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
            "--tmpfs",
            "/runtime",
            "--dir",
            "/runtime/bin",
            "--dir",
            "/runtime/lib",
            "--tmpfs",
            "/packages",
            "--ro-bind",
            str(self._runtime_executable),
            "/runtime/bin/python3",
            "--ro-bind",
            str(self._runtime_stdlib),
            f"/runtime/lib/python{sys.version_info.major}.{sys.version_info.minor}",
        ]
        for destination, source in sorted(self._package_mounts.items()):
            command.extend(("--ro-bind", str(source), f"/packages/{destination}"))
        library_directories = sorted(
            {str(path.parent) for path in self._shared_libraries},
            key=lambda value: (value.count("/"), value),
        )
        command.extend(("--tmpfs", "/lib", "--tmpfs", "/lib64"))
        created_directories: set[str] = set()
        for directory in library_directories:
            parts = Path(directory).parts
            for depth in range(1, len(parts) + 1):
                parent = str(Path(*parts[:depth]))
                if parent != "/" and parent not in created_directories:
                    command.extend(("--dir", parent))
                    created_directories.add(parent)
        for library in self._shared_libraries:
            command.extend(("--ro-bind", str(library), str(library)))
        command.extend((
            "--remount-ro", "/runtime",
            "--remount-ro", "/packages",
            "--remount-ro", "/lib",
            "--remount-ro", "/lib64",
            "--dev", "/dev",
            "--remount-ro", "/dev",
            "--ro-bind", str(self._proc_root), "/proc",
            "--tmpfs", "/tmp",
            "--bind", str(self.root), "/workspace",
            "--chdir", "/workspace",
            "/runtime/bin/python3", "-I", "-S", "-u", "-c", WORKER_SOURCE,
        ))
        return command

    def _start(self) -> None:
        self._stdout_pending = b""
        seccomp_fd = _network_seccomp_fd()
        command = self._command(seccomp_fd=seccomp_fd)
        self._launch_command = list(command)
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                bufsize=0,
                pass_fds=(seccomp_fd,),
            )
        finally:
            os.close(seccomp_fd)

    def execute(self, code: str) -> PythonResult:
        process = self._require_process()
        assert process.stdin is not None
        assert process.stdout is not None
        request_id = uuid4().hex
        request = json.dumps(
            {"request_id": request_id, "code": code}, ensure_ascii=False
        ).encode("utf-8") + b"\n"
        process.stdin.write(request)
        process.stdin.flush()
        deadline = time.monotonic() + self.timeout_seconds
        response: dict[str, Any] | None = None
        while response is None:
            while b"\n" not in self._stdout_pending:
                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ)
                ready = selector.select(max(0.0, deadline - time.monotonic()))
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
                chunk = os.read(process.stdout.fileno(), 65_536)
                if not chunk:
                    detail = self._stderr_text(process)
                    self._stop()
                    self._start()
                    return PythonResult(
                        ok=False,
                        stdout="",
                        stderr=f"Python workspace exited unexpectedly: {detail}",
                        workspace_restarted=True,
                    )
                self._stdout_pending += chunk
            raw_line, self._stdout_pending = self._stdout_pending.split(b"\n", 1)
            try:
                candidate = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(candidate, dict) and candidate.get("request_id") == request_id:
                response = candidate
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
        self._snapshot_entry(self.root, destination)

    @classmethod
    def _snapshot_entry(cls, source: Path, destination: Path) -> None:
        mode = source.lstat().st_mode
        if stat.S_ISLNK(mode):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(os.readlink(source))
            return
        if stat.S_ISDIR(mode):
            destination.mkdir()
            for child in sorted(source.iterdir(), key=lambda item: item.name):
                cls._snapshot_entry(child, destination / child.name)
            return
        if stat.S_ISREG(mode):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination, follow_symlinks=False)
            return
        kind = (
            "fifo" if stat.S_ISFIFO(mode)
            else "socket" if stat.S_ISSOCK(mode)
            else "block" if stat.S_ISBLK(mode)
            else "character" if stat.S_ISCHR(mode)
            else "unknown"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"kind": "special", "type": kind}, separators=(",", ":")),
            encoding="utf-8",
        )

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
            "process_policy": (
                "Python audit hook denies process creation and native-library loading"
            ),
            "proc_policy": "synthetic allowlist; host procfs is not mounted",
            "mounted_runtime": [
                "/runtime/bin/python3",
                f"/runtime/lib/python{sys.version_info.major}.{sys.version_info.minor}",
            ],
            "allowed_distributions": dict(self._package_versions),
        })
        return info

    def sandbox_evidence(self) -> dict[str, Any]:
        """Return host-only launch and descriptor evidence without credential values."""

        process = self._require_process()
        fd_root = Path(f"/proc/{process.pid}/fd")
        descriptors = []
        for path in sorted(fd_root.iterdir(), key=lambda item: int(item.name)):
            try:
                target = os.readlink(path)
            except FileNotFoundError:
                continue
            descriptors.append({"fd": int(path.name), "target": target})
        return {
            "process_id": process.pid,
            "process_arguments": list(self._launch_command),
            "open_file_descriptors": descriptors,
            "seccomp_filter": {
                "architecture": platform.machine(),
                "delivery": "bubblewrap --seccomp inherited descriptor",
                "denied_syscalls": ["socket", "connect", "socketpair"],
                "deny_errno": errno.EPERM,
                "program_instruction_count": 6,
            },
            "mount_table": {
                "runtime": [
                    "/runtime/bin/python3",
                    f"/runtime/lib/python{sys.version_info.major}.{sys.version_info.minor}",
                ],
                "packages": sorted(f"/packages/{name}" for name in self._package_mounts),
                "shared_libraries": [str(path) for path in self._shared_libraries],
                "writable": ["/workspace", "/tmp"],
                "synthetic": ["/proc", "/dev"],
            },
            "runtime": self.runtime_info(),
        }

    def close(self) -> None:
        self._stop()
        shutil.rmtree(self.root, ignore_errors=True)
        shutil.rmtree(self._proc_root, ignore_errors=True)

    def _require_process(self) -> subprocess.Popen[bytes]:
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
    def _stderr_text(process: subprocess.Popen[bytes]) -> str:
        if process.stderr is None:
            return ""
        return process.stderr.read()[-4000:].decode("utf-8", errors="replace")

    def __enter__(self) -> "PythonWorkspace":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
