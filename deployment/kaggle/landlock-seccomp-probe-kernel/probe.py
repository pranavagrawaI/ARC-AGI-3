from __future__ import annotations

import ctypes
import errno
import json
import os
import platform
import resource
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

WORKING = Path(os.environ.get("TYCHO_PROBE_WORKING", "/kaggle/working"))
OUTPUT = WORKING / "result.json"
ROOT = WORKING / "landlock-seccomp-probe"
SANDBOX = ROOT / "sandbox"
OUTSIDE = ROOT / "outside-sentinel.txt"
PROBE = Path(__file__).resolve()

PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2
SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_ERRNO = 0x00050000
AUDIT_ARCH_X86_64 = 0xC000003E

LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1
ACCESS_EXECUTE = 1 << 0
ACCESS_WRITE_FILE = 1 << 1
ACCESS_READ_FILE = 1 << 2
ACCESS_READ_DIR = 1 << 3
ACCESS_REMOVE_DIR = 1 << 4
ACCESS_REMOVE_FILE = 1 << 5
ACCESS_MAKE_CHAR = 1 << 6
ACCESS_MAKE_DIR = 1 << 7
ACCESS_MAKE_REG = 1 << 8
ACCESS_MAKE_SOCK = 1 << 9
ACCESS_MAKE_FIFO = 1 << 10
ACCESS_MAKE_BLOCK = 1 << 11
ACCESS_MAKE_SYM = 1 << 12
ACCESS_REFER = 1 << 13
ACCESS_TRUNCATE = 1 << 14
ACCESS_IOCTL_DEV = 1 << 15
ACCESS_TCP_BIND = 1 << 0
ACCESS_TCP_CONNECT = 1 << 1

LIBC = ctypes.CDLL(None, use_errno=True)


class SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]


class SockFprog(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ushort), ("filters", ctypes.POINTER(SockFilter))]


def syscall_number(name: str) -> int:
    numbers = {"landlock_create_ruleset": 444, "landlock_add_rule": 445, "landlock_restrict_self": 446}
    if platform.machine() != "x86_64":
        raise RuntimeError(f"unsupported architecture: {platform.machine()}")
    return numbers[name]


def checked_syscall(number: int, *args: Any) -> int:
    value = LIBC.syscall(number, *args)
    if value < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return int(value)


def landlock_abi() -> int:
    return checked_syscall(
        syscall_number("landlock_create_ruleset"),
        ctypes.c_void_p(), ctypes.c_size_t(0), ctypes.c_uint(LANDLOCK_CREATE_RULESET_VERSION),
    )


def filesystem_rights(abi: int) -> int:
    rights = (1 << 13) - 1
    if abi >= 2:
        rights |= ACCESS_REFER
    if abi >= 3:
        rights |= ACCESS_TRUNCATE
    if abi >= 5:
        rights |= ACCESS_IOCTL_DEV
    return rights


def install_landlock(path: Path) -> int:
    abi = landlock_abi()
    fs_rights = filesystem_rights(abi)
    network_rights = ACCESS_TCP_BIND | ACCESS_TCP_CONNECT if abi >= 4 else 0
    ruleset_attr = struct.pack("QQ", fs_rights, network_rights)
    ruleset_buffer = ctypes.create_string_buffer(ruleset_attr)
    ruleset_fd = checked_syscall(
        syscall_number("landlock_create_ruleset"),
        ctypes.byref(ruleset_buffer), ctypes.c_size_t(len(ruleset_attr)), ctypes.c_uint(0),
    )
    try:
        path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        try:
            # The workspace is readable and writable, but never executable.
            allowed = fs_rights & ~ACCESS_EXECUTE & ~ACCESS_IOCTL_DEV
            path_attr = ctypes.create_string_buffer(struct.pack("Q", allowed) + struct.pack("i", path_fd) + b"\0" * 4)
            checked_syscall(
                syscall_number("landlock_add_rule"),
                ctypes.c_int(ruleset_fd), ctypes.c_int(LANDLOCK_RULE_PATH_BENEATH),
                ctypes.byref(path_attr), ctypes.c_uint(0),
            )
        finally:
            os.close(path_fd)
        if LIBC.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        checked_syscall(syscall_number("landlock_restrict_self"), ctypes.c_int(ruleset_fd), ctypes.c_uint(0))
    finally:
        os.close(ruleset_fd)
    return abi


def install_seccomp() -> list[int]:
    # Default allow is intentional for this feasibility probe. These are the
    # authority-expanding families that model-authored Python must not reach.
    denied = sorted({
        41, 42, 43, 44, 45, 46, 47, 49, 50, 53, 56, 57, 58, 59, 62, 101,
        155, 161, 165, 166, 169, 170, 171, 175, 176, 200, 234, 246, 248, 249,
        250, 272, 288, 298, 299, 300, 303, 304, 308, 310, 311, 317, 321, 322,
        323, 424, 425, 426, 427, 435, 438,
    })
    instructions: list[tuple[int, int, int, int]] = [
        (0x20, 0, 0, 4),                       # load seccomp_data.arch
        (0x15, 1, 0, AUDIT_ARCH_X86_64),      # matching arch skips kill
        (0x06, 0, 0, 0x80000000),             # kill wrong architecture
        (0x20, 0, 0, 0),                       # load syscall number
    ]
    for number in denied:
        instructions.append((0x15, 0, 1, number))
        instructions.append((0x06, 0, 0, SECCOMP_RET_ERRNO | errno.EPERM))
    instructions.append((0x06, 0, 0, SECCOMP_RET_ALLOW))
    array_type = SockFilter * len(instructions)
    filters = array_type(*(SockFilter(*instruction) for instruction in instructions))
    program = SockFprog(len(instructions), filters)
    if LIBC.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    if LIBC.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(program), 0, 0) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return denied


def current_vms() -> int:
    pages = int(Path("/proc/self/statm").read_text().split()[0])
    return pages * os.sysconf("SC_PAGE_SIZE")


def install_limits() -> dict[str, list[int]]:
    vms = current_vms()
    limits = {
        "RLIMIT_AS": (vms + 64 * 1024 * 1024, vms + 64 * 1024 * 1024),
        "RLIMIT_CORE": (0, 0),
        "RLIMIT_CPU": (1, 2),
        "RLIMIT_FSIZE": (65536, 65536),
        "RLIMIT_NOFILE": (32, 32),
    }
    for name, value in limits.items():
        resource.setrlimit(getattr(resource, name), value)
    return {name: list(resource.getrlimit(getattr(resource, name))) for name in limits}


def attempt(operation: Any) -> dict[str, Any]:
    try:
        value = operation()
        return {"allowed": True, "value": value}
    except BaseException as error:
        return {"allowed": False, "error": type(error).__name__, "errno": getattr(error, "errno", None), "message": str(error)}


def boundary_child(seccomp: bool) -> dict[str, Any]:
    os.chdir(SANDBOX)
    limits = install_limits()
    abi = install_landlock(SANDBOX)
    denied = install_seccomp() if seccomp else []
    allowed_file = SANDBOX / "allowed.txt"
    checks: dict[str, Any] = {}
    checks["allowed_write"] = attempt(lambda: allowed_file.write_text("inside\n"))
    checks["allowed_read"] = attempt(lambda: allowed_file.read_text())
    checks["outside_read"] = attempt(lambda: OUTSIDE.read_text())
    checks["outside_write"] = attempt(lambda: OUTSIDE.write_text("mutated\n"))
    checks["proc_environment"] = attempt(lambda: Path("/proc/self/environ").read_bytes()[:16].hex())
    checks["execute"] = attempt(lambda: subprocess.run(["/bin/true"], check=False).returncode)
    checks["tcp_bind"] = attempt(lambda: _socket_bind())
    checks["tcp_connect"] = attempt(lambda: _socket_connect())
    if seccomp:
        checks["fork"] = attempt(lambda: _fork_once())
        checks["socket_create"] = attempt(lambda: socket.socket().close())
        checks["unshare"] = attempt(lambda: checked_syscall(272, ctypes.c_int(0x10000000)))
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        checks["file_size"] = attempt(lambda: (SANDBOX / "oversize.bin").write_bytes(b"x" * 131072))
        opened = []
        try:
            while True:
                opened.append(os.open(allowed_file, os.O_RDONLY))
        except OSError as error:
            checks["fd_limit"] = {"opened": len(opened), "errno": error.errno, "error": type(error).__name__}
        finally:
            for fd in opened:
                os.close(fd)
    return {"landlock_abi": abi, "seccomp": seccomp, "denied_syscalls": denied, "limits": limits, "checks": checks}


def _socket_bind() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _socket_connect() -> int:
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", 9))


def _fork_once() -> int:
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    return pid


def resource_child(mode: str) -> dict[str, Any]:
    os.chdir(SANDBOX)
    limits = install_limits()
    abi = install_landlock(SANDBOX)
    install_seccomp()
    if mode == "memory":
        chunks = []
        try:
            while True:
                chunks.append(bytearray(8 * 1024 * 1024))
        except MemoryError:
            return {"landlock_abi": abi, "limits": limits, "memory_error": True, "allocated_bytes": len(chunks) * 8 * 1024 * 1024}
    if mode == "cpu":
        while True:
            pass
    raise AssertionError(mode)


def run_child(mode: str, timeout: int = 8) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(PROBE), "--child", mode],
        cwd=SANDBOX, env={"PATH": "/usr/bin", "PYTHONDONTWRITEBYTECODE": "1", "TYCHO_PROBE_SECRET": "must-not-be-readable", "TYCHO_PROBE_WORKING": str(WORKING)},
        capture_output=True, text=True, timeout=timeout, check=False, close_fds=True,
    )
    parsed = None
    if completed.stdout.strip():
        try:
            parsed = json.loads(completed.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            pass
    return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "result": parsed}


def classify(report: dict[str, Any]) -> tuple[str, list[str]]:
    failures: list[str] = []
    landlock = report["landlock_only"].get("result") or {}
    full = report["full_boundary"].get("result") or {}
    if report["landlock_only"]["returncode"] != 0:
        failures.append("landlock child failed")
    if report["full_boundary"]["returncode"] != 0:
        failures.append("full boundary child failed")
    for name in ("outside_read", "outside_write", "proc_environment", "execute"):
        if landlock.get("checks", {}).get(name, {}).get("allowed") is not False:
            failures.append(f"landlock did not deny {name}")
    if landlock.get("landlock_abi", 0) >= 4:
        for name in ("tcp_bind", "tcp_connect"):
            if landlock.get("checks", {}).get(name, {}).get("allowed") is not False:
                failures.append(f"Landlock ABI >=4 did not deny {name}")
    for name in ("allowed_write", "allowed_read"):
        if full.get("checks", {}).get(name, {}).get("allowed") is not True:
            failures.append(f"full boundary denied {name}")
    for name in ("outside_read", "outside_write", "proc_environment", "execute", "tcp_bind", "tcp_connect", "fork", "socket_create", "unshare", "file_size"):
        if full.get("checks", {}).get(name, {}).get("allowed") is not False:
            failures.append(f"full boundary did not deny {name}")
    if full.get("checks", {}).get("fd_limit", {}).get("errno") != errno.EMFILE:
        failures.append("fd limit not enforced")
    memory = report["memory"].get("result") or {}
    if report["memory"]["returncode"] != 0 or memory.get("memory_error") is not True:
        failures.append("address-space limit not enforced")
    if report["cpu"]["returncode"] not in (-signal.SIGXCPU, -signal.SIGKILL):
        failures.append("CPU limit did not terminate child")
    if failures:
        return "primitive_or_boundary_failed", failures
    return "landlock_seccomp_rlimits_qualified_diagnostic", failures


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        mode = sys.argv[2]
        result = boundary_child(mode == "full") if mode in {"landlock", "full"} else resource_child(mode)
        print(json.dumps(result, sort_keys=True), flush=True)
        return
    started = time.monotonic()
    SANDBOX.mkdir(parents=True, exist_ok=True)
    OUTSIDE.write_text("outside-sentinel\n")
    report: dict[str, Any] = {
        "schema": "tycho-landlock-seccomp-kaggle-probe-v1",
        "system": {"platform": platform.platform(), "python": sys.version, "executable": sys.executable, "uid": os.getuid(), "gid": os.getgid()},
        "landlock_only": run_child("landlock"),
        "full_boundary": run_child("full"),
        "memory": run_child("memory"),
        "cpu": run_child("cpu"),
    }
    verdict, failures = classify(report)
    report.update({"verdict": verdict, "failures": failures, "elapsed_seconds": round(time.monotonic() - started, 3), "outside_sentinel_after": OUTSIDE.read_text()})
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict, "failures": failures}, sort_keys=True))


if __name__ == "__main__":
    main()
