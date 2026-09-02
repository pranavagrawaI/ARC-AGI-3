from __future__ import annotations

import ctypes
import errno
import json
import os
import platform
import resource
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

WORKING = Path(os.environ.get("TYCHO_PROBE_WORKING", "/kaggle/working"))
OUTPUT = WORKING / "result.json"
ROOT = WORKING / "host-authority-census"
GAME_A = ROOT / "game-a"
GAME_B = ROOT / "game-b"
HOST = ROOT / "host-control"
INPUT = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3")
PROBE = Path(__file__).resolve()
GAME_A_UID = 60001
GAME_B_UID = 60002

PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2
SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_ERRNO = 0x00050000
AUDIT_ARCH_X86_64 = 0xC000003E
LIBC = ctypes.CDLL(None, use_errno=True)


class SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]


class SockFprog(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ushort), ("filters", ctypes.POINTER(SockFilter))]


def status_fields() -> dict[str, str]:
    wanted = {"CapEff", "CapPrm", "CapBnd", "NoNewPrivs", "Seccomp", "Seccomp_filters", "Uid", "Gid"}
    fields: dict[str, str] = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in wanted:
            fields[key] = value.strip()
    return fields


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


def install_seccomp() -> list[int]:
    if platform.machine() != "x86_64":
        raise RuntimeError(f"unsupported architecture: {platform.machine()}")
    # Default allow is a consciously limited S0 diagnostic policy. Denied
    # families prevent new processes, network I/O, namespace/mount changes,
    # privilege changes, tracing, cross-process memory, and kernel interfaces.
    denied = sorted({
        41, 42, 43, 44, 45, 46, 47, 49, 50, 53, 56, 57, 58, 59, 62, 101,
        105, 106, 113, 114, 116, 117, 119, 126, 155, 157, 161, 165, 166, 169,
        170, 171, 175, 176, 200, 234, 246, 248, 249, 250, 272, 288, 298, 299,
        300, 303, 304, 308, 310, 311, 317, 321, 322, 323, 424, 425, 426, 427,
        435, 438,
    })
    instructions: list[tuple[int, int, int, int]] = [
        (0x20, 0, 0, 4),
        (0x15, 1, 0, AUDIT_ARCH_X86_64),
        (0x06, 0, 0, 0x80000000),
        (0x20, 0, 0, 0),
    ]
    for number in denied:
        instructions.extend(((0x15, 0, 1, number), (0x06, 0, 0, SECCOMP_RET_ERRNO | errno.EPERM)))
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


def attempt(operation: Callable[[], Any]) -> dict[str, Any]:
    try:
        return {"allowed": True, "value": operation()}
    except BaseException as error:
        return {
            "allowed": False,
            "error": type(error).__name__,
            "errno": getattr(error, "errno", None),
            "message": str(error),
        }


def environment_keys(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    keys = sorted({item.split(b"=", 1)[0].decode("utf-8", "replace") for item in raw.split(b"\0") if b"=" in item})
    return {"keys": keys, "probe_marker_visible": b"TYCHO_AUTHORITY_PARENT_MARKER=" in raw}


def input_inventory() -> dict[str, Any]:
    result: dict[str, Any] = {"exists": INPUT.exists(), "readable": os.access(INPUT, os.R_OK), "writable": os.access(INPUT, os.W_OK)}
    if not INPUT.exists():
        return result
    result["top_level_names"] = sorted(path.name for path in INPUT.iterdir())[:40]
    sample = next((path for path in INPUT.rglob("*") if path.is_file()), None)
    result["sample_file"] = None if sample is None else {
        "relative_path": str(sample.relative_to(INPUT)),
        "one_byte_read": attempt(lambda: len(sample.open("rb").read(1))),
        "writable": os.access(sample, os.W_OK),
    }
    return result


def socket_create() -> bool:
    with socket.socket():
        return True


def fork_once() -> int:
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    return pid


def guarded_child(mode: str) -> dict[str, Any]:
    os.chdir(GAME_A)
    limits = install_limits()
    before = status_fields()
    denied = install_seccomp()
    after = status_fields()
    if mode == "memory":
        chunks = []
        try:
            while True:
                chunks.append(bytearray(8 * 1024 * 1024))
        except MemoryError:
            return {"limits": limits, "memory_error": True, "allocated_bytes": len(chunks) * 8 * 1024 * 1024, "status": after}
    if mode == "cpu":
        while True:
            pass
    if mode != "census":
        raise AssertionError(mode)

    ppid = os.getppid()
    checks = {
        "own_read": attempt(lambda: (GAME_A / "canary.txt").read_text()),
        "own_write": attempt(lambda: (GAME_A / f"child-output-{os.getuid()}.txt").write_text("owned\n")),
        "sibling_read": attempt(lambda: (GAME_B / "canary.txt").read_text()),
        "sibling_write": attempt(lambda: (GAME_B / "canary.txt").write_text("mutated-by-child\n")),
        "host_read": attempt(lambda: (HOST / "canary.txt").read_text()),
        "host_write": attempt(lambda: (HOST / "canary.txt").write_text("mutated-by-child\n")),
        "etc_read": attempt(lambda: len(Path("/etc/hostname").read_bytes())),
        "parent_environment": attempt(lambda: environment_keys(Path(f"/proc/{ppid}/environ"))),
        "parent_memory_open": attempt(lambda: os.close(os.open(f"/proc/{ppid}/mem", os.O_RDONLY))),
        "pid1_environment": attempt(lambda: environment_keys(Path("/proc/1/environ"))),
        "process_count": attempt(lambda: len([path for path in Path("/proc").iterdir() if path.name.isdigit()])),
        "socket_create": attempt(socket_create),
        "fork": attempt(fork_once),
        "exec": attempt(lambda: subprocess.run(["/bin/true"], check=False).returncode),
        "regain_uid_zero": attempt(lambda: os.setresuid(0, 0, 0)),
        "regain_gid_zero": attempt(lambda: os.setresgid(0, 0, 0)),
    }
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    checks["file_size"] = attempt(lambda: (GAME_A / f"oversize-{os.getuid()}.bin").write_bytes(b"x" * 131072))
    opened = []
    try:
        while True:
            opened.append(os.open(GAME_A / "canary.txt", os.O_RDONLY))
    except OSError as error:
        checks["fd_limit"] = {"opened": len(opened), "errno": error.errno, "error": type(error).__name__}
    finally:
        for fd in opened:
            os.close(fd)
    return {
        "mode": mode,
        "uid": os.getuid(),
        "gid": os.getgid(),
        "environment_keys": sorted(os.environ),
        "parent_marker_in_direct_environment": "TYCHO_AUTHORITY_PARENT_MARKER" in os.environ,
        "status_before_seccomp": before,
        "status_after_seccomp": after,
        "limits": limits,
        "denied_syscalls": denied,
        "input_inventory": input_inventory(),
        "checks": checks,
    }


def drop_identity(uid: int, gid: int) -> Callable[[], None]:
    def drop() -> None:
        os.setgroups([])
        os.setresgid(gid, gid, gid)
        os.setresuid(uid, uid, uid)
    return drop


def run_guarded(mode: str, uid: int | None, timeout: int = 8) -> dict[str, Any]:
    environment = {"PATH": "/usr/bin", "PYTHONDONTWRITEBYTECODE": "1", "TYCHO_PROBE_WORKING": str(WORKING)}
    completed = subprocess.run(
        [sys.executable, str(PROBE), "--guarded", mode],
        cwd=GAME_A,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        close_fds=True,
        preexec_fn=None if uid is None else drop_identity(uid, uid),
    )
    parsed = None
    if completed.stdout.strip():
        try:
            parsed = json.loads(completed.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            pass
    return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "result": parsed}


def supervisor() -> dict[str, Any]:
    baseline = run_guarded("census", None)
    # The baseline mutations are deliberate evidence. Restore the frozen
    # canaries before the unique-UID treatment so its non-mutation is causal.
    (GAME_B / "canary.txt").write_text("game-b-canary\n")
    (HOST / "canary.txt").write_text("host-canary\n")
    return {
        "same_uid_baseline": baseline,
        "unique_uid_census": run_guarded("census", GAME_A_UID),
        "unique_uid_memory": run_guarded("memory", GAME_A_UID),
        "unique_uid_cpu": run_guarded("cpu", GAME_A_UID),
    }


def prepare() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    os.chmod(ROOT, 0o711)
    for path, uid in ((GAME_A, GAME_A_UID), (GAME_B, GAME_B_UID)):
        path.mkdir(exist_ok=True)
        os.chown(path, uid, uid)
        os.chmod(path, 0o700)
        canary = path / "canary.txt"
        canary.write_text(f"{path.name}-canary\n")
        os.chown(canary, uid, uid)
        os.chmod(canary, 0o600)
    HOST.mkdir(exist_ok=True)
    os.chown(HOST, 0, 0)
    os.chmod(HOST, 0o700)
    (HOST / "canary.txt").write_text("host-canary\n")
    os.chmod(HOST / "canary.txt", 0o600)


def classify(report: dict[str, Any]) -> tuple[str, list[str]]:
    failures: list[str] = []
    baseline = report["supervisor"]["same_uid_baseline"].get("result") or {}
    unique = report["supervisor"]["unique_uid_census"].get("result") or {}
    for label in ("same_uid_baseline", "unique_uid_census"):
        if report["supervisor"][label]["returncode"] != 0:
            failures.append(f"{label} failed")
    if baseline.get("checks", {}).get("sibling_read", {}).get("allowed") is not True:
        failures.append("same-UID baseline did not expose sibling read")
    if baseline.get("checks", {}).get("host_read", {}).get("allowed") is not True:
        failures.append("same-UID baseline did not expose host read")
    for name in ("own_read", "own_write", "etc_read"):
        if unique.get("checks", {}).get(name, {}).get("allowed") is not True:
            failures.append(f"unique UID denied intended {name}")
    for name in ("sibling_read", "sibling_write", "host_read", "host_write", "parent_environment", "parent_memory_open", "pid1_environment", "socket_create", "fork", "exec", "regain_uid_zero", "regain_gid_zero", "file_size"):
        if unique.get("checks", {}).get(name, {}).get("allowed") is not False:
            failures.append(f"unique UID did not deny {name}")
    if unique.get("status_before_seccomp", {}).get("CapEff") != "0000000000000000":
        failures.append("unique UID retained effective capabilities")
    if unique.get("status_after_seccomp", {}).get("NoNewPrivs") != "1":
        failures.append("no_new_privs not active")
    if unique.get("checks", {}).get("fd_limit", {}).get("errno") != errno.EMFILE:
        failures.append("FD limit not enforced")
    memory = report["supervisor"]["unique_uid_memory"].get("result") or {}
    if report["supervisor"]["unique_uid_memory"]["returncode"] != 0 or memory.get("memory_error") is not True:
        failures.append("address-space limit not enforced")
    if report["supervisor"]["unique_uid_cpu"]["returncode"] not in (-signal.SIGXCPU, -signal.SIGKILL):
        failures.append("CPU limit did not terminate child")
    if not unique.get("input_inventory", {}).get("exists") or not unique.get("input_inventory", {}).get("readable"):
        failures.append("official public competition source unavailable to unique UID")
    verdict = "unique_uid_hardened_host_qualified_diagnostic" if not failures else "authority_or_guardrail_failed"
    return verdict, failures


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--guarded":
        print(json.dumps(guarded_child(sys.argv[2]), sort_keys=True), flush=True)
        return
    if len(sys.argv) == 2 and sys.argv[1] == "--supervisor":
        print(json.dumps(supervisor(), sort_keys=True), flush=True)
        return
    started = time.monotonic()
    prepare()
    environment = dict(os.environ)
    environment["TYCHO_AUTHORITY_PARENT_MARKER"] = "marker-value-must-never-be-reported"
    environment["TYCHO_PROBE_WORKING"] = str(WORKING)
    completed = subprocess.run(
        [sys.executable, str(PROBE), "--supervisor"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
        close_fds=True,
    )
    parsed = json.loads(completed.stdout.strip().splitlines()[-1]) if completed.returncode == 0 else {}
    report: dict[str, Any] = {
        "schema": "tycho-host-authority-census-kaggle-v1",
        "system": {"platform": platform.platform(), "python": sys.version, "uid": os.getuid(), "gid": os.getgid()},
        "supervisor_process": {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr},
        "supervisor": parsed,
        "canaries_after": {
            "game_a": (GAME_A / "canary.txt").read_text(),
            "game_b": (GAME_B / "canary.txt").read_text(),
            "host": (HOST / "canary.txt").read_text(),
        },
    }
    verdict, failures = classify(report)
    report.update({"verdict": verdict, "failures": failures, "elapsed_seconds": round(time.monotonic() - started, 3)})
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict, "failures": failures}, sort_keys=True))


if __name__ == "__main__":
    main()
