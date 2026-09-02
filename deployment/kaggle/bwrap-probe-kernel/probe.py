from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

OUTPUT = Path("/kaggle/working/result.json")
WORKSPACE = Path("/kaggle/working/bwrap-probe-workspace")
OUTSIDE = Path("/kaggle/working/outside-sentinel.txt")
COMMAND_TIMEOUT_SECONDS = 20


def capture(command: list[str], timeout: int = COMMAND_TIMEOUT_SECONDS) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "argv": command,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "argv": command,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "returncode": None,
            "stdout": error.stdout or "",
            "stderr": error.stderr or "",
            "timed_out": True,
        }
    except OSError as error:
        return {
            "argv": command,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(error).__name__}: {error}",
            "timed_out": False,
        }


def read_optional(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def status_fields() -> dict[str, str]:
    wanted = {"CapEff", "NoNewPrivs", "Seccomp", "Seccomp_filters"}
    fields: dict[str, str] = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in wanted:
            fields[key] = value.strip()
    return fields


def bwrap_command(bwrap: str) -> list[str]:
    payload = r'''
import json, os, socket, subprocess
from pathlib import Path

workspace = Path.cwd()
outside = Path("/kaggle/working/outside-sentinel.txt")
result = {
    "pid": os.getpid(),
    "ppid": os.getppid(),
    "environment_keys": sorted(os.environ),
    "parent_secret": os.environ.get("TYCHO_PROBE_PARENT_SECRET"),
    "outside_visible": outside.exists(),
    "pids": sorted(path.name for path in Path("/proc").iterdir() if path.name.isdigit()),
    "interfaces": socket.if_nameindex(),
}
try:
    parent = Path(f"/proc/{os.getppid()}/environ").read_bytes()
    result["parent_environment_readable"] = True
    result["parent_secret_visible"] = b"TYCHO_PROBE_PARENT_SECRET=" in parent
except BaseException as error:
    result["parent_environment_readable"] = False
    result["parent_environment_error"] = type(error).__name__
    result["parent_secret_visible"] = False
try:
    (workspace / "sentinel.txt").write_text("mutated")
    result["workspace_write_denied"] = False
except BaseException as error:
    result["workspace_write_denied"] = True
    result["workspace_write_error"] = type(error).__name__
network = socket.socket()
network.settimeout(0.5)
try:
    result["network_connect_ex"] = network.connect_ex(("1.1.1.1", 53))
except BaseException as error:
    result["network_error"] = type(error).__name__
finally:
    network.close()
unshare = subprocess.run(
    ["/usr/bin/unshare", "--user", "--map-root-user", "/bin/true"],
    capture_output=True,
    text=True,
    check=False,
) if Path("/usr/bin/unshare").exists() else None
result["nested_userns"] = None if unshare is None else {
    "returncode": unshare.returncode,
    "stdout": unshare.stdout,
    "stderr": unshare.stderr,
}
print(json.dumps(result, sort_keys=True))
'''
    python = "/usr/bin/python3"
    return [
        bwrap,
        "--as-pid-1",
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--disable-userns",
        "--clearenv",
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/bin",
        "/bin",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib64",
        "/lib64",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        str(WORKSPACE),
        str(WORKSPACE),
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "PATH",
        "/usr/bin",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "PYTHONNOUSERSITE",
        "1",
        "--chdir",
        str(WORKSPACE),
        "--",
        python,
        "-c",
        payload,
    ]


def classify(report: dict[str, Any]) -> tuple[str, list[str]]:
    failures: list[str] = []
    bwrap = report["bwrap"]
    if bwrap["path"] is None:
        direct = report["direct_user_namespace"]
        verdict = (
            "bwrap_absent_userns_available"
            if direct["returncode"] == 0
            else "bwrap_absent_userns_rejected"
        )
        return verdict, failures
    execution = bwrap.get("execution")
    if not isinstance(execution, dict) or execution.get("returncode") != 0:
        return "bwrap_present_execution_failed", ["bwrap command did not exit zero"]
    try:
        child = json.loads(execution["stdout"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return "bwrap_present_malformed_output", ["child output was not one JSON object"]
    checks = {
        "child_pid_is_one": child.get("pid") == 1 and child.get("ppid") == 0,
        "direct_environment_cleared": child.get("parent_secret") is None,
        "parent_environment_hidden": child.get("parent_environment_readable") is False
        and child.get("parent_secret_visible") is False,
        "outside_file_hidden": child.get("outside_visible") is False,
        "workspace_read_only": child.get("workspace_write_denied") is True,
        "private_pid_view": child.get("pids") == ["1"],
        "network_namespace_private": child.get("interfaces") in ([[1, "lo"]], [(1, "lo")]),
        "nested_userns_disabled": isinstance(child.get("nested_userns"), dict)
        and child["nested_userns"].get("returncode") != 0,
    }
    bwrap["child"] = child
    bwrap["checks"] = checks
    failures.extend(name for name, passed in checks.items() if not passed)
    return ("bwrap_qualified" if not failures else "bwrap_present_boundary_failed"), failures


def main() -> None:
    started = time.monotonic()
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    (WORKSPACE / "sentinel.txt").write_text("workspace-sentinel\n")
    OUTSIDE.write_text("outside-sentinel\n")
    os.environ["TYCHO_PROBE_PARENT_SECRET"] = "must-not-cross-boundary"

    bwrap = shutil.which("bwrap")
    unshare = shutil.which("unshare")
    report: dict[str, Any] = {
        "schema": "tycho-bwrap-kaggle-probe-v1",
        "system": {
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
            "uid": os.getuid(),
            "gid": os.getgid(),
            "status": status_fields(),
            "max_user_namespaces": read_optional("/proc/sys/user/max_user_namespaces"),
            "unprivileged_userns_clone": read_optional("/proc/sys/kernel/unprivileged_userns_clone"),
            "apparmor_restrict_unprivileged_userns": read_optional(
                "/proc/sys/kernel/apparmor_restrict_unprivileged_userns"
            ),
            "nvidia_smi": shutil.which("nvidia-smi"),
        },
        "unshare": {
            "path": unshare,
            "version": capture([unshare, "--version"]) if unshare else None,
        },
        "direct_user_namespace": capture(
            [unshare, "--user", "--map-root-user", "--mount", "--pid", "--fork", "/bin/true"]
        ) if unshare else {
            "argv": [], "returncode": None, "stdout": "", "stderr": "unshare absent", "timed_out": False
        },
        "bwrap": {
            "path": bwrap,
            "version": capture([bwrap, "--version"]) if bwrap else None,
        },
    }
    if bwrap:
        report["bwrap"]["execution"] = capture(bwrap_command(bwrap))
    verdict, failures = classify(report)
    report["verdict"] = verdict
    report["failures"] = failures
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["workspace_sentinel_after"] = (WORKSPACE / "sentinel.txt").read_text()
    report["outside_sentinel_after"] = OUTSIDE.read_text()
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict, "failures": failures}, sort_keys=True))


if __name__ == "__main__":
    main()
