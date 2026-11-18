#!/usr/bin/env python3
# ruff: noqa: BLE001, DTZ007
"""One-request deployment probe for Qwen3.8-27B-FP8 on Kaggle.

Question: can the immutable Qwen FP8 checkpoint and pinned SGLang route cold-load
without network access and turn one ARC-like image into one parsed native tool
call? This is a feasibility probe, not a benchmark or Solver implementation.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.server
import importlib.metadata
import json
import os
import platform
import re
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

MODEL_ID = "Qwen/Qwen3.8-27B-FP8"
MODEL_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
EXPECTED_SERVED_MODEL_ID = (
    "/kaggle/input/datasets/jakobbrggen/qwen3-8-27b-fp8-hf-snapshot"
)
EXPECTED_IMAGE_SHA256 = "158e93fdce8151a7af1160873a5121d6886eacacc8de3fd3b73b19d35033b87d"
EXPECTED_REQUEST_SHA256 = "f4108f8f6b05f32bd84dd4bfaad69968cff537d93a09d467d0f77393a6413cf0"
EXPECTED_REQUEST_BYTES = 2447
SGLANG_READY_MARKER = "The server is fired up and ready to roll!"
CHAT_COMPLETION_ACCESS_PREFIX = 'POST /v1/chat/completions HTTP/1.1"'
CHAT_COMPLETION_ACCESS_MARKER = 'POST /v1/chat/completions HTTP/1.1" 200 OK'
FREEZE_GC_ACCESS_MARKER = 'POST /freeze_gc HTTP/1.1" 200 OK'
SGLANG_REPOSITORY = "https://github.com/sgl-project/sglang.git"
SGLANG_COMMIT = "8e04f66a7046fa083cda3f5b8e8325632a5af8fd"
BASE_RUNTIME_DATASET_SOURCE = "mahdiseddigh/kaggle-t4-stack-2026-08-28"
BASE_RUNTIME_DATASET_VERSION = 1
BASE_RUNTIME_WHEEL_SHA256 = "9d07f8bf3b718dc4cd970e72a03415b0b0597da89625acbee7a655d6387e7a8d"
RUNTIME_DATASET_SOURCE = "pranavagrawai/sglang-r8l3b-orderly-shutdown-runtime"
RUNTIME_DATASET_VERSION = 1
RUNTIME_WHEEL_SHA256 = "06cc48d1e067c0eb08a1548c5c244bd0f541b114947c29b87042ce3156bb5cc0"
RUNTIME_LOCK_SHA256 = "d473b6e092e01f66313e12069ac8873f2ff1c6c3b84b51c1b37312229de5475c"
RUNTIME_DATASET_MANIFEST_SHA256 = (
    "f1a50bb224dc26ac48c0fd219c28e4fadc92ede6d2a80a26fd18d9a004a79b94"
)
DEFAULT_MANIFEST = Path(__file__).with_name("PROTOTYPE-qwen3_8_fp8-manifest.json")
EXPECTED_GPU_FRAGMENT = "RTX PRO 6000"
PROBE_SCHEMA = "qwen3.8-fp8-r8-l3b-probe-v1"
R8_L3B_OWNER = "pranavagrawai/prototype-qwen3-8-fp8-r8-b-request-path:v13"
R8_L3B_PORT = 30000


class LifecycleObservationError(RuntimeError):
    """The external observer could not establish a required fact."""


class ResponseValidationError(RuntimeError):
    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


def utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_capture(command: list[str], timeout_s: int = 30) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "elapsed_s": round(time.monotonic() - started, 3),
        }
    except Exception as error:
        return {
            "command": command,
            "error": f"{type(error).__name__}: {error}",
            "elapsed_s": round(time.monotonic() - started, 3),
        }


def _proc_stat(pid: int) -> dict[str, int | str]:
    path = Path("/proc") / str(pid) / "stat"
    try:
        raw = path.read_text()
    except FileNotFoundError as error:
        raise ProcessLookupError(pid) from error
    except PermissionError as error:
        raise LifecycleObservationError(f"permission denied reading {path}") from error
    close = raw.rfind(")")
    if close < 0:
        raise LifecycleObservationError(f"unparsable {path}")
    fields = raw[close + 2 :].split()
    if len(fields) < 20:
        raise LifecycleObservationError(f"short {path}")
    return {
        "pid": pid,
        "state": fields[0],
        "ppid": int(fields[1]),
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
        "start_time": int(fields[19]),
    }


def _read_proc_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as error:
        raise ProcessLookupError(path) from error
    except PermissionError as error:
        raise LifecycleObservationError(f"permission denied reading {path}") from error


def process_identity(pid: int, *, inspect_environment: bool = True) -> dict[str, Any]:
    stat_value = _proc_stat(pid)
    root = Path("/proc") / str(pid)
    terminal_state = stat_value["state"] in {"Z", "X", "x"}
    command_bytes = b"" if terminal_state else _read_proc_bytes(root / "cmdline")
    command = command_bytes.replace(b"\0", b" ").decode(
        "utf-8", errors="replace"
    ).strip()
    environment = (
        []
        if terminal_state or not inspect_environment
        else _read_proc_bytes(root / "environ").split(b"\0")
    )
    try:
        cgroup = (root / "cgroup").read_text().splitlines()
    except FileNotFoundError as error:
        raise ProcessLookupError(pid) from error
    except PermissionError as error:
        raise LifecycleObservationError(f"permission denied reading {root / 'cgroup'}") from error
    return {
        **stat_value,
        "identity": f"{pid}:{stat_value['start_time']}",
        "command": command,
        "command_bytes_b64": base64.b64encode(command_bytes).decode("ascii"),
        "command_bytes_sha256": sha256_bytes(command_bytes),
        "cgroup": cgroup,
        "environment_inspected": inspect_environment and not terminal_state,
        "owner_marker": any(item == f"R8_L3B_OWNER={R8_L3B_OWNER}".encode() for item in environment),
    }


def observer_identity() -> dict[str, Any]:
    value = process_identity(os.getpid())
    value["uid"] = os.getuid()
    return value


def _same_user_pids() -> list[int]:
    pids: list[int] = []
    uid = os.getuid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid == uid:
                pids.append(int(entry.name))
        except FileNotFoundError:
            continue
        except PermissionError as error:
            raise LifecycleObservationError(f"permission denied inspecting {entry}") from error
    return sorted(pids)


def _all_process_stats() -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            value = dict(_proc_stat(pid))
            value["uid"] = entry.stat().st_uid
            result[pid] = value
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise LifecycleObservationError(f"permission denied inspecting {entry}") from error
    return result


def _listening_inodes(port: int) -> set[str]:
    inodes: set[str] = set()
    for path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = path.read_text().splitlines()[1:]
        except (FileNotFoundError, PermissionError) as error:
            raise LifecycleObservationError(f"cannot inspect {path}: {error}") from error
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                local_port = int(fields[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError) as error:
                raise LifecycleObservationError(f"unparsable listener row in {path}: {line}") from error
            if local_port == port:
                inodes.add(fields[9])
    return inodes


def _socket_inodes(pid: int) -> set[str]:
    fd_root = Path("/proc") / str(pid) / "fd"
    result: set[str] = set()
    try:
        entries = list(fd_root.iterdir())
    except FileNotFoundError:
        return result
    except PermissionError as error:
        try:
            if _proc_stat(pid)["state"] in {"Z", "X", "x"}:
                return result
        except ProcessLookupError:
            return result
        raise LifecycleObservationError(f"permission denied listing {fd_root}") from error
    for entry in entries:
        try:
            target = os.readlink(entry)
        except FileNotFoundError:
            continue
        except PermissionError as error:
            raise LifecycleObservationError(f"permission denied reading {entry}") from error
        match = re.fullmatch(r"socket:\[(\d+)\]", target)
        if match:
            result.add(match.group(1))
    return result


def gpu_processes() -> dict[int, int]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise LifecycleObservationError("nvidia-smi not found")
    captured = run_capture(
        [
            executable,
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if captured.get("returncode") != 0 or captured.get("error"):
        raise LifecycleObservationError(f"nvidia-smi compute-app query failed: {captured}")
    result: dict[int, int] = {}
    for line in captured.get("stdout", "").splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise LifecycleObservationError(f"unparsable nvidia-smi row: {line!r}")
        try:
            result[int(fields[0])] = int(fields[1])
        except ValueError as error:
            raise LifecycleObservationError(f"unparsable nvidia-smi row: {line!r}") from error
    return result


def scan_lifecycle(
    target: dict[str, Any],
    persistent_owned: set[str],
    *,
    port: int,
    gpu_query: Callable[[], dict[int, int]] = gpu_processes,
    require_listener_owner: bool = False,
    admit_numeric_group: bool = True,
) -> dict[str, Any]:
    stats = _all_process_stats()
    same_user_pids = set(_same_user_pids())
    owned_candidates: set[int] = set()
    for pid, value in stats.items():
        identity = f"{pid}:{value['start_time']}"
        if (
            identity == target["identity"]
            or (
                admit_numeric_group
                and (value["pgid"] == target["pgid"] or value["sid"] == target["sid"])
            )
            or identity in persistent_owned
        ):
            owned_candidates.add(pid)
    changed = True
    while changed:
        changed = False
        for pid, value in stats.items():
            if pid not in owned_candidates and value["ppid"] in owned_candidates:
                owned_candidates.add(pid)
                changed = True

    identities: dict[int, dict[str, Any]] = {}
    for pid in sorted(same_user_pids | owned_candidates):
        try:
            identities[pid] = process_identity(
                pid, inspect_environment=pid in same_user_pids
            )
        except ProcessLookupError:
            continue

    listener_inodes = _listening_inodes(port)
    listener_owners: dict[int, list[str]] = {}
    for pid, identity in identities.items():
        owned_inodes = (
            []
            if not listener_inodes or identity["state"] in {"Z", "X", "x"}
            else sorted(_socket_inodes(pid) & listener_inodes)
        )
        if owned_inodes:
            listener_owners[pid] = owned_inodes
    resolved_inodes = set().union(*(set(value) for value in listener_owners.values()))
    unresolved_inodes = listener_inodes - resolved_inodes
    if require_listener_owner and unresolved_inodes:
        raise LifecycleObservationError(
            f"could not resolve port-{port} listener inodes {sorted(unresolved_inodes)} "
            "to same-user processes"
        )

    reasons: dict[int, set[str]] = {pid: set() for pid in identities}
    for pid, value in identities.items():
        if value["identity"] == target["identity"]:
            reasons[pid].add("target_root")
        if admit_numeric_group and value["pgid"] == target["pgid"]:
            reasons[pid].add("target_pgid")
        if admit_numeric_group and value["sid"] == target["sid"]:
            reasons[pid].add("target_sid")
        if value["owner_marker"]:
            reasons[pid].add("owner_marker")
        if pid in listener_owners:
            reasons[pid].add(f"listener:{port}")
        if value["identity"] in persistent_owned:
            reasons[pid].add("recorded_identity")

    changed = True
    while changed:
        changed = False
        owned_pids = {pid for pid, value in reasons.items() if value}
        for pid, value in identities.items():
            if not reasons[pid] and value["ppid"] in owned_pids:
                reasons[pid].add("recursive_descendant")
                changed = True

    gpu = gpu_query()
    processes: list[dict[str, Any]] = []
    for pid in sorted(pid for pid, value in reasons.items() if value):
        value = dict(identities[pid])
        value["ownership_reasons"] = sorted(reasons[pid])
        value["listener_socket_inodes"] = listener_owners.get(pid, [])
        value["listener_ports"] = [port] if pid in listener_owners else []
        value["gpu_used_memory_mib"] = gpu.get(pid)
        processes.append(value)
        persistent_owned.add(value["identity"])

    target_gpu = {
        str(pid): memory for pid, memory in gpu.items() if reasons.get(pid)
    }
    unattributed_gpu = {
        str(pid): memory for pid, memory in gpu.items() if str(pid) not in target_gpu
    }
    return {
        "captured_at": utc_now(),
        "monotonic_s": time.monotonic(),
        "target_identity": target["identity"],
        "processes": processes,
        "listeners": {
            "port": port,
            "inodes": sorted(listener_inodes),
            "unresolved_inodes": sorted(unresolved_inodes),
            "owners": {str(pid): value for pid, value in sorted(listener_owners.items())},
        },
        "gpu_processes": target_gpu,
        "unattributed_gpu_processes": unattributed_gpu,
        "closed": (
            not processes
            and not listener_inodes
            and not target_gpu
            and not unattributed_gpu
        ),
    }


def owned_group_generation_alive(
    target: dict[str, Any],
    processes: list[dict[str, Any]],
    trusted_identities: set[str],
) -> bool:
    """Return whether the numeric PGID still contains a pre-TERM owned generation."""
    for process in processes:
        if (
            process.get("pgid") != target["pgid"]
            or process.get("identity") not in trusted_identities
        ):
            continue
        try:
            current = _proc_stat(process["pid"])
        except (FileNotFoundError, ProcessLookupError):
            continue
        if (
            f"{process['pid']}:{current['start_time']}" == process["identity"]
            and current["pgid"] == target["pgid"]
        ):
            return True
    return False


def emergency_contain_process_group(
    process: subprocess.Popen[str],
    launch_target: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Contain only a launch generation that remains identity-bound to its PGID."""
    signals: list[dict[str, Any]] = []
    if process.poll() is not None:
        return signals
    if launch_target is not None and owned_group_generation_alive(
        launch_target,
        [launch_target],
        {launch_target["identity"]},
    ):
        try:
            os.killpg(launch_target["pgid"], signal.SIGKILL)
            signals.append(
                {
                    "signal": "SIGKILL",
                    "target": f"pgid:{launch_target['pgid']}",
                    "sent": True,
                    "reason": "observer_contract_invalid",
                }
            )
        except ProcessLookupError:
            signals.append(
                {
                    "signal": "SIGKILL",
                    "target": f"pgid:{launch_target['pgid']}",
                    "sent": False,
                    "result": "group_absent",
                    "reason": "observer_contract_invalid",
                }
            )
    else:
        signals.append(
            {
                "signal": "SIGKILL",
                "target": (
                    f"pgid:{launch_target['pgid']}" if launch_target is not None else None
                ),
                "sent": False,
                "result": "not_sent_launch_identity_unverified",
                "reason": "observer_contract_invalid",
            }
        )
    if any(item["sent"] for item in signals):
        process.wait(timeout=10)
    return signals


def contain_recorded_survivors(
    target: dict[str, Any],
    survivors: list[dict[str, Any]],
    preterm_owned_identities: set[str],
    *,
    term_started_at: float | None = None,
) -> list[dict[str, Any]]:
    """Signal the group only with generation proof; otherwise signal identities singly."""

    def signal_record(**fields: Any) -> dict[str, Any]:
        elapsed = (
            round(time.monotonic() - term_started_at, 3)
            if term_started_at is not None
            else 0.0
        )
        return {"signal": "SIGKILL", "at": utc_now(), "elapsed_s": elapsed, **fields}

    signals: list[dict[str, Any]] = []
    group_signal_sent = False
    if owned_group_generation_alive(target, survivors, preterm_owned_identities):
        try:
            os.killpg(target["pgid"], signal.SIGKILL)
            group_signal_sent = True
            signals.append(
                signal_record(target=f"pgid:{target['pgid']}", sent=True)
            )
        except ProcessLookupError:
            signals.append(
                signal_record(
                    target=f"pgid:{target['pgid']}",
                    sent=False,
                    result="group_absent",
                )
            )
    else:
        signals.append(
            signal_record(
                target=f"pgid:{target['pgid']}",
                sent=False,
                result="not_sent_no_current_preterm_group_identity",
            )
        )

    for survivor in survivors:
        if group_signal_sent and survivor["pgid"] == target["pgid"]:
            continue
        try:
            current = _proc_stat(survivor["pid"])
        except (FileNotFoundError, ProcessLookupError):
            signals.append(
                signal_record(
                    target=f"pid:{survivor['pid']}",
                    target_identity=survivor["identity"],
                    sent=False,
                    result="process_absent",
                )
            )
            continue
        current_identity = f"{survivor['pid']}:{current['start_time']}"
        if current_identity != survivor["identity"]:
            signals.append(
                signal_record(
                    target=f"pid:{survivor['pid']}",
                    target_identity=survivor["identity"],
                    sent=False,
                    result=f"not_sent_pid_reused_as:{current_identity}",
                )
            )
            continue
        os.kill(survivor["pid"], signal.SIGKILL)
        signals.append(
            signal_record(
                target=f"pid:{survivor['pid']}",
                target_identity=survivor["identity"],
                sent=True,
            )
        )
    return signals


def lifecycle_observer_preflight(port: int) -> dict[str, Any]:
    readable: list[str] = []
    errors: list[dict[str, Any]] = []
    for pid in _same_user_pids():
        try:
            readable.append(process_identity(pid)["identity"])
        except ProcessLookupError:
            continue
        except LifecycleObservationError as error:
            errors.append({"pid": pid, "error": str(error)})
    listener_inodes = sorted(_listening_inodes(port))
    if listener_inodes:
        errors.append(
            {
                "port": port,
                "listener_inodes": listener_inodes,
                "error": "lifecycle port is already bound before target launch",
            }
        )
    try:
        gpu_query = gpu_processes()
    except LifecycleObservationError as error:
        gpu_query = None
        errors.append({"error": str(error)})
    if gpu_query:
        errors.append(
            {
                "gpu_query": {str(pid): memory for pid, memory in gpu_query.items()},
                "error": "GPU compute applications already exist before target launch",
            }
        )
    result = {
        "captured_at": utc_now(),
        "observer": observer_identity(),
        "same_user_readable_identities": readable,
        "port": port,
        "listener_inodes": listener_inodes,
        "gpu_query": gpu_query,
        "errors": errors,
        "status": "passed" if not errors else "invalid",
    }
    return result


def classify_teardown_log(value: bytes, *, source_offset: int = 0) -> dict[str, Any]:
    from datetime import datetime

    value.decode("utf-8")
    entries: list[dict[str, Any]] = []

    def add(start: int, end: int, class_name: str, rule: str, allowed: bool) -> None:
        entries.append(
            {
                "byte_start": start,
                "byte_end": end,
                "source_byte_start": source_offset + start,
                "source_byte_end": source_offset + end,
                "text": value[start:end].decode("utf-8"),
                "class": class_name,
                "rule": rule,
                "allowed": allowed,
            }
        )
    timestamp = re.compile(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (.+)"
    )
    allowed_lines = (
        (
            re.compile(
                r"SIGTERM received\. signum=(?:None|-?\d+) frame=None\. "
                r"Draining requests and shutting down\.\.\."
            ),
            "sglang_sigterm",
        ),
        (
            re.compile(
                r"Gracefully exiting\.\.\. Remaining number of requests 0"
                r"(?:\. Remaining requests remaining_rids=\[\]\.)?"
            ),
            "sglang_graceful_exit",
        ),
        (
            re.compile(
                r"kill_process_tree called: parent_pid=\d+, "
                r"include_parent=(?:True|False), pid=\d+"
            ),
            "sglang_kill_process_tree_notice",
        ),
        (
            re.compile(
                r"INFO:\s+(?:Shutting down|Waiting for application shutdown\.|"
                r"Waiting for background tasks to complete\. \(CTRL\+C to force quit\)|"
                r"Application shutdown complete\.|Finished server process \[\d+\])"
            ),
            "uvicorn_normal_shutdown",
        ),
        (
            re.compile(
                r"import time:\s+\d+\s*\|\s*\d+\s*\|\s*"
                r"[A-Za-z_][A-Za-z0-9_.]*"
            ),
            "python_import_timing",
        ),
    )
    offset = 0
    for line in value.splitlines(keepends=True):
        end = offset + len(line)
        decoded = line.decode("utf-8").rstrip("\r\n")
        if not decoded.strip():
            offset = end
            continue
        normalized = decoded
        if decoded.startswith("["):
            timestamp_match = timestamp.fullmatch(decoded)
            if timestamp_match is None:
                normalized = ""
            else:
                try:
                    datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    normalized = ""
                else:
                    normalized = timestamp_match.group(2)
        matched = False
        for pattern, rule in allowed_lines:
            if pattern.fullmatch(normalized):
                add(offset, end, "allowed_normal_event", rule, True)
                matched = True
                break
        if not matched:
            running = re.search(
                r"#running-req:\s*([1-9]\d*)|#queue-req:\s*([1-9]\d*)|"
                r"Remaining number of requests\s+([1-9]\d*)|"
                r"remaining_rids=\[(?!\])|request[^\r\n]*(?:cancelled|canceled)",
                decoded,
                re.IGNORECASE,
            )
            if running:
                add(
                    offset,
                    end,
                    "forbidden_request_inflight",
                    "request_running_or_queued_after_term",
                    False,
                )
            else:
                add(
                    offset,
                    end,
                    "forbidden_unclassified_log_line",
                    "default_deny_unmatched_nonempty_line",
                    False,
                )
        offset = end

    entries.sort(key=lambda item: (item["byte_start"], item["byte_end"]))
    forbidden = [entry for entry in entries if not entry["allowed"]]
    return {
        "bytes": len(value),
        "entries": entries,
        "forbidden_count": len(forbidden),
        "allowed": not forbidden,
    }


def combine_r8l3b_result(
    request_result: dict[str, Any], lifecycle: dict[str, Any]
) -> dict[str, Any]:
    """Make R8-L3B PASS one explicit conjunction over both qualified paths."""
    request_status = request_result.get("status")
    lifecycle_status = lifecycle.get("status")
    gate = {
        "request_path_passed": request_status == "passed",
        "lifecycle_passed": lifecycle_status == "passed",
        "teardown_log_accepted": lifecycle.get("log_classification_allowed") is True,
    }
    request_result["request_path_status"] = request_status
    request_result["lifecycle"] = lifecycle
    request_result["r8_l3b_gate"] = gate

    if all(gate.values()):
        request_result["status"] = "passed"
        request_result.pop("failure_classification", None)
    elif request_status != "passed":
        request_result["status"] = (
            request_status if request_status in {"failed", "invalid"} else "invalid"
        )
    else:
        request_result["status"] = (
            lifecycle_status if lifecycle_status in {"failed", "invalid"} else "failed"
        )
        request_result["failure_classification"] = (
            lifecycle.get("failure_classification") or "teardown_log_unacceptable"
        )
    return request_result


def write_r8l3b_result(
    output_dir: Path,
    request_result: dict[str, Any],
    lifecycle: dict[str, Any],
) -> dict[str, Any]:
    result = combine_r8l3b_result(request_result, lifecycle)
    write_json(output_dir / "result.json", result)
    return result


def qualify_lifecycle(
    process: subprocess.Popen[str],
    server_log: Any,
    output_dir: Path,
    *,
    port: int,
    term_grace_s: float = 30,
    kill_wait_s: float = 10,
    quiet_window_s: float = 5,
    deadline_s: float = 45,
    poll_s: float = 0.25,
    gpu_query: Callable[[], dict[int, int]] = gpu_processes,
) -> dict[str, Any]:
    server_log.flush()
    target = process_identity(process.pid)
    observer = observer_identity()
    observer_record = {
        "observer": observer,
        "target": target,
        "outside_target_pgid": observer["pgid"] != target["pgid"],
        "outside_target_sid": observer["sid"] != target["sid"],
    }
    write_json(output_dir / "lifecycle-observer.json", observer_record)
    if not observer_record["outside_target_pgid"] or not observer_record["outside_target_sid"]:
        raise LifecycleObservationError("observer shares the target PGID or SID")
    if target["pgid"] != target["pid"] or target["sid"] != target["pid"]:
        raise LifecycleObservationError("target root is not leader of its new process group/session")
    if not target["owner_marker"]:
        raise LifecycleObservationError("target root lacks the exact R8_L3B_OWNER marker")

    persistent_owned: set[str] = set()
    before = scan_lifecycle(
        target,
        persistent_owned,
        port=port,
        gpu_query=gpu_query,
        require_listener_owner=True,
    )
    write_json(output_dir / "lifecycle-owned-before.json", before)
    write_json(output_dir / "listeners-before.json", before["listeners"])
    write_json(output_dir / "gpu-processes-before.json", before["gpu_processes"])
    if target["identity"] not in {item["identity"] for item in before["processes"]}:
        raise LifecycleObservationError("pre-TERM inventory omitted the target root")
    preterm_owned_identities = {item["identity"] for item in before["processes"]}

    t0 = time.monotonic()
    log_offset = (output_dir / "server.log").stat().st_size
    signals: list[dict[str, Any]] = []
    try:
        current = _proc_stat(target["pid"])
    except (FileNotFoundError, ProcessLookupError) as error:
        raise LifecycleObservationError("target root exited before TERM") from error
    current_identity = f"{target['pid']}:{current['start_time']}"
    if current_identity != target["identity"]:
        raise LifecycleObservationError(
            f"target root identity changed before TERM: {target['identity']} -> {current_identity}"
        )
    os.kill(target["pid"], signal.SIGTERM)
    signals.append(
        {
            "signal": "SIGTERM",
            "target": f"pid:{target['pid']}",
            "target_identity": target["identity"],
            "at": utc_now(),
            "elapsed_s": 0.0,
        }
    )
    polls_path = output_dir / "lifecycle-polls.jsonl"
    latest = before
    term_closed_at: float | None = None
    first_closed_scan: dict[str, Any] | None = None
    with polls_path.open("w") as polls:
        while time.monotonic() - t0 < term_grace_s:
            process.poll()
            latest = scan_lifecycle(
                target,
                persistent_owned,
                port=port,
                gpu_query=gpu_query,
                admit_numeric_group=False,
            )
            latest["phase"] = "term_grace"
            scan_completed_elapsed_s = latest["monotonic_s"] - t0
            latest["elapsed_since_term_s"] = round(scan_completed_elapsed_s, 3)
            polls.write(json.dumps(latest, sort_keys=True) + "\n")
            polls.flush()
            if latest["closed"]:
                first_closed_scan = dict(latest)
                if scan_completed_elapsed_s <= term_grace_s:
                    term_closed_at = latest["monotonic_s"]
                break
            if scan_completed_elapsed_s >= term_grace_s:
                break
            time.sleep(poll_s)

        kill_used = term_closed_at is None
        if kill_used:
            write_json(output_dir / "lifecycle-term-timeout.json", latest)
            signals.extend(
                contain_recorded_survivors(
                    target,
                    latest["processes"],
                    preterm_owned_identities,
                    term_started_at=t0,
                )
            )
            containment_deadline = min(deadline_s, term_grace_s + kill_wait_s)
            while (
                not latest["closed"]
                and time.monotonic() - t0 < containment_deadline
            ):
                process.poll()
                latest = scan_lifecycle(
                    target,
                    persistent_owned,
                    port=port,
                    gpu_query=gpu_query,
                    admit_numeric_group=False,
                )
                latest["phase"] = "kill_containment"
                latest["elapsed_since_term_s"] = round(time.monotonic() - t0, 3)
                polls.write(json.dumps(latest, sort_keys=True) + "\n")
                polls.flush()
                if latest["closed"]:
                    if first_closed_scan is None:
                        first_closed_scan = dict(latest)
                    break
                time.sleep(poll_s)

        quiet_scans: list[dict[str, Any]] = (
            [first_closed_scan] if first_closed_scan is not None else []
        )
        while True:
            if time.monotonic() - t0 > deadline_s:
                break
            process.poll()
            quiet = scan_lifecycle(
                target,
                persistent_owned,
                port=port,
                gpu_query=gpu_query,
                admit_numeric_group=False,
            )
            quiet["phase"] = "quiet_window"
            quiet["elapsed_since_term_s"] = round(time.monotonic() - t0, 3)
            quiet_scans.append(quiet)
            if first_closed_scan is None and quiet["closed"]:
                first_closed_scan = dict(quiet)
            polls.write(json.dumps(quiet, sort_keys=True) + "\n")
            polls.flush()
            if not quiet["closed"]:
                break
            quiet_elapsed = (
                quiet_scans[-1]["monotonic_s"] - quiet_scans[0]["monotonic_s"]
            )
            if len(quiet_scans) >= 2 and quiet_elapsed >= quiet_window_s:
                break
            deadline_remaining = deadline_s - (time.monotonic() - t0)
            if deadline_remaining <= 0:
                break
            time.sleep(min(1.0, quiet_window_s - quiet_elapsed, deadline_remaining))
        latest = quiet_scans[-1] if quiet_scans else latest

    try:
        process.wait(timeout=max(0.0, deadline_s - (time.monotonic() - t0)))
    except subprocess.TimeoutExpired:
        pass
    server_log.flush()
    teardown_bytes = (output_dir / "server.log").read_bytes()[log_offset:]
    (output_dir / "teardown.log").write_bytes(teardown_bytes)
    classification = classify_teardown_log(teardown_bytes, source_offset=log_offset)
    write_json(output_dir / "teardown-log-classification.json", classification)
    write_json(output_dir / "lifecycle-signals.json", signals)
    write_json(output_dir / "lifecycle-owned-after.json", latest)
    write_json(output_dir / "listeners-after.json", latest["listeners"])
    write_json(output_dir / "gpu-processes-after.json", latest["gpu_processes"])

    quiet_valid = (
        len(quiet_scans) >= 2
        and quiet_scans[-1]["monotonic_s"] - quiet_scans[0]["monotonic_s"]
        >= quiet_window_s
        and all(scan["closed"] for scan in quiet_scans)
    )
    elapsed = time.monotonic() - t0
    if not latest["closed"] or elapsed > deadline_s:
        status, failure = "failed", "containment_incomplete"
    elif kill_used:
        status, failure = "failed", "kill_required"
    elif not quiet_valid:
        status, failure = "invalid", "quiet_window_invalid"
    elif process.returncode != 0:
        status, failure = "failed", "root_returncode_nonzero"
    elif not classification["allowed"]:
        status, failure = "failed", "teardown_log_unacceptable"
    else:
        status, failure = "passed", None
    result = {
        "status": status,
        "failure_classification": failure,
        "term_only": not kill_used,
        "containment_passed": latest["closed"],
        "quiet_window_valid": quiet_valid,
        "quiet_scan_count": len(quiet_scans),
        "quiet_window_anchor_phase": (
            quiet_scans[0].get("phase") if quiet_scans else None
        ),
        "quiet_window_anchor_monotonic_s": (
            quiet_scans[0]["monotonic_s"] if quiet_scans else None
        ),
        "first_closed_scan_completed_elapsed_s": (
            quiet_scans[0].get("elapsed_since_term_s") if quiet_scans else None
        ),
        "term_closure_completed_elapsed_s": (
            round(term_closed_at - t0, 3) if term_closed_at is not None else None
        ),
        "root_returncode": process.returncode,
        "elapsed_since_term_s": round(elapsed, 3),
        "log_classification_allowed": classification["allowed"],
    }
    write_json(output_dir / "shutdown.json", result)
    return result


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def system_snapshot() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    gpu = (
        run_capture(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        if nvidia_smi
        else {"error": "nvidia-smi not found"}
    )
    return {
        "captured_at": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "kaggle": {
            "competition_rerun": bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN")),
            "kernel_run_type": os.getenv("KAGGLE_KERNEL_RUN_TYPE"),
            "working_directory_present": Path("/kaggle/working").is_dir(),
            "input_directory_present": Path("/kaggle/input").is_dir(),
        },
        "packages": {
            name: package_version(name)
            for name in ("sglang", "torch", "transformers", "flashinfer-python")
        },
        "commands": {
            "sglang": shutil.which("sglang"),
            "nvidia_smi": nvidia_smi,
        },
        "gpu": gpu,
    }


def gpu_name(snapshot: dict[str, Any]) -> str:
    gpu = snapshot["gpu"]
    if gpu.get("returncode") != 0:
        raise RuntimeError(f"cannot inspect GPU: {gpu}")
    first_line = gpu.get("stdout", "").strip().splitlines()
    if not first_line:
        raise RuntimeError("nvidia-smi returned no GPU")
    return first_line[0].split(",", 1)[0].strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def verify_model(model_path: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise RuntimeError("probe manifest does not bind the expected model revision")

    checks: list[dict[str, Any]] = []
    started = time.monotonic()
    for expected in manifest["files"]:
        relative = expected["path"]
        actual_path = model_path / relative
        check: dict[str, Any] = {
            "path": relative,
            "expected_size": expected["size"],
            "expected_sha256": expected["sha256"],
            "required_for_runtime": expected.get("required_for_runtime", True),
        }
        if not actual_path.is_file():
            check["status"] = (
                "missing" if check["required_for_runtime"] else "optional_missing"
            )
        else:
            check["actual_size"] = actual_path.stat().st_size
            if check["actual_size"] != expected["size"]:
                check["status"] = "size_mismatch"
            else:
                check["actual_sha256"] = sha256_file(actual_path)
                check["status"] = (
                    "matched"
                    if check["actual_sha256"] == expected["sha256"]
                    else "hash_mismatch"
                )
        checks.append(check)
        if check["status"] not in ("matched", "optional_missing"):
            break

    failures = [
        check
        for check in checks
        if check["status"] not in ("matched", "optional_missing")
    ]
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    quantization = config.get("quantization_config", {})
    structure = {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "quant_method": quantization.get("quant_method"),
    }
    structure_matched = (
        structure["model_type"] == "qwen3_5"
        and structure["quant_method"] == "fp8"
        and "Qwen3_5ForConditionalGeneration" in (structure["architectures"] or [])
    )
    result = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "manifest_file_count": len(manifest["files"]),
        "checked_file_count": len(checks),
        "elapsed_s": round(time.monotonic() - started, 3),
        "structure": structure,
        "structure_matched": structure_matched,
        "failures": failures,
        "status": "matched" if not failures and structure_matched else "failed",
        "files": checks,
    }
    if result["status"] != "matched":
        raise RuntimeError(f"mounted model does not match immutable manifest: {failures or structure}")
    return result


def discover_model(input_root: Path, manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text())
    expected_size = manifest["safetensors_bytes"]
    candidates: list[Path] = []
    for config_path in input_root.rglob("config.json"):
        root = config_path.parent
        tensors = list(root.glob("*.safetensors"))
        if len(tensors) == manifest["safetensors_count"] and sum(
            path.stat().st_size for path in tensors
        ) == expected_size:
            candidates.append(root)
    if len(candidates) != 1:
        raise RuntimeError(
            "expected exactly one attached Qwen3.8 FP8 directory under "
            f"{input_root}, found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def arc_like_png() -> bytes:
    """Return a 256x256 ARC-like board with red left of yellow."""
    size = 256
    pixels: list[bytes] = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            color = (22, 24, 29)
            if x % 32 in (0, 1) or y % 32 in (0, 1):
                color = (58, 62, 70)
            if 48 <= x < 112 and 96 <= y < 160:
                color = (220, 48, 48)
            if 176 <= x < 240 and 96 <= y < 160:
                color = (245, 205, 48)
            row.extend(color)
        pixels.append(bytes(row))
    raw = b"".join(pixels)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw, 9))
        + png_chunk(b"IEND", b"")
    )


def build_request(model_name: str, image_bytes: bytes) -> dict[str, Any]:
    image_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    return {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the Game Judgment component for an ARC-like visual game. "
                    "Inspect the image and commit exactly one available action as a "
                    "one-element take_actions batch."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "The image contains one red square and one yellow square. "
                            "Choose LEFT if the red square is left of the yellow square; "
                            "otherwise choose RIGHT. Available actions: LEFT, RIGHT."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "take_actions",
                    "description": (
                        "The only action-commitment tool. Commit one or more ordered "
                        "environment actions."
                    ),
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "actions": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 1,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "action": {
                                            "type": "string",
                                            "enum": ["LEFT", "RIGHT"],
                                        }
                                    },
                                    "required": ["action"],
                                },
                            },
                            "rationale": {"type": "string", "minLength": 1},
                        },
                        "required": ["actions", "rationale"],
                    },
                },
            }
        ],
        "tool_choice": "required",
        "temperature": 0,
        "max_tokens": 2048,
        "chat_template_kwargs": {
            "enable_thinking": False,
            "preserve_thinking": True,
        },
    }


def http_json(url: str, *, payload: dict[str, Any] | None = None, timeout_s: int = 30) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read())


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def post_bytes(url: str, payload: bytes, timeout_s: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(NoRedirectHandler())
    try:
        with opener.open(request, timeout=timeout_s) as response:
            return {
                "status": response.status,
                "reason": response.reason,
                "headers": dict(response.headers.items()),
                "body": response.read(),
            }
    except urllib.error.HTTPError as error:
        return {
            "status": error.code,
            "reason": error.reason,
            "headers": dict(error.headers.items()),
            "body": error.read(),
        }


def wait_for_server(base_url: str, process: subprocess.Popen[str] | None, timeout_s: int) -> dict[str, Any]:
    started = time.monotonic()
    last_error = "not attempted"
    while time.monotonic() - started < timeout_s:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"SGLang exited before readiness with code {process.returncode}")
        try:
            models = http_json(base_url.rstrip("/") + "/v1/models", timeout_s=5)
            return {"elapsed_s": round(time.monotonic() - started, 3), "models": models}
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            time.sleep(2)
    raise TimeoutError(f"server was not ready in {timeout_s}s; last error: {last_error}")


def server_request_facts(log_text: str) -> dict[str, int]:
    return {
        "chat_completion_accesses": log_text.count(CHAT_COMPLETION_ACCESS_PREFIX),
        "successful_chat_completion_accesses": log_text.count(
            CHAT_COMPLETION_ACCESS_MARKER
        ),
        "successful_freeze_gc_accesses": log_text.count(FREEZE_GC_ACCESS_MARKER),
        "ready_marker_count": log_text.count(SGLANG_READY_MARKER),
    }


def wait_for_sglang_warmup(
    log_path: Path,
    process: subprocess.Popen[str],
    timeout_s: float,
) -> dict[str, Any]:
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        if process.poll() is not None:
            raise RuntimeError(
                f"SGLang exited before warmup-complete readiness with code {process.returncode}"
            )
        log_text = log_path.read_text(errors="replace")
        facts = server_request_facts(log_text)
        if facts["ready_marker_count"]:
            return {"elapsed_s": round(time.monotonic() - started, 3), **facts}
        time.sleep(1)
    raise TimeoutError(f"SGLang warmup did not complete in the remaining {timeout_s:.3f}s")


def wait_for_chat_completion_count(
    log_path: Path,
    process: subprocess.Popen[str],
    expected_count: int,
    timeout_s: float = 5,
) -> dict[str, Any]:
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        if process.poll() is not None:
            raise RuntimeError(
                f"SGLang exited before request-count evidence was complete with code {process.returncode}"
            )
        facts = server_request_facts(log_path.read_text(errors="replace"))
        if facts["chat_completion_accesses"] >= expected_count:
            return {"elapsed_s": round(time.monotonic() - started, 3), **facts}
        time.sleep(0.1)
    raise TimeoutError(
        f"server log did not reach {expected_count} successful chat-completion accesses"
    )


# Tycho and the NOOA join persist Solver evidence under their own schema names.
# Both carry the per-sequence Actor-call evidence this reconciliation requires;
# anything else is rejected rather than counted.
SOLVER_SESSION_SCHEMAS = (
    "tycho-nooa-public-solver-sessions-v1",
    "nooa-join-sessions-v4",
)


def expected_launcher_solver_requests(launcher_result: dict[str, Any]) -> int:
    """Derive the exact provider-request count from preserved Solver evidence."""
    sessions = launcher_result.get("solver_sessions")
    if not isinstance(sessions, dict):
        raise TypeError("official launcher omitted Solver session evidence")
    if sessions.get("schema") not in SOLVER_SESSION_SCHEMAS:
        raise RuntimeError("official launcher Solver session schema differs")

    games = sessions.get("games")
    game_count = sessions.get("game_count")
    total_actor_calls = sessions.get("total_actor_calls")
    if (
        not isinstance(games, dict)
        or not games
        or isinstance(game_count, bool)
        or not isinstance(game_count, int)
        or game_count != len(games)
        or isinstance(total_actor_calls, bool)
        or not isinstance(total_actor_calls, int)
        or total_actor_calls <= 0
    ):
        raise RuntimeError("official launcher Solver request totals are invalid")

    reconciled_total = 0
    for game, session in sorted(games.items()):
        if not isinstance(game, str) or not isinstance(session, dict):
            raise TypeError("official launcher Solver game evidence is invalid")
        model_evidence = session.get("model_evidence")
        if not isinstance(model_evidence, dict):
            raise TypeError(f"official launcher omitted model evidence for {game}")
        actor_call_count = model_evidence.get("actor_call_count")
        by_sequence = model_evidence.get("actor_calls_by_sequence")
        if (
            isinstance(actor_call_count, bool)
            or not isinstance(actor_call_count, int)
            or actor_call_count <= 0
            or not isinstance(by_sequence, dict)
            or not by_sequence
            or any(
                not isinstance(sequence, str)
                or not sequence.isdigit()
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
                for sequence, count in by_sequence.items()
            )
            or sum(by_sequence.values()) != actor_call_count
        ):
            raise RuntimeError(
                f"official launcher Actor-call evidence does not reconcile for {game}"
            )
        reconciled_total += actor_call_count

    if reconciled_total != total_actor_calls:
        raise RuntimeError("official launcher aggregate Actor-call total does not reconcile")
    return total_actor_calls


def reconcile_launcher_request_count(
    launcher_result: dict[str, Any],
    warmup_facts: dict[str, Any],
    post_request_facts: dict[str, Any],
) -> dict[str, Any]:
    """Require one SGLang request for every persisted Actor call, with no extras."""
    expected_solver_requests = expected_launcher_solver_requests(launcher_result)
    warmup_requests = warmup_facts.get("chat_completion_accesses")
    warmup_successes = warmup_facts.get("successful_chat_completion_accesses")
    observed_total = post_request_facts.get("chat_completion_accesses")
    observed_successful_total = post_request_facts.get(
        "successful_chat_completion_accesses"
    )
    freeze_gc_accesses = post_request_facts.get("successful_freeze_gc_accesses")
    ready_markers = post_request_facts.get("ready_marker_count")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (
            warmup_requests,
            warmup_successes,
            observed_total,
            observed_successful_total,
            freeze_gc_accesses,
            ready_markers,
        )
    ):
        raise RuntimeError("SGLang request-count evidence is not integral")

    observed_solver_requests = observed_total - warmup_requests
    observed_successful_solver_requests = (
        observed_successful_total - warmup_successes
    )
    facts = {
        "expected_runtime_owned_warmup_requests": 1,
        "observed_runtime_owned_warmup_requests": warmup_requests,
        "observed_successful_runtime_owned_warmup_requests": warmup_successes,
        "expected_launcher_solver_requests": expected_solver_requests,
        "observed_launcher_solver_requests": observed_solver_requests,
        "observed_successful_launcher_solver_requests": (
            observed_successful_solver_requests
        ),
        "expected_total_chat_completion_accesses": 1 + expected_solver_requests,
        "observed_total_chat_completion_accesses": observed_total,
        "observed_total_successful_chat_completion_accesses": (
            observed_successful_total
        ),
        "ready_marker_count": ready_markers,
        "successful_freeze_gc_accesses": freeze_gc_accesses,
        "whole_model_request_count_verified": False,
    }
    if (
        warmup_requests != 1
        or warmup_successes != 1
        or observed_solver_requests != expected_solver_requests
        or observed_successful_solver_requests != expected_solver_requests
        or observed_total != 1 + expected_solver_requests
        or observed_successful_total != observed_total
        or ready_markers != 1
        or freeze_gc_accesses != 1
    ):
        raise RuntimeError(
            f"official-launcher request-count contract did not close: {facts}"
        )
    facts["whole_model_request_count_verified"] = True
    return facts


def sglang_command(
    model_path: Path,
    host: str,
    port: int,
    *,
    attention_backend: str,
    sampling_backend: str | None,
) -> list[str]:
    executable = shutil.which("sglang")
    if not executable:
        raise RuntimeError("sglang executable not found; attach/install the pinned runtime first")
    command = [
        executable,
        "serve",
        "--trust-remote-code",
        "--model-path",
        str(model_path),
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--mem-fraction-static",
        "0.85",
        "--attention-backend",
        attention_backend,
    ]
    if sampling_backend is not None:
        command.extend(["--sampling-backend", sampling_backend])
    command.extend(
        [
        "--chunked-prefill-size",
        "2048",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--host",
        host,
        "--port",
        str(port),
        ]
    )
    return command


def backend_facts(
    log_text: str,
    *,
    attention_backend: str,
    sampling_backend: str | None,
    request_ready: bool,
) -> dict[str, Any]:
    attention_match = re.search(r"'attention_backend': '([^']+)'", log_text)
    sampling_match = re.search(r"'sampling_backend': '([^']+)'", log_text)
    selected_attention_backend = attention_match.group(1) if attention_match else None
    selected_sampling_backend = sampling_match.group(1) if sampling_match else None
    graph_backends = re.findall(
        r"Capture target prefill CUDA graph begin\. backend=([a-zA-Z0-9_-]+)",
        log_text,
    )
    import_lines = [
        line
        for line in log_text.splitlines()
        if "import time:" in line and "flashinfer" in line.lower()
    ]
    flashinfer_runtime_markers = [
        marker
        for marker in (
            "flashinfer_backend.py",
            "/flashinfer/",
            "RuntimeError: FlashInfer",
        )
        if marker in log_text
    ]
    cuda_kernel_markers = [
        marker
        for marker in (
            "CUDA error",
            "RuntimeError: Triton",
            "no kernel image",
            "invalid device function",
        )
        if marker in log_text
    ]
    return {
        "requested_attention_backend": attention_backend,
        "requested_sampling_backend": sampling_backend,
        "selected_attention_backend": selected_attention_backend,
        "selected_sampling_backend": selected_sampling_backend,
        "attention_backend_observed": selected_attention_backend == attention_backend,
        "sampling_backend_observed": selected_sampling_backend is not None,
        "sampling_backend_matches_request": (
            sampling_backend is None or selected_sampling_backend == sampling_backend
        ),
        "flashinfer_import_observed": bool(import_lines),
        "flashinfer_import_evidence": import_lines[:100],
        "flashinfer_runtime_path_observed": bool(flashinfer_runtime_markers),
        "flashinfer_runtime_markers": flashinfer_runtime_markers,
        "prefill_graph_capture_occurred": bool(graph_backends),
        "prefill_graph_capture_backends": graph_backends,
        "cuda_kernel_exception_observed": bool(cuda_kernel_markers),
        "cuda_kernel_exception_markers": cuda_kernel_markers,
        "server_request_ready": request_ready,
    }


def classify_startup_failure(log_text: str, error: Exception) -> str:
    lower = log_text.lower()
    if any(
        marker in lower
        for marker in ("flashinfer_backend.py", "/flashinfer/", "runtimeerror: flashinfer")
    ):
        return "residual_flashinfer_reachability"
    if any(
        marker in lower
        for marker in ("cuda_home", "cuda toolkit", "nvcc", "libcuda", "libcudart")
    ):
        return "cuda_toolkit_discovery"
    if "triton" in lower and any(
        marker in lower
        for marker in ("cuda", "kernel", "ptx", "invalid device function")
    ):
        return "triton_sm120_kernel_incompatibility"
    return f"unrelated_initialization_failure:{type(error).__name__}"


def classify_r8b_failure(stage: str, log_text: str, error: Exception) -> str:
    if isinstance(error, ResponseValidationError):
        return error.classification
    if stage in {"server_startup", "backend_resolution"}:
        return "startup_invalid_after_sampling_binding"
    if stage in {"runtime_warmup", "warmup_request_count", "probe_request_count"}:
        return "request_count_contract_invalid"
    if stage == "request_transport":
        lower = log_text.lower()
        if any(
            marker in lower
            for marker in (
                "cuda error",
                "runtimeerror: triton",
                "no kernel image",
                "invalid device function",
                "traceback (most recent call last)",
            )
        ):
            return "generation_path_failure"
        return "request_serving_failure"
    if stage == "response_decode":
        return "generation_response_protocol_failure"
    return f"pre_request_failure:{stage}:{type(error).__name__}"


def validate_response(response: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ResponseValidationError(
            "generation_response_protocol_failure",
            "response body is not a JSON object",
        )
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ResponseValidationError(
            "generation_response_protocol_failure",
            "response must contain exactly one choice",
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ResponseValidationError(
            "generation_response_protocol_failure",
            "response choice is not an object",
        )
    message = choice.get("message", {})
    if not isinstance(message, dict):
        raise ResponseValidationError(
            "generation_response_protocol_failure",
            "response choice has no message object",
        )
    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ResponseValidationError(
            "parser_contract_failure",
            "reasoning parser produced no non-empty reasoning_content",
        )
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        content = message.get("content")
        upper_content = content.upper() if isinstance(content, str) else ""
        classification = "multimodal_contract_failure"
        if "TAKE_ACTIONS" in upper_content or "<TOOL" in upper_content:
            classification = "parser_contract_failure"
        elif "LEFT" in upper_content:
            classification = "tool_generation_failure"
        raise ResponseValidationError(
            classification,
            "tool parser did not produce exactly one native tool call",
        )
    tool_call = tool_calls[0]
    if not isinstance(tool_call, dict):
        raise ResponseValidationError(
            "parser_contract_failure",
            "native tool call is not an object",
        )
    function = tool_call.get("function", {})
    if not isinstance(function, dict):
        raise ResponseValidationError(
            "parser_contract_failure",
            "native tool call has no function object",
        )
    if function.get("name") != "take_actions":
        raise ResponseValidationError(
            "parser_contract_failure",
            f"unexpected tool name: {function.get('name')!r}",
        )
    raw_arguments = function.get("arguments")
    if not isinstance(raw_arguments, str):
        raise ResponseValidationError(
            "parser_contract_failure",
            "tool arguments are not a JSON string",
        )
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        raise ResponseValidationError(
            "parser_contract_failure",
            f"tool arguments are not valid JSON: {error}",
        ) from error
    if arguments != {
        "actions": [{"action": "LEFT"}],
        "rationale": "red is left of yellow",
    }:
        raise ResponseValidationError(
            "multimodal_contract_failure",
            f"vision/tool result was not the expected LEFT action: {arguments}",
        )
    if choice.get("finish_reason") != "tool_calls":
        raise ResponseValidationError(
            "parser_contract_failure",
            f"unexpected finish reason: {choice.get('finish_reason')!r}",
        )
    return {
        "reasoning_chars": len(reasoning),
        "tool_name": function["name"],
        "arguments": arguments,
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage"),
    }


class FakeOpenAIHandler(http.server.BaseHTTPRequestHandler):
    request_payload: dict[str, Any] | None = None

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send(self, value: Any) -> None:
        encoded = json.dumps(value).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._send({"object": "list", "data": [{"id": "self-test-model"}]})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        type(self).request_payload = json.loads(self.rfile.read(length))
        self._send(
            {
                "id": "self-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "The red square is visibly left of the yellow square.",
                            "tool_calls": [
                                {
                                    "id": "call-self-test",
                                    "type": "function",
                                    "function": {
                                        "name": "take_actions",
                                        "arguments": (
                                            "{\"actions\":[{\"action\":\"LEFT\"}],"
                                            "\"rationale\":\"red is left of yellow\"}"
                                        ),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )


def start_fake_server() -> tuple[http.server.ThreadingHTTPServer, threading.Thread, str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address)
    return server, thread, f"http://{host}:{port}"


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    started = time.monotonic()
    events: list[str] = []
    request_probe_mode = args.r8_b or args.r8_b2 or args.r8_l3b
    mode = (
        "self-test"
        if args.self_test
        else "s0-official-launcher"
        if args.official_launcher_probe
        else "r8-a"
        if args.readiness_only
        else "r8-l3b"
        if args.r8_l3b
        else "r8-b2"
        if args.r8_b2
        else "r8-b"
        if args.r8_b
        else "phase-a"
    )
    stage = "preflight"

    def event(message: str) -> None:
        events.append(f"{utc_now()} {message}")
        (output_dir / "events.log").write_text("\n".join(events) + "\n")
        print(message, flush=True)

    if args.official_launcher_probe:
        claim_ceiling = (
            "A passing S0 official-launcher join establishes that the pinned official "
            "ARC-AGI-3-Agents main/Swarm loop reached the frozen Solver worker, "
            "Tycho-NOOA actor, LiteLLM transport, and qualified local Qwen/SGLang "
            "server, then satisfied the R8-L3B lifecycle gate. It does not establish "
            "successful scoring, repeated-game reliability, submission acceptance, "
            "or model quality."
        )
    elif args.readiness_only:
        claim_ceiling = (
            "A passing R8-A run establishes only that the bound checkpoint and runtime reached "
            "request readiness on the observed GPU with the recorded backend route. It sends no "
            "model request and establishes no multimodal, tool-call, gameplay, concurrency, "
            "throughput, or hosted/local parity claim."
        )
    elif args.r8_l3b:
        claim_ceiling = (
            "A passing R8-L3B run establishes only that the unchanged R8-B2 request path "
            "completed and the externally observed owned process tree, port listener, GPU "
            "applications, and teardown log satisfied the frozen TERM-only lifecycle contract. "
            "It does not establish gameplay, concurrency, throughput, S0, submission, or "
            "promotion readiness."
        )
    elif args.r8_b2:
        claim_ceiling = (
            "A passing R8-B2 run establishes only that the R8-A2-qualified substrate, with "
            "sampling resolved to PyTorch, completed its one pinned runtime-owned VLM warmup "
            "before accepting exactly one frozen probe-owned visual request and returning one "
            "parser-accepted native tool call. It does not establish clean shutdown, gameplay, "
            "concurrency, throughput, or hosted/local parity."
        )
    elif args.r8_b:
        claim_ceiling = (
            "A passing R8-B run establishes only that the R8-A2-qualified substrate, with "
            "sampling explicitly resolved to PyTorch, completed one frozen visual request and "
            "returned one parser-accepted native tool call. It does not establish clean shutdown, "
            "gameplay, concurrency, throughput, or hosted/local parity."
        )
    else:
        claim_ceiling = (
            "A passing Phase A run establishes only that the bound checkpoint and runtime "
            "cold-loaded on the observed GPU and completed one multimodal native-tool request. "
            "It does not establish gameplay, concurrency, throughput, or hosted/local parity."
        )

    config = {
        "schema": PROBE_SCHEMA,
        "mode": mode,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "manifest": str(args.manifest.resolve()),
        "model_path": str(args.model_path.resolve()) if args.model_path else None,
        "input_root": str(args.input_root.resolve()),
        "model_source": args.model_source,
        "runtime": {
            "repository": SGLANG_REPOSITORY,
            "commit": SGLANG_COMMIT,
            "provenance": str(args.runtime_provenance.resolve())
            if args.runtime_provenance
            else None,
        },
        "startup_timeout_s": args.startup_timeout_s,
        "request_timeout_s": args.request_timeout_s,
        "launcher_timeout_s": args.launcher_timeout_s,
        "attention_backend": args.attention_backend,
        "sampling_backend": args.sampling_backend,
        "profile_imports": args.profile_imports,
        "readiness_only": args.readiness_only,
        "r8_b": args.r8_b,
        "r8_b2": args.r8_b2,
        "r8_l3b": args.r8_l3b,
        "official_launcher_probe": (
            str(args.official_launcher_probe.resolve()) if args.official_launcher_probe else None
        ),
        "launcher_python": (
            str(args.launcher_python.resolve()) if args.launcher_python else None
        ),
        "claim_ceiling": claim_ceiling,
    }
    write_json(output_dir / "probe-config.json", config)
    snapshot = system_snapshot()
    write_json(output_dir / "system-before.json", snapshot)

    server: http.server.ThreadingHTTPServer | None = None
    fake_thread: threading.Thread | None = None
    process: subprocess.Popen[str] | None = None
    launch_target: dict[str, Any] | None = None
    server_log = None
    model_verification: dict[str, Any] | None = None
    runtime_verification: dict[str, Any] | None = None
    warmup_facts: dict[str, Any] | None = None
    request_count_facts: dict[str, Any] | None = None
    result_record: dict[str, Any] | None = None

    try:
        if args.self_test:
            event("starting local fake OpenAI-compatible server")
            server, fake_thread, base_url = start_fake_server()
            model_name = "self-test-model"
        else:
            stage = "accelerator_preflight"
            name = gpu_name(snapshot)
            if EXPECTED_GPU_FRAGMENT not in name:
                raise RuntimeError(
                    f"expected {EXPECTED_GPU_FRAGMENT!r} GPU, observed {name!r}"
                )
            if args.r8_l3b:
                stage = "lifecycle_observer_preflight"
                observer_preflight = lifecycle_observer_preflight(args.port)
                write_json(
                    output_dir / "lifecycle-observer-preflight.json", observer_preflight
                )
                if observer_preflight["status"] != "passed":
                    raise LifecycleObservationError(
                        f"lifecycle observer preflight failed: {observer_preflight['errors']}"
                    )
            if not args.model_source:
                raise RuntimeError("--model-source is required to bind the Kaggle attachment")
            if not args.runtime_provenance:
                raise RuntimeError("--runtime-provenance is required to bind the SGLang artifact")
            stage = "runtime_identity"
            runtime_value = json.loads(args.runtime_provenance.read_text())
            if not isinstance(runtime_value, dict):
                raise RuntimeError("runtime provenance must be one JSON object")
            runtime_verification = runtime_value
            if (
                runtime_verification.get("repository") != SGLANG_REPOSITORY
                or runtime_verification.get("commit") != SGLANG_COMMIT
                or runtime_verification.get("dataset_source") != RUNTIME_DATASET_SOURCE
                or runtime_verification.get("dataset_version") != RUNTIME_DATASET_VERSION
                or runtime_verification.get("base_dataset_source")
                != BASE_RUNTIME_DATASET_SOURCE
                or runtime_verification.get("base_dataset_version")
                != BASE_RUNTIME_DATASET_VERSION
            ):
                raise RuntimeError("runtime provenance does not bind the expected Kaggle/SGLang route")
            artifact_path = Path(str(runtime_verification.get("artifact_path", "")))
            lock_path = Path(str(runtime_verification.get("lock_path", "")))
            manifest_path = Path(str(runtime_verification.get("dataset_manifest_path", "")))
            base_artifact_path = Path(
                str(runtime_verification.get("base_artifact_path", ""))
            )
            base_lock_path = Path(str(runtime_verification.get("base_lock_path", "")))
            runtime_paths = (
                artifact_path,
                lock_path,
                manifest_path,
                base_artifact_path,
                base_lock_path,
            )
            if any(not path.is_file() or path.is_symlink() for path in runtime_paths):
                raise RuntimeError("runtime provenance paths do not name mounted files")
            runtime_verification["actual_artifact_sha256"] = sha256_file(artifact_path)
            runtime_verification["actual_lock_sha256"] = sha256_file(lock_path)
            runtime_verification["actual_dataset_manifest_sha256"] = sha256_file(
                manifest_path
            )
            runtime_verification["actual_base_artifact_sha256"] = sha256_file(
                base_artifact_path
            )
            runtime_verification["actual_base_lock_sha256"] = sha256_file(base_lock_path)
            if (
                runtime_verification["actual_artifact_sha256"] != RUNTIME_WHEEL_SHA256
                or runtime_verification["actual_lock_sha256"] != RUNTIME_LOCK_SHA256
                or runtime_verification["actual_dataset_manifest_sha256"]
                != RUNTIME_DATASET_MANIFEST_SHA256
                or runtime_verification["actual_base_artifact_sha256"]
                != BASE_RUNTIME_WHEEL_SHA256
                or runtime_verification["actual_base_lock_sha256"]
                != RUNTIME_LOCK_SHA256
            ):
                raise RuntimeError("mounted runtime files do not match the pinned Kaggle dataset bytes")

            stage = "model_identity"
            model_path = args.model_path or discover_model(args.input_root, args.manifest)
            event(f"verifying immutable model files at {model_path}")
            model_verification = verify_model(model_path, args.manifest)
            write_json(output_dir / "model-verification.json", model_verification)

            host = "127.0.0.1"
            port = args.port
            base_url = f"http://{host}:{port}"
            command = sglang_command(
                model_path,
                host,
                port,
                attention_backend=args.attention_backend,
                sampling_backend=args.sampling_backend,
            )
            write_json(output_dir / "server-command.json", command)
            stage = "server_startup"
            event("launching pinned SGLang route")
            server_log = (output_dir / "server.log").open("w")
            process_environment = os.environ.copy()
            recorded_environment = {}
            if args.r8_l3b:
                process_environment["R8_L3B_OWNER"] = R8_L3B_OWNER
                recorded_environment["R8_L3B_OWNER"] = R8_L3B_OWNER
            if args.profile_imports:
                process_environment["PYTHONPROFILEIMPORTTIME"] = "1"
                recorded_environment["PYTHONPROFILEIMPORTTIME"] = "1"
            write_json(output_dir / "server-environment.json", recorded_environment)
            process = subprocess.Popen(
                command,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
                env=process_environment,
                start_new_session=args.r8_l3b,
            )
            if args.r8_l3b:
                launch_target = process_identity(process.pid)
                launch_observer = observer_identity()
                launch_record = {
                    "captured_at": utc_now(),
                    "observer": launch_observer,
                    "target": launch_target,
                    "outside_target_pgid": launch_observer["pgid"]
                    != launch_target["pgid"],
                    "outside_target_sid": launch_observer["sid"] != launch_target["sid"],
                }
                write_json(output_dir / "lifecycle-target-launch.json", launch_record)
                if (
                    not launch_record["outside_target_pgid"]
                    or not launch_record["outside_target_sid"]
                    or not launch_target["owner_marker"]
                    or launch_target["pid"] != launch_target["pgid"]
                    or launch_target["pid"] != launch_target["sid"]
                ):
                    raise LifecycleObservationError(
                        f"target launch identity is not externally owned: {launch_record}"
                    )
            startup_wait_started = time.monotonic()
            readiness = wait_for_server(base_url, process, args.startup_timeout_s)
            models = readiness["models"].get("data", [])
            if len(models) != 1 or models[0].get("id") != EXPECTED_SERVED_MODEL_ID:
                raise RuntimeError(f"unexpected /v1/models response: {readiness['models']}")
            model_name = models[0]["id"]
            if args.r8_b2 or args.r8_l3b:
                write_json(output_dir / "model-api-readiness.json", readiness)
                model_api_elapsed_s = readiness["elapsed_s"]
                remaining_startup_s = args.startup_timeout_s - (
                    time.monotonic() - startup_wait_started
                )
                if remaining_startup_s <= 0:
                    raise TimeoutError("no startup budget remained for SGLang warmup")
                stage = "runtime_warmup"
                warmup_facts = wait_for_sglang_warmup(
                    output_dir / "server.log",
                    process,
                    remaining_startup_s,
                )
                stage = "warmup_request_count"
                write_json(output_dir / "warmup-facts.json", warmup_facts)
                if (
                    warmup_facts["chat_completion_accesses"] != 1
                    or warmup_facts["successful_chat_completion_accesses"] != 1
                    or warmup_facts["successful_freeze_gc_accesses"] != 1
                    or warmup_facts["ready_marker_count"] != 1
                ):
                    raise RuntimeError(
                        f"unexpected SGLang warmup evidence before probe request: {warmup_facts}"
                    )
                readiness = {
                    "model_api_elapsed_s": model_api_elapsed_s,
                    "warmup_complete_elapsed_s": round(
                        time.monotonic() - startup_wait_started, 3
                    ),
                    "models": readiness["models"],
                    "warmup": warmup_facts,
                }
                event(
                    "server warmup complete after "
                    f"{readiness['warmup_complete_elapsed_s']}s as {model_name!r}"
                )
            else:
                event(f"server ready after {readiness['elapsed_s']}s as {model_name!r}")
            write_json(output_dir / "readiness.json", readiness)
            server_log.flush()
            log_text = (output_dir / "server.log").read_text(errors="replace")
            stage = "backend_resolution"
            facts = backend_facts(
                log_text,
                attention_backend=args.attention_backend,
                sampling_backend=args.sampling_backend,
                request_ready=True,
            )
            write_json(output_dir / "backend-facts.json", facts)
            if request_probe_mode:
                if facts["selected_attention_backend"] != "triton":
                    raise RuntimeError("attention backend did not resolve to Triton")
                if facts["selected_sampling_backend"] != "pytorch":
                    raise RuntimeError("sampling backend did not resolve to PyTorch")
            if args.readiness_only:
                if not facts["attention_backend_observed"]:
                    raise RuntimeError("server log did not confirm the requested attention backend")
                if not facts["sampling_backend_observed"]:
                    raise RuntimeError("server log did not confirm the requested sampling backend")
                if facts["flashinfer_runtime_path_observed"]:
                    raise RuntimeError("server reached the rejected FlashInfer runtime path")
                if facts["cuda_kernel_exception_observed"]:
                    raise RuntimeError("server logged a CUDA/kernel exception before readiness")
                result = {
                    "schema": PROBE_SCHEMA,
                    "status": "passed",
                    "mode": mode,
                    "started_at": started_at,
                    "ended_at": utc_now(),
                    "elapsed_s": round(time.monotonic() - started, 3),
                    "model_verification": cast(dict[str, Any], model_verification)["status"],
                    "runtime_verification": runtime_verification,
                    "backend_facts": facts,
                    "readiness": readiness,
                    "claim_ceiling": config["claim_ceiling"],
                }
                write_json(output_dir / "result.json", result)
                event("R8-A readiness probe passed")
                return result

        if args.official_launcher_probe:
            if process is None or warmup_facts is None or args.launcher_python is None:
                raise RuntimeError(
                    "official launcher reached request stage without its live route"
                )
            stage = "official_launcher"
            launcher_result_path = output_dir / "official-launcher-result.json"
            solver_workspace = output_dir / "solver-workspaces"
            completed = subprocess.run(
                [
                    str(args.launcher_python),
                    str(args.official_launcher_probe),
                    "--base-url",
                    base_url.rstrip("/") + "/v1",
                    "--model",
                    model_name,
                    "--workspace-root",
                    str(solver_workspace),
                    "--output",
                    str(launcher_result_path),
                    "--timeout-s",
                    str(args.launcher_timeout_s - 60),
                ],
                capture_output=True,
                text=True,
                timeout=args.launcher_timeout_s,
                check=False,
                env={
                    **os.environ,
                    "TYCHO_KAGGLE_HARDENED_HOST": "1",
                    "TYCHO_SANDBOX_RUNTIME": "host",
                },
            )
            write_json(
                output_dir / "official-launcher-command.json",
                {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
            )
            if completed.returncode != 0 or not launcher_result_path.is_file():
                raise RuntimeError(
                    "official launcher command failed: "
                    f"rc={completed.returncode}, stderr={completed.stderr[-2000:]}"
                )
            launcher_result = json.loads(launcher_result_path.read_text())
            if launcher_result.get("status") != "passed":
                raise RuntimeError(
                    f"official launcher did not report success: {launcher_result}"
                )
            post_request_facts = wait_for_chat_completion_count(
                output_dir / "server.log", process, expected_count=2
            )
            request_count_facts = reconcile_launcher_request_count(
                launcher_result,
                warmup_facts,
                post_request_facts,
            )
            write_json(output_dir / "request-count-facts.json", request_count_facts)
            result = {
                "schema": PROBE_SCHEMA,
                "status": "passed",
                "mode": mode,
                "started_at": started_at,
                "ended_at": utc_now(),
                "elapsed_s": round(time.monotonic() - started, 3),
                "model_verification": cast(dict[str, Any], model_verification)[
                    "status"
                ],
                "runtime_verification": runtime_verification,
                "official_launcher": launcher_result,
                "warmup_facts": warmup_facts,
                "request_count_facts": request_count_facts,
                "claim_ceiling": config["claim_ceiling"],
            }
            result_record = result
            write_json(output_dir / "result.json", result)
            event("official launcher passed")
            return result

        stage = "request_freeze"
        image = arc_like_png()
        (output_dir / "arc-like-probe.png").write_bytes(image)
        request = build_request(model_name, image)
        request_bytes = json.dumps(request).encode("utf-8")
        write_json(output_dir / "request.json", request)
        (output_dir / "request.bin").write_bytes(request_bytes)
        request_identity = {
            "image_sha256": sha256_bytes(image),
            "request_bytes": len(request_bytes),
            "request_sha256": sha256_bytes(request_bytes),
            "served_model_id": model_name,
        }
        write_json(output_dir / "request-identity.json", request_identity)
        if request_probe_mode and not args.self_test:
            expected_request_identity = {
                "image_sha256": EXPECTED_IMAGE_SHA256,
                "request_bytes": EXPECTED_REQUEST_BYTES,
                "request_sha256": EXPECTED_REQUEST_SHA256,
                "served_model_id": EXPECTED_SERVED_MODEL_ID,
            }
            if request_identity != expected_request_identity:
                raise RuntimeError(
                    f"runtime request bytes do not match the frozen {mode} request identity"
                )
        event("sending one multimodal native-tool request")
        stage = "request_transport"
        request_started = time.monotonic()
        transport = post_bytes(
            base_url.rstrip("/") + "/v1/chat/completions",
            request_bytes,
            args.request_timeout_s,
        )
        request_elapsed = round(time.monotonic() - request_started, 3)
        response_bytes = transport.pop("body")
        (output_dir / "response.bin").write_bytes(response_bytes)
        write_json(output_dir / "response-transport.json", transport)
        if transport["status"] != 200:
            raise RuntimeError(
                f"completion endpoint returned HTTP {transport['status']}; "
                "raw response bytes were preserved"
            )
        if (args.r8_b2 or args.r8_l3b) and not args.self_test:
            if process is None or warmup_facts is None:
                raise RuntimeError("R8-B2 request completed without warmup evidence")
            stage = "probe_request_count"
            post_request_facts = wait_for_chat_completion_count(
                output_dir / "server.log",
                process,
                expected_count=2,
            )
            request_count_facts = {
                "expected_runtime_owned_warmup_requests": 1,
                "observed_runtime_owned_warmup_requests": warmup_facts[
                    "chat_completion_accesses"
                ],
                "observed_successful_runtime_owned_warmup_requests": warmup_facts[
                    "successful_chat_completion_accesses"
                ],
                "expected_probe_owned_requests": 1,
                "observed_probe_owned_requests": (
                    post_request_facts["chat_completion_accesses"]
                    - warmup_facts["chat_completion_accesses"]
                ),
                "observed_successful_probe_owned_requests": (
                    post_request_facts["successful_chat_completion_accesses"]
                    - warmup_facts["successful_chat_completion_accesses"]
                ),
                "expected_total_chat_completion_accesses": 2,
                "observed_total_chat_completion_accesses": post_request_facts[
                    "chat_completion_accesses"
                ],
                "expected_total_successful_chat_completion_accesses": 2,
                "observed_total_successful_chat_completion_accesses": post_request_facts[
                    "successful_chat_completion_accesses"
                ],
                "ready_marker_count": post_request_facts["ready_marker_count"],
                "successful_freeze_gc_accesses": post_request_facts[
                    "successful_freeze_gc_accesses"
                ],
                "evidence_wait_elapsed_s": post_request_facts["elapsed_s"],
            }
            write_json(output_dir / "request-count-facts.json", request_count_facts)
            if (
                request_count_facts["observed_runtime_owned_warmup_requests"] != 1
                or request_count_facts[
                    "observed_successful_runtime_owned_warmup_requests"
                ]
                != 1
                or request_count_facts["observed_probe_owned_requests"] != 1
                or request_count_facts["observed_successful_probe_owned_requests"] != 1
                or request_count_facts["observed_total_chat_completion_accesses"] != 2
                or request_count_facts[
                    "observed_total_successful_chat_completion_accesses"
                ]
                != 2
                or request_count_facts["ready_marker_count"] != 1
                or request_count_facts["successful_freeze_gc_accesses"] != 1
            ):
                raise RuntimeError(
                    f"R8-B2 request-count contract did not close: {request_count_facts}"
                )
        stage = "response_decode"
        response = json.loads(response_bytes)
        write_json(output_dir / "response.json", response)
        stage = "parser_acceptance"
        validation = validate_response(response)
        validation["request_elapsed_s"] = request_elapsed
        write_json(output_dir / "response-validation.json", validation)

        if args.self_test:
            received = FakeOpenAIHandler.request_payload
            if received is None:
                raise RuntimeError("self-test server did not receive the request")
            content = received["messages"][1]["content"]
            if not any(
                part.get("type") == "image_url"
                and part.get("image_url", {}).get("url", "").startswith("data:image/png;base64,")
                for part in content
            ):
                raise RuntimeError("self-test request omitted the inline image")
            if received.get("tools") != request["tools"]:
                raise RuntimeError("self-test request changed the tool schema")

        if not args.self_test and model_verification is None:
            raise RuntimeError("real probe completed without model verification")
        result = {
            "schema": PROBE_SCHEMA,
            "status": "passed",
            "mode": mode,
            "started_at": started_at,
            "ended_at": utc_now(),
            "elapsed_s": round(time.monotonic() - started, 3),
            "model_verification": (
                "not_applicable_self_test"
                if args.self_test
                else cast(dict[str, Any], model_verification)["status"]
            ),
            "runtime_verification": (
                "not_applicable_self_test" if args.self_test else runtime_verification
            ),
            "response_validation": validation,
            "request_identity": request_identity,
            "claim_ceiling": (
                "The probe harness generated, sent, captured, and validated its image/tool "
                "request against a local fake server; no model, GPU, SGLang, or Kaggle claim."
                if args.self_test
                else config["claim_ceiling"]
            ),
        }
        if (args.r8_b2 or args.r8_l3b) and not args.self_test:
            result["warmup_facts"] = warmup_facts
            result["request_count_facts"] = request_count_facts
        result_record = result
        write_json(output_dir / "result.json", result)
        event("probe passed")
        return result
    except Exception as error:
        log_text = ""
        if server_log is not None:
            server_log.flush()
            log_text = (output_dir / "server.log").read_text(errors="replace")
        if not args.self_test:
            facts_path = output_dir / "backend-facts.json"
            facts = (
                json.loads(facts_path.read_text())
                if facts_path.is_file()
                else backend_facts(
                    log_text,
                    attention_backend=args.attention_backend,
                    sampling_backend=args.sampling_backend,
                    request_ready=False,
                )
            )
            facts["failure_classification"] = (
                classify_r8b_failure(stage, log_text, error)
                if request_probe_mode
                else classify_startup_failure(log_text, error)
            )
            write_json(output_dir / "backend-facts.json", facts)
        failure_classification = (
            None
            if args.self_test
            else classify_r8b_failure(stage, log_text, error)
            if request_probe_mode
            else classify_startup_failure(log_text, error)
        )
        failure = {
            "schema": PROBE_SCHEMA,
            "status": (
                "invalid"
                if failure_classification == "startup_invalid_after_sampling_binding"
                or failure_classification == "request_count_contract_invalid"
                or (
                    isinstance(failure_classification, str)
                    and failure_classification.startswith("pre_request_failure:")
                )
                else "failed"
            ),
            "mode": mode,
            "started_at": started_at,
            "ended_at": utc_now(),
            "elapsed_s": round(time.monotonic() - started, 3),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "failure_stage": stage,
            "failure_classification": failure_classification,
            "claim_ceiling": "The run failed and establishes no deployment feasibility claim.",
        }
        result_record = failure
        write_json(output_dir / "result.json", failure)
        event(f"probe failed: {type(error).__name__}: {error}")
        raise
    finally:
        shutdown: dict[str, Any] = {
            "process_started": process is not None,
            "method": None,
            "returncode": None,
        }
        if args.r8_l3b and process is not None:
            try:
                lifecycle = qualify_lifecycle(process, server_log, output_dir, port=args.port)
            except Exception as lifecycle_error:
                lifecycle = {
                    "status": "invalid",
                    "failure_classification": "observer_contract_invalid",
                    "error_type": type(lifecycle_error).__name__,
                    "error": str(lifecycle_error),
                    "traceback": traceback.format_exc(),
                }
                write_json(output_dir / "shutdown.json", lifecycle)
                if process.poll() is None:
                    emergency_signals = emergency_contain_process_group(
                        process, launch_target
                    )
                    write_json(output_dir / "lifecycle-signals.json", emergency_signals)
            shutdown = lifecycle
            if result_record is not None:
                result_record = write_r8l3b_result(
                    output_dir, result_record, lifecycle
                )
        else:
            if process is not None and process.poll() is None:
                shutdown["method"] = "terminate"
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    shutdown["method"] = "kill_after_terminate_timeout"
                    process.kill()
                    process.wait(timeout=10)
            if process is not None:
                shutdown["returncode"] = process.returncode
            elif args.r8_l3b:
                shutdown.update(
                    {
                        "status": "invalid",
                        "failure_classification": "target_not_started",
                    }
                )
        if server_log is not None:
            server_log.close()
        if server is not None:
            server.shutdown()
            server.server_close()
        if fake_thread is not None:
            fake_thread.join(timeout=5)
        if not args.r8_l3b or process is None:
            write_json(output_dir / "shutdown.json", shutdown)
        write_json(output_dir / "system-after.json", system_snapshot())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--model-source", help="Exact Kaggle model source/version string")
    parser.add_argument("--runtime-provenance", type=Path)
    parser.add_argument("--startup-timeout-s", type=int, default=1200)
    parser.add_argument("--request-timeout-s", type=int, default=180)
    parser.add_argument("--launcher-timeout-s", type=int, default=32400)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--attention-backend", default="flashinfer")
    parser.add_argument("--sampling-backend")
    parser.add_argument("--profile-imports", action="store_true")
    parser.add_argument("--readiness-only", action="store_true")
    parser.add_argument("--r8-b", action="store_true")
    parser.add_argument("--r8-b2", action="store_true")
    parser.add_argument("--r8-l3b", action="store_true")
    parser.add_argument("--official-launcher-probe", type=Path)
    parser.add_argument("--launcher-python", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test and args.readiness_only:
        parser.error("--self-test and --readiness-only are mutually exclusive")
    if args.self_test and (args.r8_b or args.r8_b2 or args.r8_l3b):
        parser.error("--self-test and request-probe modes are mutually exclusive")
    if (args.r8_b or args.r8_b2 or args.r8_l3b) and args.readiness_only:
        parser.error("request-probe modes and --readiness-only are mutually exclusive")
    if sum((args.r8_b, args.r8_b2, args.r8_l3b)) > 1:
        parser.error("--r8-b, --r8-b2, and --r8-l3b are mutually exclusive")
    if (args.r8_b or args.r8_b2 or args.r8_l3b) and (
        args.attention_backend != "triton" or args.sampling_backend != "pytorch"
    ):
        parser.error(
            "request-probe modes require --attention-backend triton "
            "--sampling-backend pytorch"
        )
    if bool(args.official_launcher_probe) != bool(args.launcher_python):
        parser.error("--official-launcher-probe and --launcher-python are required together")
    if args.official_launcher_probe and not args.r8_l3b:
        parser.error("--official-launcher-probe requires --r8-l3b lifecycle ownership")
    if args.r8_l3b and args.port != R8_L3B_PORT:
        parser.error(f"--r8-l3b requires --port {R8_L3B_PORT}")
    if args.r8_l3b and (
        args.startup_timeout_s != 1200 or args.request_timeout_s != 180
    ):
        parser.error(
            "--r8-l3b requires --startup-timeout-s 1200 --request-timeout-s 180"
        )
    if args.launcher_timeout_s <= 0:
        parser.error("--launcher-timeout-s must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        result = run_probe(args)
    except Exception:
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
