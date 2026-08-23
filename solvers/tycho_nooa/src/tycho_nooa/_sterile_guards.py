# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed exec-worker guard set adapted from pinned NOOA sandbox guards."""

from __future__ import annotations

import ctypes
import os
import platform
import resource
import stat
import struct
import sys
from dataclasses import dataclass

_NR_LANDLOCK_CREATE_RULESET = 444
_NR_LANDLOCK_ADD_RULE = 445
_NR_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
_LANDLOCK_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38
_SECCOMP_SET_MODE_FILTER = 1

_ARCH = platform.machine()
if _ARCH == "x86_64":
    _NR_SECCOMP, _AUDIT_ARCH, _NR_SOCKET = 317, 0xC000003E, 41
elif _ARCH in ("aarch64", "arm64"):
    _NR_SECCOMP, _AUDIT_ARCH, _NR_SOCKET = 277, 0xC00000B7, 198
else:  # pragma: no cover - the treatment currently targets Linux hosts
    _NR_SECCOMP, _AUDIT_ARCH, _NR_SOCKET = -1, 0, -1

_FS = {
    "EXECUTE": 1 << 0,
    "WRITE_FILE": 1 << 1,
    "READ_FILE": 1 << 2,
    "READ_DIR": 1 << 3,
    "REMOVE_DIR": 1 << 4,
    "REMOVE_FILE": 1 << 5,
    "MAKE_CHAR": 1 << 6,
    "MAKE_DIR": 1 << 7,
    "MAKE_REG": 1 << 8,
    "MAKE_SOCK": 1 << 9,
    "MAKE_FIFO": 1 << 10,
    "MAKE_BLOCK": 1 << 11,
    "MAKE_SYM": 1 << 12,
    "REFER": 1 << 13,
    "TRUNCATE": 1 << 14,
    "IOCTL_DEV": 1 << 15,
}
_FS_READ = _FS["READ_FILE"] | _FS["READ_DIR"] | _FS["EXECUTE"]
_FS_WRITE = (
    _FS["WRITE_FILE"]
    | _FS["REMOVE_DIR"]
    | _FS["REMOVE_FILE"]
    | _FS["MAKE_CHAR"]
    | _FS["MAKE_DIR"]
    | _FS["MAKE_REG"]
    | _FS["MAKE_SOCK"]
    | _FS["MAKE_FIFO"]
    | _FS["MAKE_BLOCK"]
    | _FS["MAKE_SYM"]
    | _FS["REFER"]
    | _FS["TRUNCATE"]
)
_FS_FILE_MASK = _FS["EXECUTE"] | _FS["READ_FILE"] | _FS["WRITE_FILE"] | _FS["TRUNCATE"]


class _RulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
    ]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


@dataclass(frozen=True)
class _Rule:
    path: str
    write: bool
    required: bool = False


def install(  # pragma: no cover - exercised only through the exec process seam
    *,
    scratch: str,
    memory_headroom_mb: int,
    cpu_seconds: int,
    file_size_bytes: int,
    open_files: int,
) -> None:
    """Install every guard or fail before acknowledging worker readiness."""
    _lower_limit(resource.RLIMIT_FSIZE, file_size_bytes)
    _lower_limit(resource.RLIMIT_NOFILE, open_files)
    if hasattr(resource, "RLIMIT_NPROC"):
        _lower_limit(resource.RLIMIT_NPROC, 0)
    _apply_memory_and_cpu(memory_headroom_mb, cpu_seconds)
    _apply_landlock(_rules(scratch))
    _apply_seccomp_no_sockets()


def _rules(  # pragma: no cover - exercised only through the exec process seam
    scratch: str,
) -> list[_Rule]:
    paths: dict[str, tuple[bool, bool]] = {}

    def add(  # pragma: no cover - exercised only through the exec process seam
        path: str, *, write: bool = False, required: bool = False
    ) -> None:
        normalized = os.path.realpath(path)
        prior_write, prior_required = paths.get(normalized, (False, False))
        paths[normalized] = (prior_write or write, prior_required or required)

    for path in ("/usr", "/bin", "/lib", "/lib64", sys.base_prefix):
        add(path)
    add("/dev/null", write=True)
    add(scratch, write=True, required=True)
    return [
        _Rule(path=path, write=write, required=required)
        for path, (write, required) in sorted(paths.items())
    ]


def _lower_limit(  # pragma: no cover - exercised only through the exec process seam
    kind: int, value: int
) -> None:
    _, hard = resource.getrlimit(kind)
    limit = value if hard == resource.RLIM_INFINITY else min(value, hard)
    resource.setrlimit(kind, (limit, limit))


def _virtual_size_bytes(  # pragma: no cover - exercised only through the exec process seam
) -> int:
    try:
        with open("/proc/self/status", encoding="utf-8", errors="replace") as status:
            for line in status:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return 0
    return 0


def _apply_memory_and_cpu(  # pragma: no cover - exercised only through the exec process seam
    memory_headroom_mb: int, cpu_seconds: int
) -> None:
    _lower_limit(
        resource.RLIMIT_AS,
        _virtual_size_bytes() + memory_headroom_mb * 1024 * 1024,
    )
    _lower_limit(resource.RLIMIT_CPU, cpu_seconds)


def _libc() -> ctypes.CDLL:  # pragma: no cover - exec process seam
    return ctypes.CDLL(None, use_errno=True)


def _landlock_abi() -> int:  # pragma: no cover - exec process seam
    result = _libc().syscall(
        _NR_LANDLOCK_CREATE_RULESET,
        None,
        0,
        _LANDLOCK_CREATE_RULESET_VERSION,
    )
    return int(result) if result and result > 0 else 0


def _abi_fs_mask(abi: int) -> int:  # pragma: no cover - exec process seam
    mask = (1 << 13) - 1
    if abi >= 2:
        mask |= _FS["REFER"]
    if abi >= 3:
        mask |= _FS["TRUNCATE"]
    if abi >= 5:
        mask |= _FS["IOCTL_DEV"]
    return mask


def _install_no_new_privileges() -> None:  # pragma: no cover - exec process seam
    if _libc().prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")


def _apply_landlock(  # pragma: no cover - exercised only through the exec process seam
    rules: list[_Rule],
) -> None:
    abi = _landlock_abi()
    if abi < 1:
        raise OSError("Landlock is unavailable")
    mask = _abi_fs_mask(abi)
    libc = _libc()
    ruleset_fd = _create_landlock_ruleset(libc, mask)
    try:
        for rule in rules:
            _add_landlock_rule(libc, ruleset_fd, mask, rule)
        _install_no_new_privileges()
        if libc.syscall(_NR_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    finally:
        os.close(ruleset_fd)


def _create_landlock_ruleset(  # pragma: no cover - exec process seam
    libc: ctypes.CDLL, mask: int
) -> int:
    attr = _RulesetAttr(handled_access_fs=mask, handled_access_net=0)
    ruleset_fd = libc.syscall(
        _NR_LANDLOCK_CREATE_RULESET, ctypes.byref(attr), ctypes.sizeof(attr), 0
    )
    if ruleset_fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")
    return int(ruleset_fd)


def _add_landlock_rule(  # pragma: no cover - exec process seam
    libc: ctypes.CDLL, ruleset_fd: int, mask: int, rule: _Rule
) -> None:
    path_fd = _open_rule_path(rule)
    if path_fd is None:
        return
    try:
        access = _rule_access(path_fd, rule) & mask
        beneath = _PathBeneathAttr(allowed_access=access, parent_fd=path_fd)
        result = libc.syscall(
            _NR_LANDLOCK_ADD_RULE,
            ruleset_fd,
            _LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(beneath),
            0,
        )
        if result != 0:
            raise OSError(ctypes.get_errno(), f"landlock_add_rule failed: {rule.path}")
    finally:
        os.close(path_fd)


def _open_rule_path(  # pragma: no cover - exec process seam
    rule: _Rule,
) -> int | None:
    try:
        return os.open(rule.path, os.O_PATH | os.O_CLOEXEC)
    except OSError as error:
        if rule.required:
            raise OSError(
                error.errno, f"required sandbox path is unavailable: {rule.path}"
            ) from error
        return None


def _rule_access(  # pragma: no cover - exec process seam
    path_fd: int, rule: _Rule
) -> int:
    access = _FS_READ | (_FS_WRITE if rule.write else 0)
    return access if stat.S_ISDIR(os.fstat(path_fd).st_mode) else access & _FS_FILE_MASK


def _seccomp_program() -> bytes:  # pragma: no cover - exec process seam
    load, jump, ret = 0x00, 0x05, 0x06
    word, absolute, equal, constant = 0x00, 0x20, 0x10, 0x00
    allow, errno_result, access_denied = 0x7FFF0000, 0x00050000, 13

    def instruction(  # pragma: no cover - exec process seam
        code: int, yes: int, no: int, value: int
    ) -> bytes:
        return struct.pack("HBBI", code, yes, no, value)

    return b"".join(
        [
            instruction(load | word | absolute, 0, 0, 4),
            instruction(jump | equal | constant, 1, 0, _AUDIT_ARCH),
            instruction(ret | constant, 0, 0, allow),
            instruction(load | word | absolute, 0, 0, 0),
            instruction(jump | equal | constant, 0, 1, _NR_SOCKET),
            instruction(ret | constant, 0, 0, errno_result | access_denied),
            instruction(ret | constant, 0, 0, allow),
        ]
    )


def _apply_seccomp_no_sockets() -> None:  # pragma: no cover - exec process seam
    if _NR_SECCOMP < 0:
        raise OSError(f"unsupported sandbox architecture: {_ARCH}")
    program = _seccomp_program()
    buffer = ctypes.create_string_buffer(program, len(program))
    descriptor = _SockFprog(
        len=len(program) // 8,
        filter=ctypes.cast(buffer, ctypes.c_void_p).value,
    )
    _install_no_new_privileges()
    if (
        _libc().syscall(
            _NR_SECCOMP,
            _SECCOMP_SET_MODE_FILTER,
            0,
            ctypes.byref(descriptor),
        )
        != 0
    ):
        raise OSError(ctypes.get_errno(), "seccomp installation failed")
