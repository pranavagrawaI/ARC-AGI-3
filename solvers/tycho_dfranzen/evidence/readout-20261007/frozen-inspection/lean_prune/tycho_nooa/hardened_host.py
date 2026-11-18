"""Kaggle S0 runner for model-authored Python under a unique Unix identity.

This is an explicitly selected diagnostic boundary, not a general host sandbox.
The trusted worker stays privileged; this runner prepares one game's Unix-owned
workspace, drops all real/effective/saved identities, installs hard limits and a
seccomp denylist, and only then evaluates the already-opened Python command.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
import platform
import resource
import runpy
import sys
from pathlib import Path

HARDENED_HOST_ENV = "TYCHO_KAGGLE_HARDENED_HOST"
GAME_UID_ENV = "TYCHO_KAGGLE_GAME_UID"
GAME_GID_ENV = "TYCHO_KAGGLE_GAME_GID"

PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2
SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_ERRNO = 0x00050000
AUDIT_ARCH_X86_64 = 0xC000003E

_LIBC = ctypes.CDLL(None, use_errno=True)


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ushort),
        ("filters", ctypes.POINTER(_SockFilter)),
    ]


def enabled(environment: dict[str, str] | None = None) -> bool:
    values = os.environ if environment is None else environment
    return values.get(HARDENED_HOST_ENV) == "1"


def identity(environment: dict[str, str] | None = None) -> tuple[int, int]:
    values = os.environ if environment is None else environment
    try:
        uid = int(values[GAME_UID_ENV])
        gid = int(values[GAME_GID_ENV])
    except (KeyError, ValueError) as error:
        raise RuntimeError(
            f"{HARDENED_HOST_ENV}=1 requires numeric {GAME_UID_ENV} and {GAME_GID_ENV}"
        ) from error
    if uid <= 0 or gid <= 0:
        raise RuntimeError("hardened-host UID and GID must be positive and non-root")
    return uid, gid


def command(
    workspace: Path,
    protected_paths: tuple[Path, ...],
    python_command: list[str],
    *,
    workspace_read_only: bool,
    generated_source: Path | None,
) -> list[str]:
    """Wrap one upstream host-Python command in the trusted boundary runner."""
    uid, gid = identity()
    translated = list(python_command)
    if generated_source is not None:
        translated = [
            argument.replace(
                "/tycho-generated/source.py", str(generated_source.resolve(strict=True))
            )
            for argument in translated
        ]
    result = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve(strict=True)),
        "--workspace",
        str(workspace),
        "--uid",
        str(uid),
        "--gid",
        str(gid),
        "--workspace-mode",
        "read-only" if workspace_read_only else "writable",
    ]
    for path in protected_paths:
        result.extend(("--protected", str(path)))
    if generated_source is not None:
        result.extend(("--generated-source", str(generated_source)))
    return [*result, "--", *translated]


def prepare_tree(
    workspace: Path,
    *,
    gid: int,
    writable: bool,
    protected_paths: tuple[Path, ...] = (),
    generated_source: Path | None = None,
) -> None:
    """Give one game group access without making any path world-visible."""
    root = workspace.resolve(strict=True)
    protected = {path.resolve(strict=True) for path in protected_paths}
    writable_root = root / "notes"
    if writable:
        writable_root.mkdir(exist_ok=True)
    for path in (root, *sorted(root.rglob("*"))):
        if path.is_symlink():
            raise RuntimeError(f"hardened-host workspace contains a symlink: {path}")
        resolved = path.resolve(strict=True)
        is_protected = resolved in protected
        is_writable = writable and (
            resolved == writable_root or writable_root in resolved.parents
        )
        os.chown(resolved, 0, gid, follow_symlinks=False)
        if path.is_dir():
            os.chmod(path, 0o2770 if is_writable and not is_protected else 0o2750)
        elif path.is_file():
            os.chmod(path, 0o660 if is_writable and not is_protected else 0o640)
    if generated_source is not None:
        source = generated_source.resolve(strict=True)
        os.chown(source.parent, 0, gid)
        os.chmod(source.parent, 0o750)
        os.chown(source, 0, gid)
        os.chmod(source, 0o640)


def install_limits() -> dict[str, tuple[int, int]]:
    limits = {
        "RLIMIT_AS": (1 << 30, 1 << 30),
        "RLIMIT_CORE": (0, 0),
        "RLIMIT_CPU": (120, 120),
        "RLIMIT_FSIZE": (16 << 20, 16 << 20),
        "RLIMIT_NOFILE": (256, 256),
    }
    for name, value in limits.items():
        resource.setrlimit(getattr(resource, name), value)
    return limits


def install_seccomp() -> None:
    if platform.machine() != "x86_64":
        raise RuntimeError("hardened host currently supports only x86_64")
    denied = sorted(
        {
            41, 42, 43, 44, 45, 46, 47, 49, 50, 53, 56, 57, 58, 59, 62,
            101, 105, 106, 113, 114, 116, 117, 119, 126, 155, 157, 161, 165,
            166, 169, 170, 171, 175, 176, 200, 234, 246, 248, 249, 250, 272,
            288, 298, 299, 300, 303, 304, 308, 310, 311, 317, 321, 322, 323,
            424, 425, 426, 427, 435, 438,
        }
    )
    instructions: list[tuple[int, int, int, int]] = [
        (0x20, 0, 0, 4),
        (0x15, 1, 0, AUDIT_ARCH_X86_64),
        (0x06, 0, 0, 0x80000000),
        (0x20, 0, 0, 0),
    ]
    for number in denied:
        instructions.extend(
            (
                (0x15, 0, 1, number),
                (0x06, 0, 0, SECCOMP_RET_ERRNO | errno.EPERM),
            )
        )
    instructions.append((0x06, 0, 0, SECCOMP_RET_ALLOW))
    filters_type = _SockFilter * len(instructions)
    filters = filters_type(*(_SockFilter(*item) for item in instructions))
    program = _SockFprog(len(instructions), filters)
    _prctl(PR_SET_NO_NEW_PRIVS, 1)
    _prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(program))


def drop_identity(uid: int, gid: int) -> None:
    if os.geteuid() != 0:
        raise RuntimeError("hardened-host runner must start as root")
    os.setgroups([])
    os.setresgid(gid, gid, gid)
    os.setresuid(uid, uid, uid)


def seal_standard_input() -> None:
    """Do not let generated code consume the worker's protocol input pipe."""
    descriptor = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.dup2(descriptor, 0)
    finally:
        os.close(descriptor)


def run_python_command(arguments: list[str]) -> None:
    if not arguments:
        raise RuntimeError("missing Python command")
    executable = Path(arguments.pop(0)).resolve(strict=True)
    if executable != Path(sys.executable).resolve(strict=True):
        raise RuntimeError("hardened host accepts only the current Python interpreter")
    if arguments and arguments[0] == "-B":
        arguments.pop(0)
    if not arguments:
        raise RuntimeError("missing Python source")
    target = arguments.pop(0)
    if target == "-c":
        if not arguments:
            raise RuntimeError("missing -c source")
        source = arguments.pop(0)
        sys.argv = ["-c", *arguments]
        # This runner evaluates the already-opened command in-process. Recreate
        # fresh ``python -c`` import semantics so the current workspace remains
        # the model-visible module root instead of inheriting this runner's path.
        _set_import_origin("")
        namespace = {"__name__": "__main__", "__file__": "<string>"}
        exec(compile(source, "<string>", "exec"), namespace, namespace)  # noqa: S102
        return
    script = Path(target).resolve(strict=True)
    sys.argv = [str(script), *arguments]
    # A fresh ``python script.py`` resolves sibling imports from the script's
    # directory. ``runpy.run_path`` alone leaves this runner's path in place.
    _set_import_origin(str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


def _set_import_origin(origin: str) -> None:
    if sys.path:
        sys.path[0] = origin
    else:
        sys.path.append(origin)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--uid", type=int, required=True)
    parser.add_argument("--gid", type=int, required=True)
    parser.add_argument(
        "--workspace-mode", choices=("read-only", "writable"), required=True
    )
    parser.add_argument("--protected", action="append", type=Path, default=[])
    parser.add_argument("--generated-source", type=Path)
    parser.add_argument("python_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    python_command = list(args.python_command)
    if python_command and python_command[0] == "--":
        python_command.pop(0)

    workspace = args.workspace.resolve(strict=True)
    prepare_tree(
        workspace,
        gid=args.gid,
        writable=args.workspace_mode == "writable",
        protected_paths=tuple(args.protected),
        generated_source=args.generated_source,
    )
    os.chdir(workspace)
    install_limits()
    seal_standard_input()
    drop_identity(args.uid, args.gid)
    os.environ.clear()
    os.environ.update(
        {
            "HOME": str(workspace),
            "PATH": "/usr/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    install_seccomp()
    run_python_command(python_command)


def _prctl(option: int, argument: object, pointer: object = 0) -> None:
    if _LIBC.prctl(option, argument, pointer, 0, 0) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


if __name__ == "__main__":
    main()
