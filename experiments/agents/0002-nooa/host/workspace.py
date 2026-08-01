"""Shared workspace import convention for the paired control and NOOA arm."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from p9.agent import IMPORT_CONVENTION
from p9.frame_tools import FRAME_TOOLS_USAGE


DIRECT_FRAME_TOOLS_USAGE = "import frames"
DIRECT_IMPORT_CONVENTION = (
    "The workspace is already on the module path. Import a module you wrote directly, "
    "for example `import frames`; do not modify sys.path for workspace imports.\n"
)
_CONTROL_SYSTEM_IMPORT_CONVENTION = (
    "The interpreter runs isolated, so the workspace directory is not on sys.path. To import a\n"
    'module you wrote, run sys.path.insert(0, ".") first.'
)
_CONTROL_SYSTEM_DIRECT_IMPORT_CONVENTION = (
    "The workspace is already on the module path. Import a module you wrote directly, for\n"
    "example `import frames`; do not modify sys.path for workspace imports."
)
_CONTROL_WORKER_PATH_MARKER = 'sys.path.insert(0, "/packages")'


def _replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise RuntimeError(f"{path.name} does not contain exactly one frozen import convention")
    path.write_text(source.replace(old, new), encoding="utf-8")


def adapt_workspace_for_direct_import(root: Path) -> None:
    """Give both arms identical, executable workspace-module instructions."""

    _replace_once(root / "WORKSPACE.md", FRAME_TOOLS_USAGE, DIRECT_FRAME_TOOLS_USAGE)
    _replace_once(root / "WORKSPACE.md", IMPORT_CONVENTION, DIRECT_IMPORT_CONVENTION)
    _replace_once(root / "frames.py", FRAME_TOOLS_USAGE, DIRECT_FRAME_TOOLS_USAGE)


def adapt_control_spec(spec: Any) -> Any:
    """Return a 0002-R spec whose system prompt states the shared convention."""

    prompt = spec.system_prompt
    if prompt.count(FRAME_TOOLS_USAGE) != 1:
        raise RuntimeError("control system prompt has an unexpected helper convention")
    if prompt.count(_CONTROL_SYSTEM_IMPORT_CONVENTION) != 1:
        raise RuntimeError("control system prompt has an unexpected import convention")
    return replace(
        spec,
        system_prompt=prompt.replace(
            FRAME_TOOLS_USAGE,
            DIRECT_FRAME_TOOLS_USAGE,
        ).replace(
            _CONTROL_SYSTEM_IMPORT_CONVENTION,
            _CONTROL_SYSTEM_DIRECT_IMPORT_CONVENTION,
        ),
    )


@contextmanager
def direct_import_control_runtime() -> Iterator[None]:
    """Put /workspace on 0002-R's isolated worker path during construction."""

    from llm_repl_agent import workspace as control_runtime

    original_worker_source = control_runtime.WORKER_SOURCE
    if original_worker_source.count(_CONTROL_WORKER_PATH_MARKER) != 1:
        raise RuntimeError("control worker has an unexpected module-path bootstrap")
    control_runtime.WORKER_SOURCE = original_worker_source.replace(
        _CONTROL_WORKER_PATH_MARKER,
        _CONTROL_WORKER_PATH_MARKER + '\nsys.path.insert(0, "/workspace")',
    )
    try:
        yield
    finally:
        control_runtime.WORKER_SOURCE = original_worker_source
