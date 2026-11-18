"""Retrodict's file tools: verbatim methods of thinharness v0.5.4 ``FileTools`` (read, write,
edit and their helpers) on a workspace root; ``search`` lives in ``retrodict_qwen.search``."""

from __future__ import annotations

import itertools
import time
import uuid
from pathlib import Path

from pydantic import Field

from .defaults import (
    DEFAULT_EDIT_DESCRIPTION,
    DEFAULT_EDIT_INSTRUCTIONS,
    DEFAULT_READ_DESCRIPTION,
    DEFAULT_READ_INSTRUCTIONS,
    DEFAULT_WRITE_DESCRIPTION,
    DEFAULT_WRITE_INSTRUCTIONS,
)
from .tools_base import (
    Json,
    PathPolicy,
    PathValidationError,
    StrictArgs,
    ToolResult,
    ToolSpec,
    _path_error,
    coerce_args,
    contained_path,
)


class ReadArgs(StrictArgs):
    """Arguments for read."""

    path: str
    offset: int = Field(default=1, ge=1)
    limit: int | None = Field(default=None, ge=1)
    max_chars: int | None = Field(default=None, ge=1)

class WriteArgs(StrictArgs):
    """Arguments for write."""

    path: str
    content: str
    append: bool = False

class EditOperation(StrictArgs):
    """One exact replacement for edit."""

    path: str
    old_string: str
    new_string: str
    all: bool = False
    expected_replacements: int | None = Field(default=None, ge=1)

class EditArgs(StrictArgs):
    """Arguments for edit."""

    edits: list[EditOperation] = Field(min_length=1)


class SearchArgs(StrictArgs):
    """Arguments for search."""

    query: str = Field(description="Regex or literal search string.")
    path: str = Field(default=".", description="Optional file or directory path to search. Defaults to all readable roots.")
    file_type: str = Field(default="", description="Optional ripgrep type such as py, rust, or js.")
    max_files: int = Field(default=50, ge=1)
    max_matches_per_file: int = Field(default=10, ge=1)
    max_line_chars: int | None = Field(default=None, ge=1, description="Search-only matched line preview cap. JSONL field output is controlled by fields.")
    timeout: int | None = Field(default=None, ge=1)
    max_chars: int | None = Field(default=None, ge=1)


class FileTools:
    """Workspace-rooted read/write/edit tools (thinharness ``FileTools`` minus search/list/glob)."""

    def __init__(self, root: str | Path = ".", *, max_read_chars: int = 40_000, max_read_bytes: int = 1_000_000, max_tool_chars: int = 40_000) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.output_dir = contained_path(self.root, ".thinharness/outputs")
        self._spill_artifacts: set[Path] = set()
        self.read_policy = PathPolicy(self.root, None, "read")
        self.write_policy = PathPolicy(self.root, None, "write")
        self.max_read_chars = max_read_chars
        self.max_read_bytes = max_read_bytes
        self.max_tool_chars = max_tool_chars

    def specs(self) -> list[ToolSpec]:
        """Return the read, write and edit tool specs (thinharness order and flags)."""
        return [
            ToolSpec("read", DEFAULT_READ_DESCRIPTION, ReadArgs, self.read, instructions=DEFAULT_READ_INSTRUCTIONS),
            ToolSpec("write", DEFAULT_WRITE_DESCRIPTION, WriteArgs, self.write, sequential=True, instructions=DEFAULT_WRITE_INSTRUCTIONS),
            ToolSpec("edit", DEFAULT_EDIT_DESCRIPTION, EditArgs, self.edit, sequential=True, instructions=DEFAULT_EDIT_INSTRUCTIONS),
        ]

    def read(self, args: ReadArgs | Json) -> ToolResult:
        """Read a contained text file with line numbers."""
        args = coerce_args(args, ReadArgs)
        try:
            path = self._resolve_read_path(args.path)
        except PathValidationError as exc:
            return _path_error(exc)
        if not path.exists():
            return ToolResult(False, f"file not found: {self._display(path)}", {"path": str(path)})
        if path.is_dir():
            return ToolResult(False, f"path is a directory: {self._display(path)}", {"path": str(path)})
        offset = args.offset
        limit = args.limit
        size = path.stat().st_size
        if size > self.max_read_bytes and limit is None:
            return ToolResult(
                False,
                (
                    f"file is {size} bytes, over max_read_bytes={self.max_read_bytes}; "
                    "pass offset and limit to read a bounded range"
                ),
                {"path": str(path), "size_bytes": size, "max_read_bytes": self.max_read_bytes},
            )
        if size > self.max_read_bytes:
            assert limit is not None
            selected, total_lines = self._read_large_range(path, offset, limit)
            note = f"read {len(selected)} line(s) from large file {self._display(path)} starting at line {offset}"
            if len(selected) == limit:
                note += " (more lines may be available; increase offset/limit)"
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            selected = lines[offset - 1:] if limit is None else lines[offset - 1: offset - 1 + limit]
            total_lines = len(lines)
            note = f"read {len(selected)} of {total_lines} lines from {self._display(path)}"
            if limit is not None and offset + len(selected) - 1 < total_lines:
                note += " (more lines available; increase offset/limit)"
        limit_chars = min(args.max_chars or self.max_read_chars, self.max_read_chars)
        body = "\n".join(f"{i}\t{line}" for i, line in enumerate(selected, start=offset))
        result = self._truncate(f"{note}\n{body}" if body else note, prefix="read", max_chars=limit_chars)
        result.metadata.update({"path": str(path), "total_lines": total_lines, "returned_lines": len(selected), "size_bytes": size})
        return result

    def write(self, args: WriteArgs | Json) -> ToolResult:
        """Write a contained UTF-8 file."""
        args = coerce_args(args, WriteArgs)
        try:
            path = self.write_policy.resolve(args.path)
        except PathValidationError as exc:
            return _path_error(exc)
        content = args.content
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.append else "w"
        with path.open(mode, encoding="utf-8") as handle:
            handle.write(content)
        action = "appended" if args.append else "wrote"
        return ToolResult(True, f"{action} {len(content.encode('utf-8'))} bytes to {self._display(path)}", {"path": str(path)})

    def edit(self, args: EditArgs | Json) -> ToolResult:
        """Apply one or more exact replacements to contained UTF-8 files."""
        args = coerce_args(args, EditArgs)
        applied = 0
        results: list[Json] = []
        lines: list[str] = []
        for index, edit in enumerate(args.edits, start=1):
            result = self._edit_one(edit)
            path = self._edit_result_path(edit, result)
            replacements = result.metadata.get("replacements", 0) if result.ok else 0
            item: Json = {
                "index": index,
                "ok": result.ok,
                "path": edit.path,
                "replacements": replacements,
            }
            if result.ok:
                applied += 1
                lines.append(f"{index}. ok {path}: replaced {replacements} occurrence(s)")
            else:
                item["error"] = result.content
                if "error_type" in result.metadata:
                    item["error_type"] = result.metadata["error_type"]
                if "matches" in result.metadata:
                    item["matches"] = result.metadata["matches"]
                lines.append(f"{index}. FAILED {path}: {result.content}")
            results.append(item)
        failed = len(results) - applied
        return ToolResult(
            failed == 0,
            "\n".join(lines),
            {"applied": applied, "failed": failed, "results": results},
        )

    def _edit_one(self, args: EditOperation) -> ToolResult:
        """Apply one exact replacement and return its isolated outcome."""
        try:
            path = self.write_policy.resolve(args.path)
        except PathValidationError as exc:
            return _path_error(exc)
        old = args.old_string
        new = args.new_string
        replace_all = args.all
        if not old:
            return ToolResult(False, "old_string must not be empty", {"path": str(path)})
        if not path.exists():
            return ToolResult(False, "file not found", {"path": str(path)})
        if not path.is_file():
            return ToolResult(False, "path is not a file", {"path": str(path), "error_type": "PathTypeError"})
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(False, f"{type(exc).__name__}: {exc}", {"path": str(path), "error_type": type(exc).__name__})
        count = text.count(old)
        if count == 0:
            return ToolResult(False, "old_string not found", {"path": str(path)})
        if args.expected_replacements is not None and count != args.expected_replacements:
            return ToolResult(False, f"expected {args.expected_replacements} replacement(s), found {count}", {"path": str(path), "matches": count})
        if count > 1 and not replace_all:
            return ToolResult(False, f"old_string appears {count} times; add more context or set all=true", {"path": str(path), "matches": count})
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return ToolResult(False, f"{type(exc).__name__}: {exc}", {"path": str(path), "error_type": type(exc).__name__})
        changed = count if replace_all else 1
        return ToolResult(True, "replaced", {"path": str(path), "replacements": changed})

    def _edit_result_path(self, args: EditOperation, result: ToolResult) -> str:
        path = result.metadata.get("path")
        return self._display(Path(path)) if isinstance(path, str) else args.path

    def _resolve_read_path(self, raw: str | Path) -> Path:
        """Resolve a readable path, including generated spill artifacts."""
        try:
            return self.read_policy.resolve(raw)
        except PathValidationError as policy_error:
            resolved = contained_path(self.root, raw)
            if self._is_readable_spill_artifact(resolved):
                return resolved
            raise policy_error

    def _is_readable_spill_artifact(self, path: Path) -> bool:
        """Return whether path is an exact generated spill artifact."""
        resolved = path.resolve()
        output_dir = self.output_dir.resolve()
        return resolved in self._spill_artifacts and (resolved == output_dir or output_dir in resolved.parents)

    @staticmethod
    def _read_large_range(path: Path, offset: int, limit: int) -> tuple[list[str], int | None]:
        """Stream a bounded line range from a large file."""
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            selected = [
                line.rstrip("\n").rstrip("\r")
                for line in itertools.islice(handle, offset - 1, offset - 1 + limit)
            ]
        return selected, None

    def _display(self, path: Path) -> str:
        """Return a workspace-relative display path when possible."""
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def _truncate(self, text: str, *, prefix: str, max_chars: int | None = None) -> ToolResult:
        """Truncate long tool output and spill the full content to disk."""
        limit = max_chars or self.max_tool_chars
        if len(text) <= limit:
            return ToolResult(True, text)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifact = self.output_dir / f"{prefix}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.txt"
        artifact.write_text(text, encoding="utf-8")
        resolved_artifact = artifact.resolve()
        self._spill_artifacts.add(resolved_artifact)
        saved_to_display = self._display(resolved_artifact)
        head = limit // 2
        tail = limit - head
        content = (
            f"[truncated {len(text)} chars to {limit}; full output saved to {saved_to_display}]\n"
            f"Read the saved output with read(path=\"{saved_to_display}\", offset=1), then continue with later offsets as needed.\n"
            f"{text[:head]}\n...\n{text[-tail:]}"
        )
        return ToolResult(
            True,
            content,
            {"truncated": True, "saved_to": str(resolved_artifact), "saved_to_display": saved_to_display, "chars": len(text)},
        )
