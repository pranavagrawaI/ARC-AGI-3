"""Retrodict's ``search`` tool without ripgrep.

thinharness shells out to ``rg --json``; the offline Kaggle image has no ripgrep, so this
is a pure-Python search with the same argument model (``SearchArgs``), the same grouped
output layout and the same limits. Differences: Python ``re`` syntax instead of Rust regex,
and ``file_type`` maps a few common ripgrep type names to suffixes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .thin.defaults import DEFAULT_SEARCH_DESCRIPTION, DEFAULT_SEARCH_INSTRUCTIONS
from .thin.filesystem import SearchArgs
from .thin.tools_base import PathValidationError, ToolResult, ToolSpec, _path_error, coerce_args

FILE_TYPES = {
    "py": (".py",),
    "md": (".md", ".markdown"),
    "txt": (".txt",),
    "json": (".json",),
    "jsonl": (".jsonl",),
    "yaml": (".yaml", ".yml"),
    "toml": (".toml",),
    "sh": (".sh", ".bash"),
    "js": (".js", ".mjs"),
    "ts": (".ts",),
}
SKIP_DIRS = {".thinharness", "__pycache__", ".git"}


@dataclass
class SearchMatch:
    line_number: int
    line_text: str


@dataclass
class SearchFile:
    path: str
    matches: list[SearchMatch] = field(default_factory=list)


class SearchTool:
    def __init__(self, files, *, max_search_line_chars=180):
        self.files = files  # thin.filesystem.FileTools: root, read_policy, _display, _truncate
        self.max_search_line_chars = max_search_line_chars

    def spec(self) -> ToolSpec:
        return ToolSpec(
            "search", DEFAULT_SEARCH_DESCRIPTION, SearchArgs, self.search,
            instructions=DEFAULT_SEARCH_INSTRUCTIONS,
        )

    def search(self, args):
        args = coerce_args(args, SearchArgs)
        query = args.query
        if not query:
            return ToolResult(False, "query is required; pass a non-empty query string")
        try:
            pattern = re.compile(query)
        except re.error:
            pattern = re.compile(re.escape(query))
        try:
            roots = self._search_paths(args)
        except PathValidationError as exc:
            return _path_error(exc)
        if "path" in args.model_fields_set and args.path != "." and not roots:
            return ToolResult(False, f"path not found: {args.path}", {"path": args.path})
        roots = roots or [self.files.root]
        suffixes = FILE_TYPES.get(args.file_type, ()) if args.file_type else ()
        files = []
        for root in roots:
            candidates = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
            for path in candidates:
                if any(part in SKIP_DIRS for part in path.relative_to(self.files.root).parts):
                    continue
                if suffixes and path.suffix not in suffixes:
                    continue
                if not self.files.read_policy.allows(path):
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                matches = [
                    SearchMatch(number, line)
                    for number, line in enumerate(text.splitlines(), start=1)
                    if pattern.search(line)
                ]
                if matches:
                    files.append(SearchFile(self.files._display(path), matches))
        if not files:
            return ToolResult(True, self._no_matches_message(query, args.path, args.file_type), {"matches": 0})
        files.sort(key=lambda file: file.path)
        total_files = len(files)
        total_matches = sum(len(file.matches) for file in files)
        shown = files[: args.max_files]
        content = self._format_search_output(
            query, args.path, args.file_type, shown, total_files, total_matches,
            args.max_files, args.max_matches_per_file,
            args.max_line_chars or self.max_search_line_chars,
        )
        result = self.files._truncate(content, prefix="search", max_chars=args.max_chars or self.files.max_tool_chars)
        result.metadata.update({"matches": total_matches, "cwd": str(self.files.root)})
        return result

    def _search_paths(self, args):
        if "path" not in args.model_fields_set or args.path == ".":
            return []
        path = self.files.read_policy.resolve(args.path)
        return [path] if path.exists() else []

    @staticmethod
    def _scope(path, file_type):
        parts = []
        if path and path != ".":
            parts.append(f"path={path}")
        if file_type:
            parts.append(f"type={file_type}")
        return ", ".join(parts) if parts else "all readable files"

    def _no_matches_message(self, query, path, file_type):
        return (
            "No matches found.\n"
            f"  query: {query}\n"
            f"  scope: {self._scope(path, file_type)}\n"
            "  hint: broaden the query, remove path/file_type filters, or try simpler terms."
        )

    def _format_search_output(
        self, query, path, file_type, files, total_files, total_matches, max_files,
        max_matches_per_file, max_line_chars,
    ):
        shown_matches = sum(min(len(file.matches), max_matches_per_file) for file in files)
        omitted_matches = total_matches - shown_matches
        parts = [
            "summary:\n"
            f"  query: {query}\n"
            f"  scope: {self._scope(path, file_type)}\n"
            f"  files: {total_files} total, {len(files)} shown\n"
            f"  matches: {shown_matches} shown, {omitted_matches} omitted\n"
        ]
        for file in files:
            block = [file.path]
            for match in file.matches[:max_matches_per_file]:
                line = match.line_text
                if len(line) > max_line_chars:
                    line = f"{line[:max_line_chars]}..."
                block.append(f"  {match.line_number}: {line}")
            omitted = len(file.matches) - min(len(file.matches), max_matches_per_file)
            if omitted:
                block.append(f"  ... {omitted} more match(es)")
            parts.append("\n".join(block) + "\n")
        if total_files > max_files:
            parts.append(f"note: {total_files - max_files} more file(s) omitted")
        return "\n".join(parts)
