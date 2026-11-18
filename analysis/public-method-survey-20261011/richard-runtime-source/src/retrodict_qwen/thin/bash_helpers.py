"""Verbatim from thinharness v0.5.4 ``thinharness/tools/bash.py``: output helpers."""

from __future__ import annotations


def _format_output(stdout: str, stderr: str) -> str:
    """Return labeled command output."""
    return f"stdout:\n{stdout}\nstderr:\n{stderr}"


def _read_limited_text(handle, limit: int) -> tuple[str, bool]:
    """Read a bounded amount of UTF-8-ish text from a binary file handle."""
    handle.seek(0)
    max_bytes = limit * 4
    raw = handle.read(max_bytes + 1)
    bytes_truncated = len(raw) > max_bytes
    text = raw[:max_bytes].decode("utf-8", errors="replace")
    text, chars_truncated = _truncate_text(text, limit)
    return text, bytes_truncated or chars_truncated


def _truncate_text(text: str, limit: int) -> tuple[str, bool]:
    """Return bounded output and whether it was truncated."""
    if len(text) <= limit:
        return text, False
    marker = "...[truncated]"
    if limit <= len(marker):
        return marker[:limit], True
    return f"{text[: limit - len(marker)]}{marker}", True
