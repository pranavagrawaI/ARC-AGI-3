"""Check immutable note readbacks at storage, capture and final-request boundaries."""

from __future__ import annotations

from pathlib import Path

from nooa.config.truncation_config import FormatConfig
from stall_contract import append


class IntegrityGuard:
    def __init__(self, ipc: Path, formatter):
        self.ipc = ipc
        self.render_event = formatter.format_event
        self.reads: list[dict] = []
        self.captures: dict[str, list[dict]] = {}
        self.outputs: dict[str, tuple[str, list[dict]]] = {}
        append(
            ipc / "integrity.jsonl",
            {
                "event": "GUARD_INSTALLED",
                "scope": "note writes and complete printed readbacks",
            },
        )

    def written(self, name: str, expected: str, stored: str) -> None:
        append(
            self.ipc / "integrity.jsonl",
            {
                "event": "NOTE_WRITE",
                "name": name,
                "authored": expected,
                "stored": stored,
            },
        )
        if expected != stored:
            self.fail("authored note differs from stored bytes")

    def read(self, name: str, body: str) -> None:
        self.reads.append({"name": name, "stored": body})

    def capture(self, tool_id: str, before: str, after: str) -> None:
        complete = []
        for read in self.reads:
            if read["stored"] and read["stored"] in before:
                if read["stored"] not in after:
                    self.fail(
                        "note readback mutated during output redaction",
                        {"before": before, "after": after},
                    )
                complete.append(read)
            else:
                append(
                    self.ipc / "integrity.jsonl",
                    {
                        "event": "NOTE_COVERAGE",
                        **read,
                        "coverage": "UNOBSERVED",
                        "reason": "not a complete printed readback",
                    },
                )
        self.reads = []
        if complete:
            if not tool_id:
                self.fail("printed note has no execution identity")
            self.captures[tool_id] = complete

    def output(self, event) -> None:
        reads = self.captures.pop(event.tool_call_id, None)
        if reads:
            # Snapshot this version, not the file's latest content. The same
            # formatter/config will later render the event in the request.
            config = event.metadata.get("truncation_event_format")
            if isinstance(config, dict):
                config = FormatConfig(**config)
            try:
                expected = self.render_event(
                    event.model_copy(deep=True), event_format=config
                )
            except Exception as error:
                self.fail(f"could not capture note event rendering: {error}")
            self.outputs[event.tool_call_id] = (expected, reads)

    def check_messages(self, messages: list[dict]) -> None:
        self.require_clear()
        if self.captures:
            self.fail("complete note capture has no corresponding output event")
        user = []
        for message in messages:
            if message.get("role") not in ("user", "tool"):
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            if isinstance(content, str):
                user.append(content)
        for tool_id, (expected, reads) in self.outputs.items():
            visible = [text for text in user if tool_id in text]
            if not visible:
                append(
                    self.ipc / "integrity.jsonl",
                    {
                        "event": "NOTE_COVERAGE",
                        "tool_call_id": tool_id,
                        "coverage": "UNOBSERVED",
                        "reason": "output evicted or absent",
                    },
                )
                continue
            if not any(expected in text for text in visible):
                self.fail(
                    "visible note event differs at final request",
                    {"expected": expected, "messages": messages},
                )
            for read in reads:
                append(
                    self.ipc / "integrity.jsonl",
                    {
                        "event": "NOTE_RENDER",
                        "tool_call_id": tool_id,
                        **read,
                        "rendered": read["stored"],
                        "coverage": "COMPLETE",
                        "expected_event": expected,
                        "rendered_event": next(
                            text[
                                text.index(expected) : text.index(expected)
                                + len(expected)
                            ]
                            for text in visible
                            if expected in text
                        ),
                    },
                )

    def require_clear(self) -> None:
        if (self.ipc / "ABORT").exists():
            raise RuntimeError((self.ipc / "ABORT").read_text())

    def fail(self, reason: str, evidence: dict | None = None) -> None:
        append(
            self.ipc / "integrity.jsonl",
            {"event": "INTEGRITY_FAILED", "reason": reason, **(evidence or {})},
        )
        (self.ipc / "ABORT").write_text(reason)
        raise RuntimeError(reason)
