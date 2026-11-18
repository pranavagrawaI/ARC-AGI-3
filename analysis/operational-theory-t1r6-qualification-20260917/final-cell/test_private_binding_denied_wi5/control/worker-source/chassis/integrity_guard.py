"""Check immutable note readbacks at storage, capture and final-request boundaries."""

from __future__ import annotations

from pathlib import Path
from contextvars import ContextVar
import hashlib

from nooa.config.truncation_config import FormatConfig
from nooa.context_blocks import XMLBlockFormatter
from stall_contract import append


class IntegrityGuard:
    def __init__(self, ipc: Path, formatter):
        self.ipc = ipc
        self.render_event = formatter.format_event
        self.reads: list[dict] = []
        self.captures: dict[str, list[dict]] = {}
        self.outputs: dict[str, tuple[str, list[dict]]] = {}
        # A summarizer runs in a background task. Bind the original input to
        # that call, not to mutable shared state or the Actor's current request.
        self.summary_input = ContextVar("note_summary_input", default=None)
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

    async def guard_summary_call(self, ctx, nxt):
        if ctx.method_name != "summarize":
            return await nxt(ctx)
        history = ctx.kwargs.get("history_markdown", ctx.args[0] if ctx.args else None)
        target = ctx.kwargs.get("target_chars", ctx.args[1] if len(ctx.args) > 1 else None)
        if not isinstance(history, str) or type(target) is not int:
            self.fail("unsupported summarizer input contract")
        active = {event.tool_call_id: event.tag for event in ctx.agent.target_event_manager.values()
                  if type(event).__name__ == "PythonOutput"}
        self.check_messages([{"role": "user", "content": history}], active, history=True)
        token = self.summary_input.set((history, target))
        try:
            return await nxt(ctx)
        finally:
            self.summary_input.reset(token)

    def check_summary_messages(self, messages: list[dict], summarizer) -> None:
        self.require_clear()
        original = self.summary_input.get()
        if original is None:
            self.fail("summarizer request has no bound history input")
        history, target = original
        tasks = [event for event in summarizer.event_manager.values()
                 if type(event).__name__ == "Task"]
        if not tasks:
            self.fail("summarizer request has no task event")
        task = tasks[-1]
        # Validate original parameters before serialization, then use the same
        # pinned formatter as NOOA. Its enclosing quote style depends on history.
        _, marker, parameters = task.prompt.partition("\n## Input parameters:\n")
        parameters, trailer, after = parameters.rpartition(
            "\n\nPerform the task and return the result directly.")
        expected = f"history_markdown = {history!r}\ntarget_chars = {target!r}"
        if not marker or not trailer or after or parameters != expected:
            self.fail("summarizer history differs before final rendering")
        config = task.metadata.get("truncation_event_format")
        if isinstance(config, dict):
            config = FormatConfig(**config)
        formatter = getattr(summarizer, "_block_formatter", None) or XMLBlockFormatter()
        rendered = formatter.format_event(task.model_copy(deep=True), event_format=config)
        prefix = f'<sys tag="{task.tag}">\n'
        visible = [m["content"] for m in messages if m.get("role") == "user"
                   and isinstance(m.get("content"), str) and m["content"].startswith(prefix)]
        if visible != [prefix + rendered + "\n</sys>"]:
            self.fail("summarizer history differs at final request")
        append(self.ipc / "integrity.jsonl", {
            "event": "SUMMARY_INPUT_VERIFIED",
            "history_sha256": hashlib.sha256(history.encode()).hexdigest(),
            "history_characters": len(history), "target_chars": target,
            "scope": "exact input serialization; summary meaning is not verified",
        })

    def check_messages(self, messages: list[dict], active_output_ids=None, *, history=False) -> None:
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
            # A Summary can mention an archived tool ID without containing its
            # original output. Use active events when checking Actor requests.
            active = active_output_ids is None or tool_id in active_output_ids
            visible = [text for text in user if tool_id in text] if active else []
            if active and active_output_ids is not None:
                tag = active_output_ids[tool_id]
                if history:
                    # Markdown history contains multiple event blocks in one
                    # string. The event heading binds a readback to its source.
                    prefix = f'### User Message `{{"tag": "{tag}"}}`\n\n'
                    # Note bodies can themselves contain Markdown event-looking
                    # headings. Compare the captured extent instead of splitting
                    # arbitrary authored text into apparent events.
                    visible = [text.partition(prefix)[2][:len(expected)]
                               for text in visible if prefix in text]
                else:
                    prefix = f'<sys tag="{tag}">\n'
                    visible = [text[len(prefix):-len("\n</sys>")]
                               for text in visible
                               if text.startswith(prefix) and text.endswith("\n</sys>")]
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
            intact = (visible == [expected] if active_output_ids is not None
                      else any(expected in text for text in visible))
            if not intact:
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
