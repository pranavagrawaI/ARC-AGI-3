"""Rolling conversation that tells the model when its transcript loses turns.

``RollingConversation`` evicts whole turns silently: the host logs the eviction,
the model does not learn of it. This subclass keeps the eviction policy and adds
one pointer-only notice to the model-visible message list.

The notice is a fixed template with two substitutions: the evicted step numbers
and the authoritative paths. It contains no description, ranking, or summary of
what was evicted, so compaction cannot promote a host reading of the episode
into the model's context.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping

from llm_repl_agent.conversation import RollingConversation


OMISSION_NOTICE_TEMPLATE = (
    "Transcript omission. Conversation turns for action steps {steps} are no "
    "longer in this context and were not summarized. Every public observation "
    "for those steps remains readable at {observations_glob}. Files you wrote in "
    "the workspace are unchanged."
)


def omission_notice_text(*, steps: tuple[int, ...], observations_glob: str) -> str:
    return OMISSION_NOTICE_TEMPLATE.format(
        steps=compact_steps(steps),
        observations_glob=observations_glob,
    )


def omission_notice_template_sha256() -> str:
    return sha256(OMISSION_NOTICE_TEMPLATE.encode("utf-8")).hexdigest()


def compact_steps(steps: tuple[int, ...]) -> str:
    """Render sorted step numbers as ranges: (1, 2, 3, 5) -> '1-3, 5'."""

    ordered = sorted(set(int(step) for step in steps))
    if not ordered:
        return "none"
    groups: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for step in ordered[1:]:
        if step == previous + 1:
            previous = step
            continue
        groups.append((start, previous))
        start = previous = step
    groups.append((start, previous))
    return ", ".join(
        str(low) if low == high else f"{low}-{high}" for low, high in groups
    )


class AnnouncedConversation(RollingConversation):
    """Evict exactly as 0002 does, then state the omission to the model."""

    def __init__(
        self,
        *,
        system_message: Mapping[str, Any],
        max_turns: int = 6,
        observations_glob: str,
        superseded_frame_note: str | None = None,
    ) -> None:
        super().__init__(system_message=system_message, max_turns=max_turns)
        self.observations_glob = observations_glob
        self.superseded_frame_note = superseded_frame_note
        self._evicted_steps: list[int] = []

    def append(self, *, step: int, messages: list[dict[str, Any]]) -> tuple[int, ...]:
        evicted = super().append(step=step, messages=messages)
        self._evicted_steps.extend(evicted)
        return evicted

    @property
    def evicted_steps(self) -> tuple[int, ...]:
        return tuple(self._evicted_steps)

    def omission_notice(self) -> dict[str, Any] | None:
        if not self._evicted_steps:
            return None
        return {
            "role": "user",
            "content": omission_notice_text(
                steps=tuple(self._evicted_steps),
                observations_glob=self.observations_glob,
            ),
        }

    @staticmethod
    def _without_superseded_frames(
        message: Mapping[str, Any], note: str
    ) -> dict[str, Any]:
        content = message.get("content")
        if not isinstance(content, list):
            return dict(message)
        rebuilt = []
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "image_url":
                rebuilt.append({"type": "text", "text": note})
            else:
                rebuilt.append(part)
        return {**message, "content": rebuilt}

    def messages(self) -> list[dict[str, Any]]:
        output = super().messages()
        # Every retained turn carries its own frame image, all superseded by the
        # current one. Replace them with a pointer rather than dropping them silently.
        if self.superseded_frame_note is not None:
            output = [output[0]] + [
                self._without_superseded_frames(message, self.superseded_frame_note)
                for message in output[1:]
            ]
        notice = self.omission_notice()
        if notice is not None:
            output.insert(1, notice)
        return output
