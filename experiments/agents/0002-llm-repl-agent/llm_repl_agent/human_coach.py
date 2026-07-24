"""Optional human coach notes for 0002 live runs.

Environment observations stay labeled separately from human text so the model
can tell game data from coach guidance. This is not contestant-admissible for
official scoring claims unless the experiment is explicitly human-supervised.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol

from experiments.harness import Transition


HUMAN_COACH_SYSTEM_ADDENDUM = """
Human-coach channel: some user-role messages include a human_coach object with
natural-language guidance from a human who can see the same public frames you see.
Treat coach notes as optional, fallible advice — not environment ground truth and
not mandatory goals. Prefer public transitions when coach notes conflict with
evidence. You remain the sole actor; do not wait for permission. Never invent a
human request that was not provided in human_coach.
""".strip()


class HumanCoach(Protocol):
    def note_after_transition(
        self,
        *,
        step: int,
        transition: Transition,
        frame_path: Path | None = None,
    ) -> str | None:
        """Return coach text to attach before the next act, or None to skip."""


class InteractiveHumanCoach:
    """Prompt on stdin after each environment transition."""

    def __init__(
        self,
        *,
        prompt_prefix: str = "coach",
        auto_skip: bool = False,
    ) -> None:
        self.prompt_prefix = prompt_prefix
        self.enabled = True
        self.auto_skip = auto_skip

    def note_after_transition(
        self,
        *,
        step: int,
        transition: Transition,
        frame_path: Path | None = None,
    ) -> str | None:
        if not self.enabled:
            return None
        before = transition.before
        after = transition.after
        header = (
            f"\n[{self.prompt_prefix}] after env step {step} | "
            f"levels {before.levels_completed}->{after.levels_completed} | "
            f"state {after.state} | legal {list(after.available_actions)}"
        )
        if frame_path is not None:
            header += f" | frame {frame_path}"
        print(header, flush=True)
        if self.auto_skip:
            print(
                f"[{self.prompt_prefix}] auto-skip (no human note this session)",
                flush=True,
            )
            return None
        try:
            text = input(
                f"[{self.prompt_prefix}] human note (empty=skip, quit=disable coach): "
            ).strip()
        except EOFError:
            self.enabled = False
            return None
        if text.lower() in {"quit", "q", ":q"}:
            self.enabled = False
            print(
                f"[{self.prompt_prefix}] coach disabled for remainder of run",
                flush=True,
            )
            return None
        return text or None


class ScriptedHumanCoach:
    """Replay notes from JSONL: {\"after_step\": N, \"text\": \"...\"}."""

    def __init__(self, path: Path) -> None:
        self._by_step: dict[int, str] = {}
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            step = row.get("after_step")
            text = row.get("text")
            if not isinstance(step, int) or isinstance(step, bool) or step < 0:
                raise ValueError(
                    f"{path}:{line_number}: after_step must be a non-negative int"
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"{path}:{line_number}: text must be a nonempty string"
                )
            self._by_step[step] = text.strip()

    def note_after_transition(
        self,
        *,
        step: int,
        transition: Transition,
        frame_path: Path | None = None,
    ) -> str | None:
        return self._by_step.get(step)


class CoachingAgent:
    """Wrap a 0002 agent: after each observe(), optionally inject human text."""

    def __init__(self, agent: Any, coach: HumanCoach) -> None:
        self.agent = agent
        self.coach = coach

    def act(self, observation):  # type: ignore[no-untyped-def]
        return self.agent.act(observation)

    def observe(self, transition: Transition) -> None:
        self.agent.observe(transition)
        step = int(getattr(self.agent, "actions", 0))
        current = getattr(self.agent, "_current", None)
        raw_frame_path = getattr(current, "png_path", None)
        frame_path = raw_frame_path if isinstance(raw_frame_path, Path) else None
        note = self.coach.note_after_transition(
            step=step,
            transition=transition,
            frame_path=frame_path,
        )
        if note:
            self.agent.add_human_feedback(note, after_step=step)

    def finalize(self, observation):  # type: ignore[no-untyped-def]
        return self.agent.finalize(observation)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.agent, name)
