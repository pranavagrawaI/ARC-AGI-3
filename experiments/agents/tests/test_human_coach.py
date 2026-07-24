"""Tests for optional human-coach feedback channel on 0002."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np

from experiments.harness import Decision, Observation, Transition

AGENT_ROOT = Path(__file__).resolve().parents[1] / "0002-llm-repl-agent"
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from llm_repl_agent.agent import LlmReplAgent  # noqa: E402
from llm_repl_agent.client import ChatResponse  # noqa: E402
from llm_repl_agent.human_coach import (  # noqa: E402
    CoachingAgent,
    InteractiveHumanCoach,
    ScriptedHumanCoach,
)
from llm_repl_agent.process_log import ProcessLog  # noqa: E402
from llm_repl_agent.workspace import PythonWorkspace  # noqa: E402


def _obs(*, levels: int = 0, value: int = 0) -> Observation:
    pixels = np.full((1, 4, 4), value, dtype=np.int8)
    pixels.setflags(write=False)
    return Observation(
        pixels=pixels,
        available_actions=(1, 2, 3, 4),
        levels_completed=levels,
        state="NOT_FINISHED",
        full_reset=False,
    )


class HumanCoachTests(unittest.TestCase):
    def test_interactive_coach_returns_typed_note(self) -> None:
        coach = InteractiveHumanCoach()
        before = _obs(value=0)
        after = _obs(value=1)

        output = StringIO()
        with (
            patch("builtins.input", return_value="  avoid action 1  "),
            redirect_stdout(output),
        ):
            note = coach.note_after_transition(
                step=1,
                transition=Transition(
                    before=before,
                    decision=Decision(action=1),
                    after=after,
                ),
                frame_path=Path("/tmp/model-facing-frame.png"),
            )

        self.assertEqual(note, "avoid action 1")
        self.assertIn("/tmp/model-facing-frame.png", output.getvalue())

    def test_interactive_coach_can_abstain_for_whole_session(self) -> None:
        coach = InteractiveHumanCoach(auto_skip=True)
        before = _obs(value=0)
        after = _obs(value=1)

        note = coach.note_after_transition(
            step=1,
            transition=Transition(
                before=before,
                decision=Decision(action=1),
                after=after,
            ),
        )

        self.assertIsNone(note)
        self.assertTrue(coach.enabled)

    def test_observation_message_includes_labeled_human_coach_notes(self) -> None:
        client = MagicMock()
        with TemporaryDirectory() as directory, PythonWorkspace(timeout_seconds=5) as workspace:
            log_path = Path(directory) / "process.jsonl"
            with ProcessLog(log_path) as process_log:
                agent = LlmReplAgent(
                    client=client,
                    workspace=workspace,
                    process_log=process_log,
                )
                agent.add_human_feedback("try clicking the yellow bar", after_step=0)
                current = agent._observation(_obs())
                message = agent._observation_message(
                    current,
                    model_round=1,
                    python_calls_this_action=0,
                )
            text = message["content"][0]["text"]
            payload = json.loads(text)
            self.assertIn("environment_observation", payload)
            self.assertIn("human_coach", payload)
            self.assertEqual(
                payload["human_coach"]["notes"][0]["text"],
                "try clicking the yellow bar",
            )
            events = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertTrue(any(ev.get("type") == "human_feedback" for ev in events))
            # Building a request does not consume the note before later rounds.
            self.assertEqual(len(agent._pending_human_feedback), 1)

    def test_human_note_persists_through_tool_round_until_decision(self) -> None:
        client = MagicMock()
        client.complete.side_effect = [
            ChatResponse(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "python",
                            "arguments": json.dumps({"code": "print('inspect')"}),
                        },
                    }],
                },
                usage={},
            ),
            ChatResponse(
                message={
                    "role": "assistant",
                    "content": '{"action": 1, "data": null, "reason": "use note"}',
                },
                usage={},
            ),
        ]
        with TemporaryDirectory() as directory, PythonWorkspace(timeout_seconds=5) as workspace:
            with ProcessLog(Path(directory) / "process.jsonl") as process_log:
                agent = LlmReplAgent(
                    client=client,
                    workspace=workspace,
                    process_log=process_log,
                )
                agent.add_human_feedback("avoid repeating action 4", after_step=0)

                decision = agent.act(_obs())

        self.assertEqual(decision.action, 1)
        self.assertEqual(agent._pending_human_feedback, [])
        for call in client.complete.call_args_list:
            observation_messages = [
                message
                for message in call.kwargs["messages"]
                if message.get("role") == "user"
                and isinstance(message.get("content"), list)
            ]
            payload = json.loads(observation_messages[-1]["content"][0]["text"])
            self.assertEqual(
                payload["human_coach"]["notes"][0]["text"],
                "avoid repeating action 4",
            )

    def test_scripted_coach_wrap_injects_after_observe(self) -> None:
        client = MagicMock()
        with TemporaryDirectory() as directory, PythonWorkspace(timeout_seconds=5) as workspace:
            script = Path(directory) / "notes.jsonl"
            script.write_text(
                json.dumps({"after_step": 1, "text": "stop repeating the same click"})
                + "\n",
                encoding="utf-8",
            )
            log_path = Path(directory) / "process.jsonl"
            with ProcessLog(log_path) as process_log:
                inner = LlmReplAgent(
                    client=client,
                    workspace=workspace,
                    process_log=process_log,
                )
                coach = MagicMock(wraps=ScriptedHumanCoach(script))
                agent = CoachingAgent(inner, coach)
                before = _obs(value=0)
                after = _obs(value=1)
                # Seed the same pre-act state act() would have left.
                inner._pending_before = inner.store.write_observation(before, step=0)
                inner._current = inner._pending_before
                agent.observe(
                    Transition(
                        before=before,
                        decision=Decision(action=1),
                        after=after,
                    )
                )
                self.assertEqual(inner.actions, 1)
                self.assertEqual(len(inner._pending_human_feedback), 1)
                self.assertIn("stop repeating", inner._pending_human_feedback[0]["text"])
                self.assertEqual(
                    inner._current.png_path,
                    inner.store.observations_dir / "step-00001.png",
                )
                self.assertEqual(
                    coach.note_after_transition.call_args.kwargs["frame_path"],
                    inner._current.png_path,
                )
                events = [json.loads(line) for line in log_path.read_text().splitlines()]
                self.assertTrue(any(ev.get("type") == "human_feedback" for ev in events))


if __name__ == "__main__":
    unittest.main()
