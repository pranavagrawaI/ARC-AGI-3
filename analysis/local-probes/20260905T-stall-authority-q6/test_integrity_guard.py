from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "patched-source/after"))
from integrity_guard import IntegrityGuard  # noqa: E402
from nooa.context_blocks.formatter import XMLBlockFormatter  # noqa: E402
from nooa.events import PythonOutput, ResultStatus  # noqa: E402
from nooa.unifiedllm import FakeLLMClient  # noqa: E402
from solver_agent import MdArcSolverAgent  # noqa: E402
from stall_contract import read_rows  # noqa: E402


def make_event(text):
    return PythonOutput(
        tool_call_id="note-call",
        execution_count=1,
        execution_status=ResultStatus.COMPLETE,
        stdout=text,
    )


def prepare(tmp_path):
    formatter = XMLBlockFormatter()
    guard = IntegrityGuard(tmp_path, formatter)
    body = "Decorations/goals remain unproven.\n"
    guard.written("hypotheses.md", body, body)
    guard.read("hypotheses.md", body)
    guard.capture("note-call", body, body)
    event = make_event(body)
    guard.output(event)
    return guard, formatter.format_event(event), body


def test_real_renderer_readback_verified_and_eviction_is_not_a_pass(tmp_path):
    guard, rendered, body = prepare(tmp_path)
    guard.check_messages([{"role": "user", "content": rendered}])
    assert read_rows(tmp_path / "integrity.jsonl")[-1]["coverage"] == "COMPLETE"
    guard.check_messages([{"role": "assistant", "content": body}])
    assert read_rows(tmp_path / "integrity.jsonl")[-1]["coverage"] == "UNOBSERVED"


def test_mutated_final_feedback_blocks_and_latches_even_if_assistant_has_original(
    tmp_path,
):
    guard, rendered, body = prepare(tmp_path)
    with pytest.raises(RuntimeError, match="final request"):
        guard.check_messages(
            [
                {"role": "assistant", "content": rendered},
                {
                    "role": "user",
                    "content": rendered.replace(
                        "Decorations/goals", "Decorations/workspacegoals"
                    ),
                },
            ]
        )
    with pytest.raises(RuntimeError):
        guard.check_messages([{"role": "user", "content": rendered}])
    assert (tmp_path / "ABORT").exists()


def test_capture_truncation_does_not_claim_complete_coverage(tmp_path):
    guard = IntegrityGuard(tmp_path, XMLBlockFormatter())
    guard.read("hypotheses.md", "a very long readback of notes")
    guard.capture("truncated", "a very long <truncated>", "a very long <truncated>")
    assert read_rows(tmp_path / "integrity.jsonl")[-1]["coverage"] == "UNOBSERVED"
    assert not guard.outputs


def test_read_snapshot_survives_later_authoring(tmp_path):
    guard, rendered, body = prepare(tmp_path)
    guard.written("hypotheses.md", "replacement", "replacement")
    guard.check_messages([{"role": "user", "content": rendered}])
    assert read_rows(tmp_path / "integrity.jsonl")[-1]["stored"] == body


def test_real_actor_installs_guard_before_any_fake_model_call(tmp_path):
    async def run():
        fake = FakeLLMClient(scripted_responses=[])
        actor = MdArcSolverAgent(llm=fake, run_dir=tmp_path)
        actor.write_notes("hypotheses.md", "decorations/goals")
        assert actor.read_notes("hypotheses.md") == "decorations/goals"
        assert fake.call_count == 0
        assert (
            read_rows(tmp_path / "ipc/integrity.jsonl")[0]["event"] == "GUARD_INSTALLED"
        )

    asyncio.run(run())


def test_real_codeact_printed_note_reaches_second_fake_call(tmp_path):
    from nooa.unifiedllm import LLMResponse, ToolCall
    from host_contract import HostEvidence
    import json

    def response(code, call_id):
        return LLMResponse(
            raw_response=None,
            content="",
            tool_calls=[
                ToolCall(
                    id=call_id,
                    name="execute_python",
                    arguments=json.dumps({"code": code}),
                )
            ],
            finish_reason="tool_calls",
            assistant_message={"role": "assistant", "content": ""},
        )

    async def run():
        fake = FakeLLMClient(
            scripted_responses=[
                response("print(self.read_notes('hypotheses.md'))", "read-note"),
                response("self.submit_actions(['UP'], rationale='move')", "submit"),
            ]
        )
        actor = MdArcSolverAgent(llm=fake, run_dir=tmp_path)
        actor.write_notes("hypotheses.md", "decorations/goals remain unproven")
        HostEvidence(actor.ipc_dir).publish(
            {
                "grid": [[0]],
                "levels_completed": 0,
                "state": "NOT_FINISHED",
                "available_actions": ["UP", "DOWN"],
            },
            0,
        )
        task = asyncio.create_task(
            actor.handle({"user_messages": ["Read the notes and act."]})
        )

        async def wait_for_submission():
            while not read_rows(tmp_path / "ipc/actions.jsonl"):
                if task.done():
                    await task
                await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(wait_for_submission(), timeout=10)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert fake.call_count == 2
        assert any(
            r.get("event") == "NOTE_RENDER"
            for r in read_rows(tmp_path / "ipc/integrity.jsonl")
        )
        assert len(read_rows(tmp_path / "ipc/actions.jsonl")) == 1

    asyncio.run(run())
