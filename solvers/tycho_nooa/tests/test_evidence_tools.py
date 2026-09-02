from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_tool(name: str):
    path = REPO_ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_symmetry_uptake_rejects_any_errored_trace_row() -> None:
    uptake = _load_tool("symmetry_uptake")
    errored = {
        "name": "run_python",
        "arguments": {"code": "symmetry.canon(fig)"},
        "status": "error",
        "result": "SandboxError: worker exited without a Python traceback",
    }
    complete = {**errored, "status": "complete", "result": "((0, 0, 1),)"}

    assert uptake._successful_reach(errored) is False
    assert uptake._successful_reach(complete) is True


def test_protocol_pairs_by_tool_call_id_without_shifting_after_an_unexecuted_call() -> (
    None
):
    protocol = _load_tool("read_protocol")
    calls = [
        {"call_index": 1, "tool_call_ids": ["blocked"], "tool_calls": ["take_actions"]},
        {"call_index": 2, "tool_call_ids": ["read-2"], "tool_calls": ["read_file"]},
    ]
    tools = [
        {
            "call_index": 1,
            "tool_call_id": "read-2",
            "name": "read_file",
            "result": "later result",
        }
    ]

    paired = protocol._pair_calls_with_tools(calls, tools)

    assert paired[0][1] == []
    assert [entry["tool_call_id"] for entry in paired[1][1]] == ["read-2"]


def test_protocol_reads_scribe_calls_from_the_shared_tool_trace(
    tmp_path: Path, capsys
) -> None:
    protocol = _load_tool("read_protocol")
    (tmp_path / "llm-trace.jsonl").write_text(
        json.dumps(
            {
                "source": "scribe",
                "call_index": 3,
                "tool_call_ids": ["scribe-edit"],
                "prompt_tokens": 42,
                "reasoning": "record the accepted successor",
            }
        )
        + "\n"
    )
    (tmp_path / "tool-trace.jsonl").write_text(
        json.dumps(
            {
                "source": "scribe",
                "tool_call_id": "scribe-edit",
                "turn_in_level": 1,
                "name": "write_file",
                "arguments": {"path": "notes/turn.md"},
                "status": "complete",
                "result": "wrote notes/turn.md",
            }
        )
        + "\n"
    )

    protocol.show_narrative(
        tmp_path,
        argparse.Namespace(who="scribe", turn=None, chars=200, result_chars=200),
    )

    output = capsys.readouterr().out
    assert "[scribe]" in output
    assert "write_file" in output
    assert "wrote notes/turn.md" in output


def test_protocol_trigger_report_accepts_a_builder_with_no_recommendation(
    tmp_path: Path, capsys
) -> None:
    protocol = _load_tool("read_protocol")
    (tmp_path / "trigger-builder-trace.jsonl").write_text(
        '{"event":"actor_uptake","turn_in_level":19,'
        '"recommended_action":null,"actor_action":{"action":"LEFT"},'
        '"followed_recommended_action":null,"tested_report":false}\n'
    )

    protocol.show_triggers(tmp_path, argparse.Namespace())

    output = capsys.readouterr().out
    assert "recommended=none" in output
    assert "followed=n/a" in output
