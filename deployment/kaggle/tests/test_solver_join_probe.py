from __future__ import annotations

from pathlib import Path

from s0_solver_join import load_probe


def test_fake_endpoint_reaches_real_worker_and_returns_typed_decision(
    tmp_path: Path,
) -> None:
    probe = load_probe()
    server, thread, base_url = probe.start_fake_server()
    try:
        result = probe.run(
            base_url=base_url,
            model="s0-solver-join-model",
            workspace_root=tmp_path / "workspaces",
            fake=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result["decision"] == {"action": "UP", "row": None, "col": None}
    assert result["request_facts"]["count"] == 1
    assert result["request_facts"]["model"] == "s0-solver-join-model"
    assert "take_action" in result["request_facts"]["tool_names"]
    assert result["request_facts"]["message_count"] >= 2
    assert result["request_facts"]["max_tokens"] == 16_000
    assert result["worker_metadata"]["treatment"] == "decision_discipline"
    assert result["worker_metadata"]["decision_envelope"] == {
        "force_action_tool": True,
        "max_actor_calls": 26,
        "max_output_tokens_per_call": 16_000,
        "model_retries": 0,
        "model_timeout_seconds": 600.0,
    }
    assert result["worker"]["stderr"] == ""
