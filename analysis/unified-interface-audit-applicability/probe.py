from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import tycho.workspace.sandbox as tycho_sandbox
from nooa.unifiedllm import FakeLLMClient

import tycho_nooa.agent as tycho_nooa_agent
import tycho_nooa.models as tycho_nooa_models
import tycho_nooa.tools as tycho_nooa_tools
from tycho_nooa import GameTurn, TychoNooaAgent


def source_binding(module_path: str | None) -> dict[str, str]:
    if module_path is None:
        raise RuntimeError("probe dependency has no source file")
    path = Path(module_path).resolve()
    return {
        "path": str(path),
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    os.environ["TYCHO_MODE"] = "no_world_model"
    os.environ["TYCHO_SANDBOX_RUNTIME"] = "host"

    with tempfile.TemporaryDirectory(prefix="unified-audit-probe-") as root_name:
        root = Path(root_name)
        trace_path = root / "tool-trace.jsonl"
        agent = TychoNooaAgent(
            llm=FakeLLMClient(scripted_responses=[]),
            trigger_builder=False,
            game_id="applicability-probe",
            workspace_root=root / "workspace-root",
            available_actions=["UP", "DOWN"],
            tool_trace_path=trace_path,
        )
        turn = GameTurn(
            grid=[[0, 1], [2, 3]],
            available_actions=["UP", "DOWN"],
            sequence=0,
            level=0,
            turn_in_level=0,
        )
        agent._begin_decision(turn, None)
        surface = agent._tool_surface
        take_action = next(
            tool for tool in surface.native_tools() if tool.name == "take_action"
        )
        schema = take_action.parameters_model.model_json_schema()

        action_error = None
        try:
            surface.dispatch(
                take_action,
                {"action": "DOWN", "row": 10, "col": 10},
                tool_call_id="conditional-schema-probe",
                raw_arguments='{"action":"DOWN","row":10,"col":10}',
            )
        except Exception as error:  # noqa: BLE001 - classification is the probe
            action_error = {"type": type(error).__name__, "message": str(error)}

        trace_rows = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
        ]
        action_trace = next(
            row
            for row in trace_rows
            if row["tool_call_id"] == "conditional-schema-probe"
        )

        run_python_result = surface.run_python(
            "from pathlib import Path\n"
            "print(sorted(path.name for path in Path('.').glob('.tycho-run-*.py')))"
        )
        post_run_transients = sorted(
            path.name for path in agent.workspace_dir.glob(".tycho-run-*.py")
        )

        frame_path = agent.workspace_dir / "level_0" / "turn_000.json"
        durable_frame = json.loads(frame_path.read_text(encoding="utf-8"))

        result = {
            "schema": "unified-interface-audit-applicability-probe-v1",
            "bound_sources": {
                "tycho_nooa_agent": source_binding(tycho_nooa_agent.__file__),
                "tycho_nooa_models": source_binding(tycho_nooa_models.__file__),
                "tycho_nooa_tools": source_binding(tycho_nooa_tools.__file__),
                "tycho_sandbox": source_binding(tycho_sandbox.__file__),
            },
            "audit_report_gate": {
                "surface_has_audit_enabled_state": hasattr(surface, "_audit_enabled"),
                "surface_has_audit_evidence_revision": hasattr(
                    surface, "_audit_evidence_revision"
                ),
            },
            "conditional_action_schema": {
                "action_schema": schema["properties"]["action"],
                "row_schema": schema["properties"]["row"],
                "col_schema": schema["properties"]["col"],
                "invalid_non_click_coordinates": {
                    "input": {"action": "DOWN", "row": 10, "col": 10},
                    "raised": action_error,
                    "trace": {
                        key: action_trace[key]
                        for key in (
                            "parse_status",
                            "validation_status",
                            "execution_status",
                            "error_stage",
                            "status",
                            "result",
                        )
                    },
                },
            },
            "run_python_transient_visibility": {
                "child_observed": run_python_result.strip(),
                "workspace_after_return": post_run_transients,
            },
            "action_vocabulary_fields": {
                "live_model_field": "available_actions",
                "durable_frame_keys": sorted(durable_frame),
                "durable_available": durable_frame.get("available"),
                "durable_available_actions": durable_frame.get("available_actions"),
            },
        }
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
