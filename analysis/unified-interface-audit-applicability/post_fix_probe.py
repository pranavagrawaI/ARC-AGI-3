from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from nooa.unifiedllm import FakeLLMClient

import tycho_nooa.agent as tycho_nooa_agent
import tycho_nooa.models as tycho_nooa_models
import tycho_nooa.planner as tycho_nooa_planner
import tycho_nooa.tools as tycho_nooa_tools
import tycho_nooa.wm_probe as tycho_nooa_wm_probe
from tycho_nooa import GameTurn, TychoNooaAgent
from tycho_nooa.models import (
    TYCHO_WORKSPACE_TURN_SCHEMA,
    tycho_workspace_available_actions,
)


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

    with tempfile.TemporaryDirectory(prefix="unified-audit-fix-probe-") as root_name:
        root = Path(root_name)
        trace_path = root / "tool-trace.jsonl"
        agent = TychoNooaAgent(
            llm=FakeLLMClient(scripted_responses=[]),
            trigger_builder=False,
            game_id="fix-verification-probe",
            workspace_root=root / "workspace-root",
            available_actions=["UP", "CLICK"],
            tool_trace_path=trace_path,
        )
        turn = GameTurn(
            grid=[[0, 1], [2, 3]],
            available_actions=["UP", "CLICK"],
            sequence=0,
            level=0,
            turn_in_level=0,
        )
        agent._begin_decision(turn, None)
        surface = agent._tool_surface
        take_action = next(
            tool for tool in surface.native_tools() if tool.name == "take_action"
        )
        provider_schema = take_action.get_parameter_schema()

        action_error = None
        try:
            surface.dispatch(
                take_action,
                {"action": "UP", "row": 10, "col": 10},
                tool_call_id="conditional-schema-fix-probe",
                raw_arguments='{"action":"UP","row":10,"col":10}',
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
            if row["tool_call_id"] == "conditional-schema-fix-probe"
        )

        run_python_result = surface.run_python(
            "import json\n"
            "from pathlib import Path\n"
            "print(json.dumps(sorted("
            "path.name for pattern in ('.tycho-run-*.py', '.tycho-command-*.py') "
            "for path in Path('.').glob(pattern))))"
        )
        child_observed = json.loads(run_python_result)
        workspace_after = sorted(
            path.name
            for pattern in (".tycho-run-*.py", ".tycho-command-*.py")
            for path in agent.workspace_dir.glob(pattern)
        )

        frame_path = agent.workspace_dir / "level_0" / "turn_000.json"
        durable_frame = json.loads(frame_path.read_text(encoding="utf-8"))
        projected_actions = tycho_workspace_available_actions(durable_frame)

        assertions = {
            "audit_report_gate_absent": not hasattr(surface, "_audit_enabled")
            and not hasattr(surface, "_audit_evidence_revision"),
            "conditional_schema_published": isinstance(
                provider_schema.get("oneOf"), list
            )
            and len(provider_schema["oneOf"]) == 2
            and provider_schema.get("additionalProperties") is False,
            "invalid_coordinates_rejected_before_execution": action_trace[
                "validation_status"
            ]
            == "error"
            and action_trace["execution_status"] == "not_attempted"
            and action_trace["error_stage"] == "validation",
            "generated_source_absent_from_workspace": child_observed == []
            and workspace_after == [],
            "workspace_v1_projection_is_explicit": TYCHO_WORKSPACE_TURN_SCHEMA
            == "tycho-workspace-turn-v1"
            and projected_actions == turn.available_actions
            and durable_frame.get("available") == turn.available_actions
            and "available_actions" not in durable_frame,
        }
        if not all(assertions.values()):
            failed = sorted(name for name, passed in assertions.items() if not passed)
            raise AssertionError(f"post-fix assertions failed: {failed}")

        result = {
            "schema": "unified-interface-audit-fix-verification-v1",
            "assertions": assertions,
            "bound_sources": {
                "post_fix_probe": source_binding(__file__),
                "tycho_nooa_agent": source_binding(tycho_nooa_agent.__file__),
                "tycho_nooa_models": source_binding(tycho_nooa_models.__file__),
                "tycho_nooa_planner": source_binding(tycho_nooa_planner.__file__),
                "tycho_nooa_tools": source_binding(tycho_nooa_tools.__file__),
                "tycho_nooa_wm_probe": source_binding(tycho_nooa_wm_probe.__file__),
            },
            "conditional_action_schema": {
                "provider_schema": provider_schema,
                "invalid_non_click_coordinates": {
                    "input": {"action": "UP", "row": 10, "col": 10},
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
                "child_observed": child_observed,
                "workspace_after_return": workspace_after,
            },
            "action_vocabulary_contract": {
                "live_model_field": "available_actions",
                "durable_schema": TYCHO_WORKSPACE_TURN_SCHEMA,
                "durable_field": "available",
                "durable_frame_keys": sorted(durable_frame),
                "projected_available_actions": projected_actions,
            },
        }
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
