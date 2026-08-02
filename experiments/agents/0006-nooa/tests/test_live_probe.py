"""Tests for the bounded 0006-NOOA live-runner authority and evidence surface."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import runpy
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_live_probe.py"
MODULE = runpy.run_path(str(RUNNER))


def parsed_args() -> argparse.Namespace:
    return MODULE["build_parser"]().parse_args(
        [
            "--game",
            "ls20",
            "--model",
            "openai/google/gemma-4-31b-it",
            "--wire-model",
            "google/gemma-4-31b-it",
            "--api-base",
            "https://openrouter.ai/api/v1",
            "--run-root",
            "/tmp/0006-nooa-approved-run",
        ]
    )


class LiveProbeTests(unittest.TestCase):
    def test_approval_confirmation_binds_exact_spec_sources_and_run_identity(self) -> None:
        args = parsed_args()
        MODULE["_validate_live_args"](args)
        spec = MODULE["_paired_spec"](args)
        source_state = {
            "schema": "source-test",
            "source_sha256": {"runner.py": "a" * 64},
        }
        first = MODULE["_approval_request"](
            spec=spec,
            run_root=Path(args.run_root),
            source_state=source_state,
        )
        second = MODULE["_approval_request"](
            spec=spec,
            run_root=Path("/tmp/different-run"),
            source_state=source_state,
        )
        self.assertNotEqual(first["required_confirmation"], second["required_confirmation"])

        changed_state = deepcopy(source_state)
        changed_state["source_sha256"]["runner.py"] = "b" * 64
        third = MODULE["_approval_request"](
            spec=spec,
            run_root=Path(args.run_root),
            source_state=changed_state,
        )
        self.assertNotEqual(first["required_confirmation"], third["required_confirmation"])

        with tempfile.TemporaryDirectory() as directory:
            request_path = Path(directory) / "approval.json"
            request_path.write_text(json.dumps(first), encoding="utf-8")
            MODULE["_require_approved_request"](
                request_path,
                expected=first,
                confirmation=first["required_confirmation"],
            )
            with self.assertRaises(RuntimeError):
                MODULE["_require_approved_request"](
                    request_path,
                    expected=second,
                    confirmation=first["required_confirmation"],
                )
            consumed = MODULE["_consume_approved_request"](
                request_path,
                expected=first,
                confirmation=first["required_confirmation"],
            )
            self.assertFalse(request_path.exists())
            self.assertTrue(consumed.exists())
            with self.assertRaises(RuntimeError):
                MODULE["_consume_approved_request"](
                    request_path,
                    expected=first,
                    confirmation=first["required_confirmation"],
                )

    def test_runner_rejects_profile_drift_and_oversized_smoke(self) -> None:
        args = parsed_args()
        MODULE["_validate_live_args"](args)

        for field, value in (
            ("game", "other"),
            ("api_base", "https://direct.example/v1"),
            ("max_actions", 4),
            ("max_model_calls", 13),
            ("max_reported_tokens", 32_769),
            ("max_model_rounds_per_action", 5),
            ("max_decision_repairs", 3),
            ("transport_max_retries", 1),
            ("max_output_tokens", 32_769),
        ):
            invalid = deepcopy(args)
            setattr(invalid, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                MODULE["_validate_live_args"](invalid)

        incoherent = deepcopy(args)
        incoherent.max_reported_tokens = 8_000
        with self.assertRaises(ValueError):
            MODULE["_validate_live_args"](incoherent)

        for value in (float("nan"), float("inf"), float("-inf")):
            unbounded = deepcopy(args)
            unbounded.response_timeout_seconds = value
            with self.subTest(response_timeout_seconds=value), self.assertRaises(ValueError):
                MODULE["_validate_live_args"](unbounded)

    def test_default_response_timeout_budgets_workspace_per_round(self) -> None:
        self.assertEqual(MODULE["_response_timeout"](parsed_args()), 830.0)

    def test_source_state_commits_transitive_sources_locks_and_runtimes(self) -> None:
        source_state = MODULE["_source_state"]()
        sources = source_state["source_sha256"]
        self.assertIn(
            "experiments/agents/0006-capability-environment/p9/agent.py",
            sources,
        )
        self.assertIn(
            "experiments/agents/0002-llm-repl-agent/llm_repl_agent/observations.py",
            sources,
        )
        self.assertIn(
            "experiments/agents/0002-nooa/nooa-runtime/pyproject.toml",
            sources,
        )
        self.assertIn(
            "experiments/agents/0002-nooa/nooa-runtime/uv.lock",
            sources,
        )
        self.assertIn(
            "experiments/agents/0006-nooa/provider_runtime_launcher.py",
            sources,
        )
        self.assertIn("pyproject.toml", sources)
        self.assertIn("uv.lock", sources)
        self.assertTrue(source_state["frozen_control"]["matches"])
        self.assertIn(
            "experiments/agents/0002-nooa/host/responses_client.py",
            source_state["frozen_control"]["checked_paths"],
        )
        unexpected = MODULE["CONTROL_ROOT"] / "host" / "unexpected.py"
        drifted_control = MODULE["_frozen_control_record"](
            current_source_paths=[*MODULE["_source_paths"](), unexpected]
        )
        self.assertFalse(drifted_control["matches"])
        self.assertIn(
            unexpected.relative_to(MODULE["REPO"]).as_posix(),
            drifted_control["mismatches"],
        )
        self.assertTrue(source_state["runtime_versions"]["host_python"])
        self.assertTrue(source_state["runtime_versions"]["nooa_python"])

        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            snapshot_root = MODULE["_snapshot_participating_sources"](
                run_root,
                expected_sha256=source_state["source_sha256"],
            )
            copied = snapshot_root / "experiments/agents/0006-nooa/provider_runtime_launcher.py"
            self.assertEqual(
                copied.read_bytes(),
                (ROOT / "provider_runtime_launcher.py").read_bytes(),
            )
            runtime_root = MODULE["_materialize_runtime_overlay"](
                run_root,
                expected_sha256=source_state["source_sha256"],
            )
            transport = json.loads((runtime_root / "transport-config.json").read_text())
            self.assertEqual(
                Path(transport["base_runtime_root"]),
                snapshot_root / "experiments/agents/0002-nooa/nooa-runtime",
            )
            self.assertEqual((runtime_root / "launcher.py").read_bytes(), copied.read_bytes())

        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            snapshot_root = MODULE["_snapshot_participating_sources"](
                run_root,
                expected_sha256=source_state["source_sha256"],
            )
            copied = snapshot_root / "experiments/agents/0006-nooa/provider_runtime_launcher.py"
            copied.write_text("# drift after approval\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "source snapshot"):
                MODULE["_materialize_runtime_overlay"](
                    run_root,
                    expected_sha256=source_state["source_sha256"],
                )

    def test_provider_report_projects_actual_crossing(self) -> None:
        args = parsed_args()
        profile = MODULE["_paired_spec"](args).provider_request_profile
        crossing = {
            "request": {
                "url": "https://openrouter.ai/api/v1/responses",
                "method": "POST",
                "body": {
                    "model": "google/gemma-4-31b-it",
                    "max_output_tokens": 16_384,
                    "reasoning": {"effort": "medium"},
                    "stream": False,
                    "truncation": "disabled",
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [{"type": "function"}],
                    "provider": {
                        "order": ["modelrun/fp4"],
                        "only": ["modelrun/fp4"],
                        "allow_fallbacks": False,
                        "require_parameters": False,
                        "quantizations": ["fp4"],
                    },
                },
            }
        }
        report = MODULE["_provider_profile_report"](profile, [crossing])
        self.assertTrue(report["success"])
        drifted = deepcopy(crossing)
        drifted["request"]["body"]["model"] = "different/model"
        self.assertFalse(MODULE["_provider_profile_report"](profile, [drifted])["success"])
        unpinned = deepcopy(crossing)
        del unpinned["request"]["body"]["provider"]
        self.assertFalse(MODULE["_provider_profile_report"](profile, [unpinned])["success"])
        wrong_origin = deepcopy(crossing)
        wrong_origin["request"]["url"] = "https://evil.example/responses"
        self.assertFalse(
            MODULE["_provider_profile_report"](profile, [wrong_origin])["success"]
        )
        wrong_method = deepcopy(crossing)
        wrong_method["request"]["method"] = "GET"
        self.assertFalse(
            MODULE["_provider_profile_report"](profile, [wrong_method])["success"]
        )

        crossing["response"] = {
            "headers": {"x-generation-id": "gen-test"},
            "body": {"id": "gen-test"},
        }
        self.assertEqual(MODULE["_generation_ids"]([crossing]), ["gen-test"])
        valid_payload = {
            "data": {
                "id": "gen-test",
                "model": "google/gemma-4-31b-it-20260402",
                "provider_name": "ModelRun",
                "provider_responses": [
                    {
                        "provider_name": "ModelRun",
                        "status": 200,
                        "model_permaslug": "google/gemma-4-31b-it-20260402",
                    }
                ],
                "native_finish_reason": "tool_calls",
                "finish_reason": "tool_calls",
                "native_tokens_prompt": 100,
                "native_tokens_completion": 20,
                "native_tokens_reasoning": 5,
                "total_cost": 0.001,
            }
        }
        identity = MODULE["_provider_identity_report"](
            ["gen-test"],
            [valid_payload],
            expected_wire_model="google/gemma-4-31b-it",
            crossing_usage={
                "gen-test": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "reasoning_tokens": 5,
                    "cost_usd": 0.001,
                }
            },
        )
        self.assertTrue(identity["success"])
        self.assertEqual(identity["actual_quantization"], "fp4")
        for field, value in (
            ("provider_name", "CoreWeave"),
            ("model", "unrelated/model"),
            ("native_finish_reason", None),
            ("native_tokens_prompt", None),
            ("total_cost", None),
        ):
            bad_payload = deepcopy(valid_payload)
            bad_payload["data"][field] = value
            self.assertFalse(
                MODULE["_provider_identity_report"](
                    ["gen-test"],
                    [bad_payload],
                    expected_wire_model="google/gemma-4-31b-it",
                    crossing_usage={
                        "gen-test": {
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "reasoning_tokens": 5,
                            "cost_usd": 0.001,
                        }
                    },
                )["success"]
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = MODULE["_generation_artifact_path"](root, "gen-test_123")
            self.assertEqual(valid_path.parent, root / "openrouter-generations")
            for invalid_id in ("../escape", "/absolute", "gen-x/y", "not-a-generation"):
                with self.assertRaisesRegex(ValueError, "generation ID"):
                    MODULE["_generation_artifact_path"](root, invalid_id)
        handler = MODULE["_NoRedirectHandler"]()
        self.assertIsNone(
            handler.redirect_request(None, None, 302, "Found", {}, "https://evil.example")
        )

    def test_runtime_overlay_injects_exact_modelrun_fp4_pin(self) -> None:
        overlay_module = runpy.run_path(str(ROOT / "provider_runtime_launcher.py"))

        class FakeClient:
            def __init__(self) -> None:
                self.config: dict[str, object] = {}

            def close(self) -> None:
                raise AssertionError("clean client must not be closed")

        client = FakeClient()
        overlay_module["_pin_client"](client, MODULE["PROVIDER_ROUTE"])
        self.assertEqual(
            client.config,
            {"extra_body": {"provider": MODULE["PROVIDER_ROUTE"]}},
        )

        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            source_state = MODULE["_source_state"]()
            MODULE["_snapshot_participating_sources"](
                run_root,
                expected_sha256=source_state["source_sha256"],
            )
            runtime_root = MODULE["_materialize_runtime_overlay"](
                run_root,
                expected_sha256=source_state["source_sha256"],
            )

            config = json.loads((runtime_root / "transport-config.json").read_text())
            self.assertEqual(config["provider"], MODULE["PROVIDER_ROUTE"])
            self.assertFalse(config["provider"]["require_parameters"])
            self.assertTrue((runtime_root / ".venv").is_symlink())
            self.assertEqual(
                (runtime_root / ".venv").resolve(),
                (MODULE["CONTROL_RUNTIME_ROOT"] / ".venv").resolve(),
            )
            help_result = subprocess.run(
                [
                    str(runtime_root / ".venv" / "bin" / "python"),
                    str(runtime_root / "launcher.py"),
                    "--help",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn("--provider", help_result.stdout)

    def test_operational_verdict_requires_predecessor_result_exposure(self) -> None:
        summary = {
            "actions": 2,
            "attempted_actions": 2,
            "observed_successors": 2,
            "stop_reason": "action_budget",
            "clean_shutdown": True,
            "trace_complete": True,
        }
        events = [
            {"type": "observation"},
            {"type": "context_assembled"},
            {"type": "commitment", "claim_ids": ["claim-control"]},
            {"type": "transition_occurrence"},
            {"type": "prediction_result", "outcome": "match"},
            {"type": "observation"},
            {"type": "context_assembled"},
            {"type": "commitment", "claim_ids": ["claim-control"]},
        ]
        verdict = MODULE["_operational_verdict"](
            summary,
            competence_events=events,
            provider_success=True,
            required_actions=2,
        )
        self.assertTrue(verdict["success"])
        self.assertFalse(
            MODULE["_operational_verdict"](
                summary,
                competence_events=events[:5],
                provider_success=True,
                required_actions=2,
            )["success"]
        )
        self.assertFalse(
            MODULE["_operational_verdict"](
                summary,
                competence_events=events,
                provider_success=False,
                required_actions=2,
            )["success"]
        )
        no_continuity = [dict(event) for event in events]
        no_continuity[-1] = {"type": "commitment", "claim_ids": ["claim-other"]}
        self.assertFalse(
            MODULE["_operational_verdict"](
                summary,
                competence_events=no_continuity,
                provider_success=True,
                required_actions=2,
            )["success"]
        )
        overrun = {**summary, "actions": 3, "attempted_actions": 3, "observed_successors": 3}
        self.assertFalse(
            MODULE["_operational_verdict"](
                overrun,
                competence_events=events,
                provider_success=True,
                required_actions=2,
            )["success"]
        )
        unreconciled = [dict(event) for event in events]
        unreconciled[4] = {"type": "prediction_result", "outcome": "mismatch"}
        self.assertFalse(
            MODULE["_operational_verdict"](
                summary,
                competence_events=unreconciled,
                provider_success=True,
                required_actions=2,
            )["success"]
        )
        reconciled = [*unreconciled[:5], {"type": "reconciliation"}, *unreconciled[5:]]
        self.assertTrue(
            MODULE["_operational_verdict"](
                summary,
                competence_events=reconciled,
                provider_success=True,
                required_actions=2,
            )["success"]
        )

    def test_one_action_smoke_passes_on_admission_and_one_successor(self) -> None:
        verdict = MODULE["_operational_verdict"](
            {
                "actions": 1,
                "attempted_actions": 1,
                "observed_successors": 1,
                "stop_reason": "action_budget",
                "clean_shutdown": True,
                "trace_complete": True,
            },
            competence_events=[{"type": "commitment"}],
            provider_success=True,
            required_actions=1,
        )

        self.assertEqual(verdict["goal"], "one-action-admission")
        self.assertTrue(verdict["success"])
        self.assertNotIn("predecessor_prediction_result_exposed", verdict["checks"])

    def test_failure_summary_preserves_phase_and_error(self) -> None:
        summary = MODULE["_failure_summary"](
            game="ls20",
            environment_seed=0,
            run_root=Path("/tmp/run"),
            phase="environment_create",
            error=RuntimeError("boom"),
        )
        self.assertEqual(summary["stop_reason"], "stage_exception")
        self.assertEqual(summary["failure_phase"], "environment_create")
        self.assertEqual(summary["error"], {"type": "RuntimeError", "message": "boom"})
        self.assertFalse(summary["clean_shutdown"])
        self.assertFalse(summary["trace_complete"])

    def test_failure_summary_recovers_attempts_and_successors_from_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            (run_root / "worker").mkdir()
            (run_root / "trace.jsonl").write_text(
                json.dumps({"type": "transition"}) + "\n",
                encoding="utf-8",
            )
            (run_root / "worker" / "process.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"kind": "competence_action_admitted"}),
                        json.dumps({"kind": "transition_recorded"}),
                        json.dumps({"kind": "competence_action_admitted"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            summary = MODULE["_failure_summary"](
                game="ls20",
                environment_seed=0,
                run_root=run_root,
                phase="environment_step",
                error=RuntimeError("boom"),
            )
            self.assertEqual(summary["attempted_actions"], 2)
            self.assertEqual(summary["observed_successors"], 1)
            self.assertEqual(summary["actions"], 1)

    def test_stage_exception_still_writes_terminal_summary(self) -> None:
        args = parsed_args()
        spec = MODULE["_paired_spec"](args)
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            run_root.mkdir()
            with mock.patch.object(
                MODULE["arc_agi"],
                "Arcade",
                side_effect=RuntimeError("arcade unavailable"),
            ):
                summary = MODULE["_run_stage"](
                    args,
                    spec=spec,
                    run_root=run_root,
                    response_timeout=830.0,
                    expected_source_sha256=MODULE["_source_state"]()["source_sha256"],
                )
            persisted = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary, persisted)
            self.assertEqual(summary["failure_phase"], "arcade_init")
            self.assertEqual(summary["stop_reason"], "stage_exception")

    def test_no_spend_preparation_and_missing_key_leave_approval_unconsumed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / "approval.json"
            run_root = root / "live-run"
            args = [
                "--game",
                "ls20",
                "--model",
                "openai/google/gemma-4-31b-it",
                "--wire-model",
                "google/gemma-4-31b-it",
                "--api-base",
                "https://openrouter.ai/api/v1",
                "--run-root",
                str(run_root),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    MODULE["main"]([*args, "--prepare-approval-request", str(request)]),
                    0,
                )
            approval = json.loads(request.read_text(encoding="utf-8"))
            with (
                mock.patch.dict("os.environ", {}, clear=True),
                self.assertRaisesRegex(RuntimeError, "ARC_LLM_API_KEY"),
            ):
                MODULE["main"](
                    [
                        *args,
                        "--approval-request",
                        str(request),
                        "--confirm-live-run",
                        approval["required_confirmation"],
                    ]
                )
            self.assertTrue(request.exists())
            self.assertFalse(request.with_name(request.name + ".consumed").exists())
            self.assertFalse(run_root.exists())


if __name__ == "__main__":
    unittest.main()
