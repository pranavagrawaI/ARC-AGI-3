"""Tests for the bounded 0006-NOOA live-runner authority and evidence surface."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_live_probe.py"
MODULE = runpy.run_path(str(RUNNER))
MODULE_CHAIN = runpy.run_path(str(ROOT / "evidence_chain.py"))

from competence import CompetenceController
from competence_runner import EXPECTED_PROJECTION_DTYPE, attach_competence_context
from evidence_chain import (
    _changed_pixels,
    canonical_digest,
    competence_packet_digest,
    two_action_chain_report,
)


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


def two_action_fixture(
    *, mismatch: bool = False, retire_claim: bool = False
) -> tuple[list[dict], list[dict]]:
    controller = CompetenceController()
    public = {
        "step": 0,
        "pixels": [[[0]]],
        "available_actions": [1, 2],
        "levels_completed": 0,
        "state": "NOT_FINISHED",
        "full_reset": True,
    }
    controller.receive_observation(public)
    first_packet = controller.context_packet()
    first_decision = {
            "action": 1,
            "data": None,
            "working_state": {},
            "test": "first",
            "claim_ids": ["claim-control"],
            "ledger": [
                {
                    "op": "create",
                    "claim_id": "claim-control",
                    "statement": "Action 1 is under test.",
                }
            ],
            "assertions": [
                {
                    "observable": "changed_cells",
                    "op": ">",
                    "value": 9 if mismatch else 0,
                }
            ],
            "evidence_requests": [],
            "reconciliation": None,
        }
    first = controller.admit(first_decision, legal_actions=[1, 2])
    successor = {**public, "step": 1, "pixels": [[[1]]], "full_reset": False}
    controller.receive_observation(successor)
    second_packet = controller.context_packet()
    reconciliation = None
    if mismatch:
        unresolved = second_packet["unresolved_mismatches"]
        assert isinstance(unresolved, list) and isinstance(unresolved[0], dict)
        mismatch_id = unresolved[0]["mismatch_id"]
        reconciliation = [
            {"mismatch_id": mismatch_id, "disposition": "NARROW", "note": "revise"}
        ]
    second_decision = {
            "action": 2,
            "data": None,
            "working_state": {},
            "test": "second",
            # Retiring the first action's claim is legitimate continuity, but a
            # commitment still has to work from some claim, so the retirement
            # comes with the claim that replaces it.
            "claim_ids": ["claim-successor"] if retire_claim else ["claim-control"],
            "ledger": (
                [
                    {"op": "retire", "claim_id": "claim-control"},
                    {
                        "op": "create",
                        "claim_id": "claim-successor",
                        "statement": "Action 2 is under test instead.",
                    },
                ]
                if retire_claim
                else []
            ),
            "assertions": [
                {"observable": "levels_completed", "op": "==", "value": 0}
            ],
            "evidence_requests": [],
            "reconciliation": reconciliation,
        }
    second = controller.admit(second_decision, legal_actions=[1, 2])
    final_successor = {**successor, "step": 2}
    controller.receive_observation(final_successor)
    process_protocol = "0002-nooa-ipc-v1"
    run_id = "fixture-run"
    process_records: list[dict] = [
        {
            "protocol": process_protocol,
            "kind": "worker_started",
            "run_id": run_id,
            "pid": 1,
            "runtime_python": "/fixture/python",
        }
    ]
    paired_run_spec = {
        key: None
        for key in (
            "allowed_distributions",
            "allowed_python_policy",
            "coordinate_convention",
            "environment_profile",
            "environment_seed",
            "final_action_schema",
            "host_workspace_files",
            "host_workspace_initialization",
            "include_changed_pixel_counts",
            "legal_event_schema",
            "max_actions",
            "max_context_turns",
            "max_decision_repairs",
            "max_model_calls",
            "max_model_rounds_per_action",
            "max_recent_transitions",
            "max_reported_tokens",
            "model_profile",
            "model_seed",
            "observation_representations",
            "omission_notice_schema",
            "provider_endpoint_profile",
            "provider_request_profile",
            "public_status_fields",
            "request_profile",
            "transition_observation_policy",
            "transport_max_retries",
            "usability_probe_instruction",
            "workspace_timeout_seconds",
        )
    }

    def projection(public_observation: dict) -> dict:
        pixels = np.asarray(public_observation["pixels"], dtype=EXPECTED_PROJECTION_DTYPE)
        return {
            "digest": sha256(pixels.tobytes()).hexdigest()[:16],
            "dtype": pixels.dtype.str,
            "shape": list(pixels.shape),
        }

    def representations(public_observation: dict) -> dict:
        pixels = public_observation["pixels"]
        return {
            "grid": {
                "axes": ["layer", "row", "column"],
                "encoding": "nested_integer_array",
                "json_pointer": "/pixels",
                "json_value_type": "integer",
                "path": "current_observation.json",
                "shape": [len(pixels), len(pixels[-1]), len(pixels[-1][0])],
                "values": "integer_color_ids",
            },
            "rendered_image": {
                "annotation": {
                    "cell_pixels": 7,
                    "gridline_every": 10,
                    "label_every": 5,
                    "margin_pixels": 24,
                    "note": "fixture coordinate note",
                    "rulers": "grid coordinates on the top and left margins",
                    "x_axis": "column, labelled along the top",
                    "y_axis": "row, labelled down the left",
                },
                "axes": ["row", "column", "channel"],
                "dtype": "uint8",
                "encoding": "rgb",
                "path": "current_frame.png",
                "shape": [len(pixels[-1]) * 8, len(pixels[-1][0]) * 8, 3],
                "source_grid_layer": len(pixels) - 1,
                "source_grid_scale": 8,
            },
        }

    def model_call(
        *,
        worker_decision: dict,
        provider_request_profile: object,
        sequence: int,
    ) -> dict:
        canonical_request = {
            "provider_request_profile": provider_request_profile,
            "messages": [{"role": "user", "content": "fixture"}],
            "tools": [],
            "kwargs": {},
        }
        crossing = {"request": {}, "response": {}, "sequence": sequence}
        return {
            "schema": "0002-nooa-provider-crossing-v1",
            "messages": canonical_request["messages"],
            "tools": canonical_request["tools"],
            "canonical_request": canonical_request,
            "canonical_request_sha256": canonical_digest(canonical_request),
            "canonical_audit_request_sha256": canonical_digest(canonical_request),
            "canonicalization": "json-utf8-sort-keys-compact-v1",
            "actual_provider_crossings": [crossing],
            "attempts": [
                {
                    "attempt": 1,
                    "outcome": "success",
                    "actual_provider_crossings": [crossing],
                }
            ],
            "outcome": {
                "kind": "success",
                "response": {
                    "tool_calls": [
                        {
                            "arguments": json.dumps({"result": worker_decision}),
                            "id": f"call-{sequence}",
                            "name": "return_result",
                        }
                    ]
                },
            },
        }

    prior_transitions: list[dict] = []
    for action_point, packet, commitment, authored, before, after in (
        (0, first_packet, first, first_decision, public, successor),
        (1, second_packet, second, second_decision, successor, final_successor),
    ):
        reason = json.dumps(
            {
                key: value
                for key, value in authored.items()
                if key not in {"action", "data"}
            },
            separators=(",", ":"),
        )
        transition = {
            "step": action_point + 1,
            "action": commitment["action"],
            "action_data": commitment["data"],
            "before_digest": projection(before)["digest"],
            "after_digest": projection(after)["digest"],
            "changed_pixels": _changed_pixels(before, after),
            "levels_before": before["levels_completed"],
            "levels_after": after["levels_completed"],
            "state_before": before["state"],
            "state_after": after["state"],
            "full_reset": after["full_reset"],
            "available_actions_after": after["available_actions"],
            "before_file": f"observations/step-{action_point:05d}.json",
            "after_file": f"observations/step-{action_point + 1:05d}.json",
        }
        request = {
            "protocol": process_protocol,
            "run_id": run_id,
            "action_point_id": action_point,
            "observation": {
                "step": action_point,
                "current_metadata": {
                    "available_actions": before["available_actions"],
                    "action_requirements": {
                        str(action): {"data": None}
                        for action in before["available_actions"]
                    },
                    "levels_completed": before["levels_completed"],
                    "state": before["state"],
                    "full_reset": before["full_reset"],
                    "frame": {"layers": 1, "height": 1, "width": 1},
                },
                "representations": representations(before),
                "recent_transitions": deepcopy(prior_transitions),
                "full_history_file": "public_history.jsonl",
                "current_observation_file": "current_observation.json",
                "observations_glob": "observations/step-*.json",
                "competence": attach_competence_context({}, packet)["competence"]
            },
            "legal_actions": [1, 2],
            "coordinate_actions": [],
            "frame_width": 1,
            "frame_height": 1,
            "paired_run_spec": paired_run_spec,
        }
        worker_decision = {
            "action": commitment["action"],
            "data": commitment["data"],
            "reason": reason,
        }
        process_records.extend(
            [
                {
                    "protocol": process_protocol,
                    "kind": "competence_context_assembled",
                    "run_id": run_id,
                    "action_point_id": action_point,
                    "step": action_point,
                    "current_evidence_id": commitment["before_evidence_id"],
                    "competence_packet_sha256": canonical_digest(packet),
                    "projection": projection(before),
                },
                {
                    "protocol": process_protocol,
                    "kind": "action_request",
                    "run_id": run_id,
                    "action_point_id": action_point,
                    "step": action_point,
                    "host_observation": {
                        "path": f"host-observations/step-{action_point:05d}.json",
                        "sha256": sha256(
                            json.dumps(
                                before, sort_keys=True, separators=(",", ":")
                            ).encode()
                        ).hexdigest(),
                    },
                    "request_sha256": canonical_digest(request),
                    "request": request,
                },
                {
                    "protocol": process_protocol,
                    "kind": "model_crossing",
                    "run_id": run_id,
                    "action_point_id": action_point,
                    "step": action_point,
                    "audit": {
                        "model_calls": [
                            model_call(
                                worker_decision=worker_decision,
                                provider_request_profile=paired_run_spec[
                                    "provider_request_profile"
                                ],
                                # Zero-based, as the frozen recorder numbers
                                # them. The fixture used to start at one, which
                                # is why no test saw the verifier reject the
                                # first crossing of every real run.
                                sequence=action_point,
                            )
                        ]
                    },
                    "worker_decision": worker_decision,
                    "worker_error": None,
                },
                {
                    "protocol": process_protocol,
                    "kind": "decision_validated",
                    "run_id": run_id,
                    "action_point_id": action_point,
                    "step": action_point,
                    "decision": {
                        "action": commitment["action"],
                        "data": commitment["data"],
                        "reasoning": {
                            "reason": reason,
                            "competence": commitment,
                        },
                    },
                    "workspace_commitment": "b" * 64,
                },
                {
                    "protocol": process_protocol,
                    "kind": "competence_action_admitted",
                    "run_id": run_id,
                    "action_point_id": action_point,
                    "step": action_point,
                    "commitment": commitment,
                },
                {
                    "protocol": process_protocol,
                    "kind": "transition_recorded",
                    "run_id": run_id,
                    "action_point_id": action_point,
                    "transition": transition,
                },
                {
                    "protocol": process_protocol,
                    "kind": "competence_successor_observed",
                    "run_id": run_id,
                    "action_point_id": action_point,
                    "successor_step": action_point + 1,
                    "evidence_id": next(
                        event["record"]["evidence_id"]
                        for event in controller.events
                        if event.get("type") == "observation"
                        and event.get("record", {}).get("step") == action_point + 1
                    ),
                    "host_observation": {
                        "path": (
                            f"host-observations/step-{action_point + 1:05d}.json"
                        ),
                        "sha256": sha256(
                            json.dumps(
                                after, sort_keys=True, separators=(",", ":")
                            ).encode()
                        ).hexdigest(),
                    },
                    "projection": projection(after),
                },
            ]
        )
        prior_transitions.append(transition)
    process_records.append(
        {
            "protocol": process_protocol,
            "kind": "worker_stopped",
            "run_id": run_id,
            "returncode": 0,
            "stderr": "",
        }
    )
    return controller.events, process_records


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
            route_name="pinned",
        )
        second = MODULE["_approval_request"](
            spec=spec,
            run_root=Path("/tmp/different-run"),
            source_state=source_state,
            route_name="pinned",
        )
        self.assertNotEqual(first["required_confirmation"], second["required_confirmation"])

        changed_state = deepcopy(source_state)
        changed_state["source_sha256"]["runner.py"] = "b" * 64
        third = MODULE["_approval_request"](
            spec=spec,
            run_root=Path(args.run_root),
            source_state=changed_state,
            route_name="pinned",
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
            # Derived so a changed ceiling cannot leave the test asserting
            # against a bound the runner no longer has.
            *(
                (field, maximum + 1)
                for field, maximum in MODULE["SMOKE_MAXIMA"].items()
            ),
            ("transport_max_retries", 1),
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
        self.assertIn("experiments/__init__.py", sources)
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
        self.assertIn("experiments/agents/0006-nooa/evidence_chain.py", sources)
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
        self.assertEqual(source_state["runtime_environment"]["schema"], "0006-nooa-runtime-environment-v1")
        self.assertTrue(source_state["runtime_environment"]["tree_sha256"])
        loaded_project_sources: set[str] = set()
        for module in sys.modules.values():
            module_file = getattr(module, "__file__", None)
            if not isinstance(module_file, str):
                continue
            module_path = Path(module_file).resolve()
            if (
                module_path.is_relative_to(MODULE["REPO"])
                and ".venv" not in module_path.parts
                and "tests" not in module_path.parts
            ):
                loaded_project_sources.add(
                    module_path.relative_to(MODULE["REPO"]).as_posix()
                )
        self.assertEqual(loaded_project_sources - set(sources), set())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_repo = root / "repo"
            real_directory = fake_repo / "real-directory"
            real_directory.mkdir(parents=True)
            source = real_directory / "source.py"
            source.write_bytes(b"approved source\n")
            source.chmod(0o644)
            (fake_repo / "linked-directory").symlink_to(
                real_directory, target_is_directory=True
            )
            relative = "linked-directory/source.py"
            expected_sha256 = sha256(source.read_bytes()).hexdigest()
            production_globals = MODULE["_snapshot_participating_sources"].__globals__
            with mock.patch.dict(production_globals, {"REPO": fake_repo}):
                self.assertTrue(
                    MODULE["_has_symlink_component"](
                        fake_repo / relative, root=fake_repo
                    )
                )
                with self.assertRaisesRegex(RuntimeError, "symlink component"):
                    MODULE["_snapshot_participating_sources"](
                        root / "run",
                        expected_sha256={relative: expected_sha256},
                        expected_mode={relative: "644"},
                    )
            source_state_globals = MODULE["_source_state"].__globals__
            with mock.patch.dict(
                source_state_globals,
                {
                    "REPO": fake_repo,
                    "_runtime_environment_record": lambda _root: {
                        "interpreter_version": "3.14.0"
                    },
                    "_frozen_control_record": lambda: {
                        "matches": True,
                        "mismatches": [],
                    },
                    "_participating_sources": lambda: {
                        relative: expected_sha256
                    },
                },
            ), self.assertRaisesRegex(RuntimeError, "symlink components"):
                MODULE["_source_state"]()

        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            snapshot_root = MODULE["_snapshot_participating_sources"](
                run_root,
                expected_sha256=source_state["source_sha256"],
                expected_mode=source_state["source_mode"],
            )
            copied = snapshot_root / "experiments/agents/0006-nooa/provider_runtime_launcher.py"
            self.assertEqual(
                copied.read_bytes(),
                (ROOT / "provider_runtime_launcher.py").read_bytes(),
            )
            undeclared = snapshot_root / "__pycache__" / "runner.pyc"
            undeclared.parent.mkdir()
            undeclared.write_bytes(b"undeclared")
            with self.assertRaisesRegex(RuntimeError, "source snapshot tree"):
                MODULE["_verify_source_snapshot"](
                    snapshot_root,
                    expected_sha256=source_state["source_sha256"],
                expected_mode=source_state["source_mode"],
                )
            undeclared.unlink()
            undeclared.parent.rmdir()
            record_path = snapshot_root / "record.json"
            approved_record = json.loads(record_path.read_text(encoding="utf-8"))
            record_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "snapshot record"):
                MODULE["_verify_source_snapshot"](
                    snapshot_root,
                    expected_sha256=source_state["source_sha256"],
                expected_mode=source_state["source_mode"],
                )
            MODULE["_write_json"](record_path, approved_record)
            record_path.chmod(0o644)
            record_path.unlink()
            with self.assertRaisesRegex(RuntimeError, "snapshot record"):
                MODULE["_verify_source_snapshot"](
                    snapshot_root,
                    expected_sha256=source_state["source_sha256"],
                expected_mode=source_state["source_mode"],
                )
            MODULE["_write_json"](record_path, approved_record)
            record_path.chmod(0o644)
            external_source = run_root / "external-launcher.py"
            external_source.write_bytes(copied.read_bytes())
            copied.unlink()
            copied.symlink_to(external_source)
            with self.assertRaisesRegex(RuntimeError, "source snapshot"):
                MODULE["_verify_source_snapshot"](
                    snapshot_root,
                    expected_sha256=source_state["source_sha256"],
                expected_mode=source_state["source_mode"],
                )
            copied.unlink()
            copied.write_bytes(external_source.read_bytes())
            copied.chmod(0o755)
            with self.assertRaisesRegex(RuntimeError, "approved hashes"):
                MODULE["_verify_source_snapshot"](
                    snapshot_root,
                    expected_sha256=source_state["source_sha256"],
                    expected_mode=source_state["source_mode"],
                )
            copied.chmod(
                int(
                    source_state["source_mode"][
                        "experiments/agents/0006-nooa/provider_runtime_launcher.py"
                    ],
                    8,
                )
            )
            empty_directory = snapshot_root / "undeclared-empty-directory"
            empty_directory.mkdir()
            with self.assertRaisesRegex(RuntimeError, "directory_extras"):
                MODULE["_verify_source_snapshot"](
                    snapshot_root,
                    expected_sha256=source_state["source_sha256"],
                    expected_mode=source_state["source_mode"],
                )
            empty_directory.rmdir()
            snapshot_root.chmod(0o700)
            with self.assertRaisesRegex(RuntimeError, "root mode"):
                MODULE["_verify_source_snapshot"](
                    snapshot_root,
                    expected_sha256=source_state["source_sha256"],
                    expected_mode=source_state["source_mode"],
                )
            snapshot_root.chmod(0o755)
            runtime_source = run_root / "approved-venv"
            (runtime_source / "bin").mkdir(parents=True)
            external_interpreter = run_root / "external-python"
            external_interpreter.write_bytes(b"approved interpreter\n")
            external_interpreter.chmod(0o755)
            (runtime_source / "bin" / "python").symlink_to(external_interpreter)
            (runtime_source / "pyvenv.cfg").write_text(
                "version = 3.14.0\n", encoding="utf-8"
            )
            (runtime_source / "marker.txt").write_text("approved", encoding="utf-8")
            runtime_environment = MODULE["_runtime_environment_record"](runtime_source)
            self.assertTrue(runtime_environment["external_symlinks"])
            linked_runtime = run_root / "linked-venv"
            linked_runtime.symlink_to(runtime_source, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "real directory"):
                MODULE["_runtime_environment_record"](linked_runtime)
            runtime_root = MODULE["_materialize_runtime_overlay"](
                run_root,
                route=MODULE["PROVIDER_ROUTES"]["pinned"],
                expected_sha256=source_state["source_sha256"],
                expected_mode=source_state["source_mode"],
                runtime_environment_source=runtime_source,
                expected_runtime_environment=runtime_environment,
            )
            transport = json.loads((runtime_root / "transport-config.json").read_text())
            self.assertEqual(
                Path(transport["base_runtime_root"]),
                snapshot_root / "experiments/agents/0002-nooa/nooa-runtime",
            )
            self.assertEqual((runtime_root / "launcher.py").read_bytes(), copied.read_bytes())
            self.assertFalse((runtime_root / ".venv").is_symlink())
            self.assertEqual(
                MODULE["_runtime_environment_record"](runtime_root / ".venv"),
                runtime_environment,
            )
            runtime_overlay = MODULE["_runtime_overlay_record"](runtime_root)
            external_record = runtime_overlay["environment"]["external_symlinks"][0]
            self.assertEqual(external_record["target_type"], "file")
            self.assertEqual(external_record["target_mode"], "755")
            external_interpreter.chmod(0o700)
            with self.assertRaisesRegex(RuntimeError, "runtime overlay"):
                MODULE["_verify_runtime_overlay"](
                    runtime_root, expected=runtime_overlay
                )
            external_interpreter.chmod(0o755)
            runtime_root.chmod(0o700)
            with self.assertRaisesRegex(RuntimeError, "runtime overlay"):
                MODULE["_verify_runtime_overlay"](
                    runtime_root, expected=runtime_overlay
                )
            runtime_root.chmod(0o755)
            real_runtime_root = run_root / "provider-runtime-real"
            runtime_root.rename(real_runtime_root)
            runtime_root.symlink_to(real_runtime_root, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "real directory"):
                MODULE["_verify_runtime_overlay"](
                    runtime_root, expected=runtime_overlay
                )
            runtime_root.unlink()
            real_runtime_root.rename(runtime_root)
            for name in ("launcher.py", "transport-config.json"):
                runtime_file = runtime_root / name
                approved_bytes = runtime_file.read_bytes()
                runtime_file.write_bytes(approved_bytes + b"\n# drift")
                with self.assertRaisesRegex(RuntimeError, "runtime overlay"):
                    MODULE["_verify_runtime_overlay"](
                        runtime_root, expected=runtime_overlay
                    )
                runtime_file.write_bytes(approved_bytes)
            copied_marker = runtime_root / ".venv" / "marker.txt"
            approved_marker_mode = copied_marker.stat().st_mode
            copied_marker.chmod(0o600)
            with self.assertRaisesRegex(RuntimeError, "runtime overlay"):
                MODULE["_verify_runtime_overlay"](
                    runtime_root, expected=runtime_overlay
                )
            copied_marker.chmod(approved_marker_mode)
            copied_environment = runtime_root / ".venv"
            approved_root_mode = copied_environment.stat().st_mode
            copied_environment.chmod(0o700)
            with self.assertRaisesRegex(RuntimeError, "runtime overlay"):
                MODULE["_verify_runtime_overlay"](
                    runtime_root, expected=runtime_overlay
                )
            copied_environment.chmod(approved_root_mode)

        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            snapshot_root = MODULE["_snapshot_participating_sources"](
                run_root,
                expected_sha256=source_state["source_sha256"],
                expected_mode=source_state["source_mode"],
            )
            copied = snapshot_root / "experiments/agents/0006-nooa/provider_runtime_launcher.py"
            copied.write_text("# drift after approval\n", encoding="utf-8")
            runtime_source = run_root / "approved-venv"
            (runtime_source / "bin").mkdir(parents=True)
            (runtime_source / "bin" / "python").symlink_to(Path(sys.executable).resolve())
            (runtime_source / "pyvenv.cfg").write_text(
                "version = 3.14.0\n", encoding="utf-8"
            )
            runtime_environment = MODULE["_runtime_environment_record"](runtime_source)
            with self.assertRaisesRegex(RuntimeError, "source snapshot"):
                MODULE["_materialize_runtime_overlay"](
                    run_root,
                    route=MODULE["PROVIDER_ROUTES"]["pinned"],
                    expected_sha256=source_state["source_sha256"],
                expected_mode=source_state["source_mode"],
                    runtime_environment_source=runtime_source,
                    expected_runtime_environment=runtime_environment,
                )

        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            MODULE["_snapshot_participating_sources"](
                run_root,
                expected_sha256=source_state["source_sha256"],
                expected_mode=source_state["source_mode"],
            )
            runtime_source = run_root / "approved-venv"
            (runtime_source / "bin").mkdir(parents=True)
            (runtime_source / "bin" / "python").symlink_to(
                Path(sys.executable).resolve()
            )
            (runtime_source / "pyvenv.cfg").write_text(
                "version = 3.14.0\n", encoding="utf-8"
            )
            runtime_environment = MODULE["_runtime_environment_record"](runtime_source)
            original_copytree = MODULE["shutil"].copytree

            def copytree_then_mutate(*args: object, **kwargs: object) -> object:
                result = original_copytree(*args, **kwargs)
                launcher = run_root / "provider-runtime" / "launcher.py"
                launcher.write_bytes(launcher.read_bytes() + b"\n# pre-baseline drift\n")
                return result

            with mock.patch.object(
                MODULE["shutil"], "copytree", side_effect=copytree_then_mutate
            ), self.assertRaisesRegex(RuntimeError, "runtime overlay"):
                MODULE["_materialize_runtime_overlay"](
                    run_root,
                    route=MODULE["PROVIDER_ROUTES"]["pinned"],
                    expected_sha256=source_state["source_sha256"],
                expected_mode=source_state["source_mode"],
                    runtime_environment_source=runtime_source,
                    expected_runtime_environment=runtime_environment,
                )

    def test_summary_carries_the_agent_error_the_harness_discards(self) -> None:
        """Run 20260802T113343Z stopped with agent_error and no stated cause.

        The shared harness maps the exception to a stop reason and drops it, so
        diagnosing a live failure needed a second paid approval. The treatment
        records its own error and the summary must carry it.
        """

        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            (run_root / "worker").mkdir()
            (run_root / "worker" / "process.jsonl").write_text(
                json.dumps(
                    {
                        "kind": "agent_error",
                        "action_point_id": 0,
                        "error": {
                            "type": "NooaWorkerError",
                            "message": "observation dtype differs from the approved ls20 contract",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            counts = MODULE["_artifact_counts"](run_root)

        self.assertEqual(counts["attempted_actions"], 0)
        self.assertEqual(len(counts["agent_errors"]), 1)
        self.assertIn("approved ls20 contract", counts["agent_errors"][0]["message"])

    def test_missing_game_is_refused_before_any_scorecard_is_opened(self) -> None:
        """Arcade reports a missing game only after opening a scorecard.

        Run 20260802T112449Z created scorecard ff8add52 and then failed the
        lookup, because the child executes inside the source snapshot where no
        game files exist. The check is a filesystem read and must precede the
        service call.
        """

        require = MODULE["_require_game_environment"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "game environment directory is absent"):
                require(root / "absent", "ls20")

            environments = root / "environment_files"
            (environments / "cd82").mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, r"'ls20' is absent.*cd82"):
                require(environments, "ls20")

            (environments / "ls20").mkdir()
            require(environments, "ls20")

    def test_forwarded_paths_survive_the_snapshot_working_directory_change(self) -> None:
        """A relative --run-root must not re-resolve inside the snapshot.

        The parent chdirs into the run-local snapshot before re-exec. Forwarding
        a relative path verbatim made the child resolve it against the snapshot
        directory and reject its own handoff as "outside the exact run root".
        """

        absolutize = MODULE["_absolutize_forwarded_paths"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            original = Path.cwd()
            os.chdir(root)
            try:
                argv = [
                    "--game",
                    "ls20",
                    "--run-root",
                    "artifacts/run",
                    "--approval-request=plans/approval-request.json",
                    "--confirm-live-run",
                    "RUN-0006-NOOA-SINGLE-ARM:abc",
                ]
                normalized = absolutize(argv)
                os.chdir(root / "..")
            finally:
                os.chdir(original)

        self.assertEqual(
            normalized,
            [
                "--game",
                "ls20",
                "--run-root",
                str(root / "artifacts/run"),
                f"--approval-request={root / 'plans/approval-request.json'}",
                "--confirm-live-run",
                "RUN-0006-NOOA-SINGLE-ARM:abc",
            ],
        )
        # Non-path tokens that happen to look like values are left alone.
        self.assertEqual(absolutize(["--game", "ls20"]), ["--game", "ls20"])
        self.assertEqual(absolutize(["--run-root"]), ["--run-root"])

    def test_snapshot_handoff_executes_exact_run_local_runner_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "run"
            snapshot_root = run_root / "source-snapshot"
            treatment_root = snapshot_root / "experiments/agents/0006-nooa"
            treatment_root.mkdir(parents=True)
            snapshot_root.chmod(0o755)
            for parent in treatment_root.parents:
                if parent == run_root:
                    break
                parent.chmod(0o755)
            treatment_root.chmod(0o755)
            runner = treatment_root / "run_live_probe.py"
            runner.write_text("# approved runner\n", encoding="utf-8")
            runner.chmod(0o644)
            relative = runner.relative_to(snapshot_root).as_posix()
            source_state = {
                "schema": "source-test",
                "source_sha256": {
                    relative: MODULE["sha256"](runner.read_bytes()).hexdigest()
                },
                "source_mode": {relative: "644"},
            }
            MODULE["_write_json"](
                snapshot_root / "record.json",
                {
                    "schema": "0006-nooa-live-source-snapshot-v2",
                    "root_mode": "755",
                    "directory_mode": "755",
                    "record_mode": "644",
                    "source_sha256": source_state["source_sha256"],
                    "source_mode": source_state["source_mode"],
                },
            )
            (snapshot_root / "record.json").chmod(0o644)
            args = parsed_args()
            args.run_root = run_root
            spec = MODULE["_paired_spec"](args)
            approval = MODULE["_approval_request"](
                spec=spec,
                run_root=run_root,
                source_state=source_state,
            route_name="pinned",
            )
            consumed_approval = root / "approval.json.consumed"
            consumed_approval.write_text(json.dumps(approval), encoding="utf-8")
            # Game files live in the real repository, never in the snapshot: the
            # parent writes the handoff while REPO still names the repository,
            # and the child runs with REPO pointing at the snapshot.
            (root / "environment_files" / "ls20").mkdir(parents=True)
            globals_patch = {
                "__file__": str(runner),
                "REPO": snapshot_root,
                "TREATMENT_ROOT": treatment_root,
            }
            with mock.patch.dict(MODULE["_load_snapshot_handoff"].__globals__, globals_patch):
                handoff, nonce = MODULE["_write_snapshot_handoff"](
                    run_root,
                    approval_request=approval,
                    consumed_approval=consumed_approval,
                    environments_dir=root / "environment_files",
                )
                with mock.patch("os.chdir") as chdir, mock.patch(
                    "os.execve", side_effect=RuntimeError("exec intercepted")
                ) as execve, self.assertRaisesRegex(RuntimeError, "exec intercepted"):
                    MODULE["_reexec_approved_snapshot"](
                        run_root=run_root,
                        handoff=handoff,
                        nonce=nonce,
                        argv=["--game", "ls20"],
                        expected_source_sha256=source_state["source_sha256"],
                        expected_source_mode=source_state["source_mode"],
                        expected_approval_request=approval,
                    )
                chdir.assert_called_once_with(snapshot_root)
                executable, command, environment = execve.call_args.args
                self.assertEqual(executable, sys.executable)
                self.assertEqual(command[:4], [sys.executable, "-I", "-B", str(runner)])
                self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")
                self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
                self.assertNotIn("PYTHONPATH", environment)
                self.assertEqual(environment[MODULE["SNAPSHOT_HANDOFF_ENV"]], nonce)
                args.approved_snapshot_handoff = handoff
                args.confirm_live_run = approval["required_confirmation"]
                with mock.patch.dict(
                    "os.environ",
                    {MODULE["SNAPSHOT_HANDOFF_ENV"]: nonce},
                    clear=False,
                ):
                    loaded_state, loaded_approval, loaded_environments = MODULE[
                        "_load_snapshot_handoff"
                    ](
                        args,
                        spec=spec,
                    )
                self.assertEqual(loaded_state, source_state)
                self.assertEqual(loaded_approval, consumed_approval)
                self.assertEqual(loaded_environments, root / "environment_files")
                self.assertFalse(handoff.exists())
                consumed_handoff = handoff.with_name(handoff.name + ".consumed")
                self.assertTrue(consumed_handoff.exists())
                consumed_handoff.replace(handoff)
                runner.chmod(0o755)
                with mock.patch("os.execve") as rejected_exec, self.assertRaisesRegex(
                    RuntimeError, "source snapshot"
                ):
                    MODULE["_reexec_approved_snapshot"](
                        run_root=run_root,
                        handoff=handoff,
                        nonce=nonce,
                        argv=["--game", "ls20"],
                        expected_source_sha256=source_state["source_sha256"],
                        expected_source_mode=source_state["source_mode"],
                        expected_approval_request=approval,
                    )
                rejected_exec.assert_not_called()
                runner.chmod(0o644)
                runner.write_text("# unapproved pre-import drift\n", encoding="utf-8")
                with mock.patch("os.execve") as rejected_exec, self.assertRaisesRegex(
                    RuntimeError, "source snapshot"
                ):
                    MODULE["_reexec_approved_snapshot"](
                        run_root=run_root,
                        handoff=handoff,
                        nonce=nonce,
                        argv=["--game", "ls20"],
                        expected_source_sha256=source_state["source_sha256"],
                        expected_source_mode=source_state["source_mode"],
                        expected_approval_request=approval,
                    )
                rejected_exec.assert_not_called()

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
        missing_identity = deepcopy(crossing)
        missing_identity["response"] = {"headers": {}, "body": {}}
        with self.assertRaisesRegex(RuntimeError, "0 generation identities"):
            MODULE["_generation_ids"]([missing_identity])
        conflicting_identity = deepcopy(crossing)
        conflicting_identity["response"]["body"]["id"] = "gen-other"
        with self.assertRaisesRegex(RuntimeError, "2 generation identities"):
            MODULE["_generation_ids"]([conflicting_identity])
        with self.assertRaisesRegex(RuntimeError, "reused across crossings"):
            MODULE["_generation_ids"]([crossing, deepcopy(crossing)])
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
        self.assertIsNone(identity["actual_quantization"])
        self.assertEqual(identity["requested_quantization"], "fp4")
        self.assertEqual(identity["quantization_evidence_scope"], "requested-route-only")
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
                expected_mode=source_state["source_mode"],
            )
            runtime_root = MODULE["_materialize_runtime_overlay"](
                run_root,
                route=MODULE["PROVIDER_ROUTES"]["pinned"],
                expected_sha256=source_state["source_sha256"],
                expected_mode=source_state["source_mode"],
                runtime_environment_source=Path(
                    source_state["runtime_environment_source"]
                ),
                expected_runtime_environment=source_state["runtime_environment"],
            )

            config = json.loads((runtime_root / "transport-config.json").read_text())
            self.assertEqual(config["provider"], MODULE["PROVIDER_ROUTE"])
            self.assertFalse(config["provider"]["require_parameters"])
            self.assertFalse((runtime_root / ".venv").is_symlink())
            self.assertEqual(
                MODULE["_runtime_environment_record"](runtime_root / ".venv"),
                source_state["runtime_environment"],
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
        events, process_records = two_action_fixture()
        verdict = MODULE["_operational_verdict"](
            summary,
            competence_events=events,
            process_records=process_records,
            provider_success=True,
            required_actions=2,
        )
        self.assertTrue(verdict["success"])

        without_claim_creation = [
            event
            for event in events
            if not (
                event.get("type") == "ledger" and event.get("action_id") == 1
            )
        ]
        self.assertFalse(
            MODULE["_operational_verdict"](
                summary,
                competence_events=without_claim_creation,
                process_records=process_records,
                provider_success=True,
                required_actions=2,
            )["success"]
        )

        noncausal_process = sorted(
            deepcopy(process_records),
            key=lambda record: record.get("kind") != "competence_action_admitted",
        )
        self.assertFalse(
            MODULE["_operational_verdict"](
                summary,
                competence_events=events,
                process_records=noncausal_process,
                provider_success=True,
                required_actions=2,
            )["success"]
        )

        noncausal_events = deepcopy(events)
        initial_position = next(
            index
            for index, event in enumerate(noncausal_events)
            if event.get("type") == "observation"
            and event.get("record", {}).get("step") == 0
        )
        initial_event = noncausal_events.pop(initial_position)
        first_commitment_position = next(
            index
            for index, event in enumerate(noncausal_events)
            if event.get("type") == "commitment"
        )
        noncausal_events.insert(first_commitment_position + 1, initial_event)
        self.assertFalse(
            MODULE["_operational_verdict"](
                summary,
                competence_events=noncausal_events,
                process_records=process_records,
                provider_success=True,
                required_actions=2,
            )["success"]
        )

        wrong_nested_schema_events = deepcopy(events)
        wrong_nested_schema_process = deepcopy(process_records)
        first_request = next(
            record
            for record in wrong_nested_schema_process
            if record.get("kind") == "action_request"
        )
        first_packet = first_request["request"]["observation"]["competence"][
            "context"
        ]
        first_packet["schema"] = "forged-schema"
        first_request["request_sha256"] = canonical_digest(first_request["request"])
        first_context = next(
            record
            for record in wrong_nested_schema_process
            if record.get("kind") == "competence_context_assembled"
        )
        first_context["competence_packet_sha256"] = canonical_digest(first_packet)
        first_context_event = next(
            event
            for event in wrong_nested_schema_events
            if event.get("type") == "context_assembled"
        )
        first_context_event["packet_sha256"] = competence_packet_digest(first_packet)
        self.assertFalse(
            MODULE["_operational_verdict"](
                summary,
                competence_events=wrong_nested_schema_events,
                process_records=wrong_nested_schema_process,
                provider_success=True,
                required_actions=2,
            )["success"]
        )

        detached_events = deepcopy(events)
        detached_process = deepcopy(process_records)
        detached_id = "obs-detached"
        first_commitment = next(
            event for event in detached_events if event.get("type") == "commitment"
        )
        first_commitment["before_evidence_id"] = detached_id
        first_transition = next(
            event
            for event in detached_events
            if event.get("type") == "transition_occurrence"
        )
        first_transition["record"]["before_evidence_id"] = detached_id
        first_prediction = next(
            event
            for event in detached_events
            if event.get("type") == "prediction_result"
        )
        first_prediction["before_evidence_id"] = detached_id
        detached_request = next(
            record
            for record in detached_process
            if record.get("kind") == "action_request"
        )
        detached_packet = detached_request["request"]["observation"]["competence"][
            "context"
        ]
        detached_packet["current_observation"]["evidence_id"] = detached_id
        detached_request["request_sha256"] = canonical_digest(
            detached_request["request"]
        )
        detached_context = next(
            record
            for record in detached_process
            if record.get("kind") == "competence_context_assembled"
        )
        detached_context["current_evidence_id"] = detached_id
        detached_context["competence_packet_sha256"] = canonical_digest(
            detached_packet
        )
        detached_context_event = next(
            event
            for event in detached_events
            if event.get("type") == "context_assembled"
        )
        detached_context_event["packet_sha256"] = competence_packet_digest(
            detached_packet
        )
        detached_admitted = next(
            record
            for record in detached_process
            if record.get("kind") == "competence_action_admitted"
        )
        detached_admitted["commitment"] = {
            key: value for key, value in first_commitment.items() if key != "type"
        }
        self.assertFalse(
            MODULE["_operational_verdict"](
                summary,
                competence_events=detached_events,
                process_records=detached_process,
                provider_success=True,
                required_actions=2,
            )["success"]
        )

        retire_events, retire_process = two_action_fixture(retire_claim=True)
        self.assertTrue(
            MODULE["_operational_verdict"](
                summary,
                competence_events=retire_events,
                process_records=retire_process,
                provider_success=True,
                required_actions=2,
            )["success"]
        )

        no_transition = [event for event in events if event.get("type") != "transition_occurrence"]
        self.assertFalse(
            MODULE["_operational_verdict"](
                summary,
                competence_events=no_transition,
                process_records=process_records,
                provider_success=True,
                required_actions=2,
            )["success"]
        )

        wrong_second_transition = deepcopy(events)
        transitions = [
            event
            for event in wrong_second_transition
            if event.get("type") == "transition_occurrence"
        ]
        transitions[1]["record"]["action_id"] = 999
        self.assertFalse(
            MODULE["_operational_verdict"](
                summary,
                competence_events=wrong_second_transition,
                process_records=process_records,
                provider_success=True,
                required_actions=2,
            )["success"]
        )

        wrong_schema = deepcopy(events)
        wrong_schema[0].pop("schema")
        self.assertFalse(
            MODULE["_operational_verdict"](
                summary,
                competence_events=wrong_schema,
                process_records=process_records,
                provider_success=True,
                required_actions=2,
            )["success"]
        )

        wrong_prediction = deepcopy(events)
        prediction = next(
            event for event in wrong_prediction if event.get("type") == "prediction_result"
        )
        prediction["action_id"] = 999
        self.assertFalse(
            MODULE["_operational_verdict"](
                summary,
                competence_events=wrong_prediction,
                process_records=process_records,
                provider_success=True,
                required_actions=2,
            )["success"]
        )

        wrong_evaluation_events = deepcopy(events)
        wrong_evaluation_process = deepcopy(process_records)
        wrong_evaluation = next(
            event
            for event in wrong_evaluation_events
            if event.get("type") == "prediction_result"
        )
        wrong_evaluation["checks"][0]["observed"] = 999
        second_evaluation_request = [
            record
            for record in wrong_evaluation_process
            if record.get("kind") == "action_request"
        ][1]
        evaluation_packet = second_evaluation_request["request"]["observation"][
            "competence"
        ]["context"]
        evaluation_packet["last_prediction_result"] = {
            key: value for key, value in wrong_evaluation.items() if key != "type"
        }
        second_evaluation_request["request_sha256"] = canonical_digest(
            second_evaluation_request["request"]
        )
        second_evaluation_context = [
            record
            for record in wrong_evaluation_process
            if record.get("kind") == "competence_context_assembled"
        ][1]
        second_evaluation_context["competence_packet_sha256"] = canonical_digest(
            evaluation_packet
        )
        second_evaluation_event = [
            event
            for event in wrong_evaluation_events
            if event.get("type") == "context_assembled"
        ][1]
        second_evaluation_event["packet_sha256"] = competence_packet_digest(
            evaluation_packet
        )
        self.assertFalse(
            MODULE["_operational_verdict"](
                summary,
                competence_events=wrong_evaluation_events,
                process_records=wrong_evaluation_process,
                provider_success=True,
                required_actions=2,
            )["success"]
        )

        wrong_packet_events = deepcopy(events)
        wrong_packet_process = deepcopy(process_records)
        second_request = [
            record for record in wrong_packet_process if record.get("kind") == "action_request"
        ][1]
        packet = second_request["request"]["observation"]["competence"]["context"]
        packet["last_prediction_result"]["action_id"] = 999
        second_context = [
            record
            for record in wrong_packet_process
            if record.get("kind") == "competence_context_assembled"
        ][1]
        second_context["competence_packet_sha256"] = canonical_digest(packet)
        second_context_event = [
            event for event in wrong_packet_events if event.get("type") == "context_assembled"
        ][1]
        second_context_event["packet_sha256"] = competence_packet_digest(packet)
        self.assertFalse(
            MODULE["_operational_verdict"](
                summary,
                competence_events=wrong_packet_events,
                process_records=wrong_packet_process,
                provider_success=True,
                required_actions=2,
            )["success"]
        )

        no_continuity = deepcopy(events)
        second_commitment = [
            event for event in no_continuity if event.get("type") == "commitment"
        ][1]
        second_commitment["claim_ids"] = ["claim-other"]
        no_continuity_process = deepcopy(process_records)
        admitted = [
            record
            for record in no_continuity_process
            if record.get("kind") == "competence_action_admitted"
        ]
        admitted[1]["commitment"] = second_commitment
        self.assertFalse(
            MODULE["_operational_verdict"](
                summary,
                competence_events=no_continuity,
                process_records=no_continuity_process,
                provider_success=True,
                required_actions=2,
            )["success"]
        )

        overrun = {**summary, "actions": 3, "attempted_actions": 3, "observed_successors": 3}
        self.assertFalse(
            MODULE["_operational_verdict"](
                overrun,
                competence_events=events,
                process_records=process_records,
                provider_success=True,
                required_actions=2,
            )["success"]
        )

        mismatch_events, mismatch_process = two_action_fixture(mismatch=True)
        self.assertTrue(
            MODULE["_operational_verdict"](
                summary,
                competence_events=mismatch_events,
                process_records=mismatch_process,
                provider_success=True,
                required_actions=2,
            )["success"]
        )
        forged_mismatch_events = deepcopy(mismatch_events)
        forged_mismatch_process = deepcopy(mismatch_process)
        forged_id = "mismatch-forged-unrelated"
        forged_request = [
            record
            for record in forged_mismatch_process
            if record.get("kind") == "action_request"
        ][1]
        forged_packet = forged_request["request"]["observation"]["competence"][
            "context"
        ]
        forged_packet["unresolved_mismatches"][0]["mismatch_id"] = forged_id
        forged_request["request_sha256"] = canonical_digest(forged_request["request"])
        forged_context = [
            record
            for record in forged_mismatch_process
            if record.get("kind") == "competence_context_assembled"
        ][1]
        forged_context["competence_packet_sha256"] = canonical_digest(forged_packet)
        forged_context_event = [
            event
            for event in forged_mismatch_events
            if event.get("type") == "context_assembled"
        ][1]
        forged_context_event["packet_sha256"] = competence_packet_digest(forged_packet)
        forged_reconciliation = next(
            event
            for event in forged_mismatch_events
            if event.get("type") == "reconciliation"
        )
        forged_reconciliation["items"][0]["mismatch_id"] = forged_id
        forged_second_commitment = [
            event
            for event in forged_mismatch_events
            if event.get("type") == "commitment"
        ][1]
        forged_second_commitment["reconciliation"] = forged_reconciliation["items"]
        forged_admitted = [
            record
            for record in forged_mismatch_process
            if record.get("kind") == "competence_action_admitted"
        ][1]
        forged_admitted["commitment"] = {
            key: value
            for key, value in forged_second_commitment.items()
            if key != "type"
        }
        self.assertFalse(
            MODULE["_operational_verdict"](
                summary,
                competence_events=forged_mismatch_events,
                process_records=forged_mismatch_process,
                provider_success=True,
                required_actions=2,
            )["success"]
        )
        wrong_reconciliation = deepcopy(mismatch_events)
        reconciliation = next(
            event for event in wrong_reconciliation if event.get("type") == "reconciliation"
        )
        reconciliation["items"][0]["mismatch_id"] = "mismatch-wrong"
        second_commitment = [
            event for event in wrong_reconciliation if event.get("type") == "commitment"
        ][1]
        second_commitment["reconciliation"] = reconciliation["items"]
        wrong_reconciliation_process = deepcopy(mismatch_process)
        admitted = [
            record
            for record in wrong_reconciliation_process
            if record.get("kind") == "competence_action_admitted"
        ]
        admitted[1]["commitment"] = second_commitment
        self.assertFalse(
            MODULE["_operational_verdict"](
                summary,
                competence_events=wrong_reconciliation,
                process_records=wrong_reconciliation_process,
                provider_success=True,
                required_actions=2,
            )["success"]
        )

    def test_a_refused_crossing_stays_visible_and_must_precede_its_action(self) -> None:
        """A repair is evidence, not noise, and only counts before the action.

        Run 20260802T120006Z lost a correct commitment to one surplus brace with
        no way to answer. Once the host can re-cross, the log must still show
        that a refusal happened and that it came before the action it belongs to.
        """

        crossing_kinds = {
            "action_request",
            "model_crossing",
            "decision_validated",
            "competence_action_admitted",
        }

        def with_refusal(after_admission: bool) -> list[dict]:
            _events, process = two_action_fixture()
            # Leave crossing id 0 free for the refused attempt.
            for record in process:
                if record.get("kind") in crossing_kinds:
                    record["action_point_id"] = record["step"] + 1
                    if record["kind"] == "action_request":
                        record["request"]["action_point_id"] = record["action_point_id"]
                        record["request_sha256"] = canonical_digest(record["request"])
            request = deepcopy(
                next(r for r in process if r.get("kind") == "action_request")
            )
            request["action_point_id"] = 0
            request["request"]["action_point_id"] = 0
            request["request_sha256"] = canonical_digest(request["request"])
            crossing = deepcopy(
                next(r for r in process if r.get("kind") == "model_crossing")
            )
            crossing["action_point_id"] = 0
            refusal = {
                "protocol": request["protocol"],
                "kind": "competence_rejected",
                "run_id": request["run_id"],
                "action_point_id": 0,
                "step": 0,
                "error": {
                    "type": "ModelCorrectableError",
                    "message": "competence commitment rejected: not a JSON object",
                    "correctable": True,
                },
                "attempts_remaining": 1,
            }
            triple = [request, crossing, refusal]
            anchor_kind = (
                "competence_action_admitted" if after_admission else "action_request"
            )
            index = process.index(
                next(r for r in process if r.get("kind") == anchor_kind)
            )
            offset = index + 1 if after_admission else index
            return process[:offset] + triple + process[offset:]

        events, _baseline = two_action_fixture()

        report = two_action_chain_report(events, with_refusal(after_admission=False))
        self.assertTrue(all(report.values()), report)

        # A refusal recorded after the action was released is not a repair.
        late = two_action_chain_report(events, with_refusal(after_admission=True))
        self.assertFalse(late["refused_crossings_precede_their_action"])

    def test_reviewed_false_positive_mutants_fail_closed(self) -> None:
        summary = {
            "actions": 2,
            "attempted_actions": 2,
            "observed_successors": 2,
            "stop_reason": "action_budget",
            "clean_shutdown": True,
            "trace_complete": True,
        }

        def accepted(events: list[dict], process_records: list[dict]) -> bool:
            return MODULE["_operational_verdict"](
                summary,
                competence_events=events,
                process_records=process_records,
                provider_success=True,
                required_actions=2,
            )["success"]

        baseline_events, baseline_process = two_action_fixture()
        self.assertTrue(accepted(baseline_events, baseline_process))

        forged_step_events = deepcopy(baseline_events)
        forged_step_process = deepcopy(baseline_process)
        first_commitment = next(
            event
            for event in forged_step_events
            if event.get("type") == "commitment"
        )
        first_commitment["before_step"] = 999
        first_commitment_record = {
            key: value for key, value in first_commitment.items() if key != "type"
        }
        first_admission = next(
            record
            for record in forged_step_process
            if record.get("kind") == "competence_action_admitted"
        )
        first_admission["commitment"] = first_commitment_record
        first_validation = next(
            record
            for record in forged_step_process
            if record.get("kind") == "decision_validated"
        )
        first_validation["decision"]["reasoning"]["competence"] = (
            first_commitment_record
        )
        self.assertFalse(accepted(forged_step_events, forged_step_process))

        forged_wrapper_process = deepcopy(baseline_process)
        first_request_record = next(
            record
            for record in forged_wrapper_process
            if record.get("kind") == "action_request"
        )
        first_request_record["request"]["observation"]["competence"]["schema"] = (
            "forged-wrapper-schema"
        )
        first_request_record["request_sha256"] = canonical_digest(
            first_request_record["request"]
        )
        self.assertFalse(accepted(baseline_events, forged_wrapper_process))

        illegal_action_process = deepcopy(baseline_process)
        first_request_record = next(
            record
            for record in illegal_action_process
            if record.get("kind") == "action_request"
        )
        first_request_record["request"]["legal_actions"] = [2]
        first_request_record["request_sha256"] = canonical_digest(
            first_request_record["request"]
        )
        self.assertFalse(accepted(baseline_events, illegal_action_process))

        wrong_environment_transition = deepcopy(baseline_process)
        first_transition_record = next(
            record
            for record in wrong_environment_transition
            if record.get("kind") == "transition_recorded"
        )
        first_transition_record["transition"]["action"] = 999
        self.assertFalse(accepted(baseline_events, wrong_environment_transition))

        unauthored_ledger_process = deepcopy(baseline_process)
        first_crossing = next(
            record
            for record in unauthored_ledger_process
            if record.get("kind") == "model_crossing"
        )
        first_payload = json.loads(first_crossing["worker_decision"]["reason"])
        first_payload["ledger"] = []
        first_payload["claim_ids"] = []
        forged_reason = json.dumps(first_payload, separators=(",", ":"))
        first_crossing["worker_decision"]["reason"] = forged_reason
        next(
            record
            for record in unauthored_ledger_process
            if record.get("kind") == "decision_validated"
        )["decision"]["reasoning"]["reason"] = forged_reason
        self.assertFalse(accepted(baseline_events, unauthored_ledger_process))

        forged_request_identity = deepcopy(baseline_process)
        first_request_record = next(
            record
            for record in forged_request_identity
            if record.get("kind") == "action_request"
        )
        first_request_record["request"]["run_id"] = "forged-run"
        first_request_record["request_sha256"] = canonical_digest(
            first_request_record["request"]
        )
        self.assertFalse(accepted(baseline_events, forged_request_identity))

        for nested_path, field in (
            (("observation", "current_metadata"), "forged_metadata"),
            (("observation", "current_metadata", "frame"), "forged_frame"),
            (("observation", "representations", "grid"), "forged_representation"),
            (("paired_run_spec",), "forged_spec"),
        ):
            unsupported_nested_request = deepcopy(baseline_process)
            request_record = next(
                record
                for record in unsupported_nested_request
                if record.get("kind") == "action_request"
            )
            target = request_record["request"]
            for key in nested_path:
                target = target[key]
            target[field] = True
            request_record["request_sha256"] = canonical_digest(
                request_record["request"]
            )
            self.assertFalse(accepted(baseline_events, unsupported_nested_request))

        malformed_audit = deepcopy(baseline_process)
        next(
            record
            for record in malformed_audit
            if record.get("kind") == "model_crossing"
        )["audit"]["model_calls"] = [{}]
        self.assertFalse(accepted(baseline_events, malformed_audit))

        mismatched_audit_messages = deepcopy(baseline_process)
        audit_call = next(
            record
            for record in mismatched_audit_messages
            if record.get("kind") == "model_crossing"
        )["audit"]["model_calls"][0]
        audit_call["messages"] = [{"role": "user", "content": "forged"}]
        self.assertFalse(accepted(baseline_events, mismatched_audit_messages))

        forged_audit_outcome = deepcopy(baseline_process)
        audit_call = next(
            record
            for record in forged_audit_outcome
            if record.get("kind") == "model_crossing"
        )["audit"]["model_calls"][-1]
        audit_call["outcome"]["response"]["tool_calls"][-1]["arguments"] = (
            json.dumps({"result": {"action": 999, "data": None, "reason": "forged"}})
        )
        self.assertFalse(accepted(baseline_events, forged_audit_outcome))

        forged_projection_dtype = deepcopy(baseline_process)
        context_projection = next(
            record
            for record in forged_projection_dtype
            if record.get("kind") == "competence_context_assembled"
        )["projection"]
        context_projection["dtype"] = "<i4"
        self.assertFalse(accepted(baseline_events, forged_projection_dtype))

        unsupported_event_field = deepcopy(baseline_events)
        next(
            event
            for event in unsupported_event_field
            if event.get("type") == "commitment"
        )["forged"] = True
        self.assertFalse(accepted(unsupported_event_field, baseline_process))

        missing_public_field = deepcopy(baseline_events)
        [
            event
            for event in missing_public_field
            if event.get("type") == "observation"
        ][1]["record"]["public"].pop("full_reset")
        self.assertFalse(accepted(missing_public_field, baseline_process))

        mismatch_events, mismatch_process = two_action_fixture(mismatch=True)
        unauthored_reconciliation = deepcopy(mismatch_process)
        second_crossing = [
            record
            for record in unauthored_reconciliation
            if record.get("kind") == "model_crossing"
        ][1]
        second_payload = json.loads(second_crossing["worker_decision"]["reason"])
        second_payload["reconciliation"] = None
        forged_reason = json.dumps(second_payload, separators=(",", ":"))
        second_crossing["worker_decision"]["reason"] = forged_reason
        [
            record
            for record in unauthored_reconciliation
            if record.get("kind") == "decision_validated"
        ][1]["decision"]["reasoning"]["reason"] = forged_reason
        self.assertFalse(accepted(mismatch_events, unauthored_reconciliation))

        invalid_disposition_events = deepcopy(mismatch_events)
        invalid_disposition_process = deepcopy(mismatch_process)
        reconciliation = next(
            event
            for event in invalid_disposition_events
            if event.get("type") == "reconciliation"
        )
        reconciliation["items"][0]["disposition"] = "FORGED"
        second_commitment = [
            event
            for event in invalid_disposition_events
            if event.get("type") == "commitment"
        ][1]
        second_commitment["reconciliation"] = reconciliation["items"]
        second_commitment_record = {
            key: value for key, value in second_commitment.items() if key != "type"
        }
        second_admission = [
            record
            for record in invalid_disposition_process
            if record.get("kind") == "competence_action_admitted"
        ][1]
        second_admission["commitment"] = second_commitment_record
        second_validation = [
            record
            for record in invalid_disposition_process
            if record.get("kind") == "decision_validated"
        ][1]
        second_validation["decision"]["reasoning"]["competence"] = (
            second_commitment_record
        )
        self.assertFalse(accepted(invalid_disposition_events, invalid_disposition_process))

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
            source_state = MODULE["_source_state"]()
            MODULE["_snapshot_participating_sources"](
                run_root,
                expected_sha256=source_state["source_sha256"],
                expected_mode=source_state["source_mode"],
            )
            environments_dir = Path(directory) / "environment_files"
            (environments_dir / args.game).mkdir(parents=True)
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
                    expected_source_sha256=source_state["source_sha256"],
                        expected_source_mode=source_state["source_mode"],
                    runtime_environment_source=Path(
                        source_state["runtime_environment_source"]
                    ),
                    expected_runtime_environment=source_state["runtime_environment"],
                    environments_dir=environments_dir,
                    route=MODULE["PROVIDER_ROUTES"]["pinned"],
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


class ProviderRouteTests(unittest.TestCase):
    def test_an_open_route_cannot_qualify_a_gate(self) -> None:
        """Automatic routing shows the loop closes, not that it closes everywhere.

        0002's G9 gate passed a broken substrate because unpinned load balancing
        put its later crossings on CoreWeave, which accepts an unanswered tool
        call that ModelRun refuses. An open run must never be able to stand in
        for the pinned evidence run again.
        """

        summary = {
            "actions": 2,
            "attempted_actions": 2,
            "observed_successors": 2,
            "stop_reason": "action_budget",
            "clean_shutdown": True,
            "trace_complete": True,
        }
        events, process = two_action_fixture()

        def verdict(route: str) -> dict:
            return MODULE["_operational_verdict"](
                summary,
                competence_events=events,
                process_records=process,
                provider_success=True,
                required_actions=2,
                route_name=route,
            )

        pinned = verdict("pinned")
        self.assertTrue(pinned["success"], pinned)
        self.assertTrue(pinned["checks"]["route_can_qualify"])
        self.assertEqual(pinned["route"], "pinned")

        # Identical evidence, only the route differs.
        opened = verdict("open")
        self.assertFalse(opened["success"])
        self.assertFalse(opened["checks"]["route_can_qualify"])
        self.assertEqual(opened["route"], "open")
        differing = {
            key
            for key in pinned["checks"]
            if pinned["checks"][key] != opened["checks"][key]
        }
        self.assertEqual(differing, {"route_can_qualify"})

    def test_both_routes_are_accepted_by_the_audited_launcher(self) -> None:
        overlay = runpy.run_path(str(ROOT / "provider_runtime_launcher.py"))
        for route in MODULE["PROVIDER_ROUTES"].values():
            self.assertIn(route, overlay["EXPECTED_PROVIDER_ROUTES"])
        # An open route carries no provider block, so nothing is pinned onto the
        # request; a pinned one must still reach extra_body intact.
        class _Client:
            def __init__(self) -> None:
                self.config: dict = {}

        opened = overlay["_pin_client"](_Client(), MODULE["PROVIDER_ROUTES"]["open"])
        self.assertEqual(opened.config, {})
        pinned = overlay["_pin_client"](_Client(), MODULE["PROVIDER_ROUTES"]["pinned"])
        self.assertEqual(
            pinned.config["extra_body"],
            {"provider": MODULE["PROVIDER_ROUTES"]["pinned"]},
        )


class InlineReturnRouteTests(unittest.TestCase):
    """The verifier must accept the route the packet requires.

    Run 20260802T182855Z closed the two-action loop and still failed
    linked_process_records, because the verifier demanded a separate
    return_result tool call while the packet forbids one. The audit shape here is
    the one measured from that run: {"arguments", "id", "name"} with arguments a
    JSON object carrying "code".
    """

    @staticmethod
    def _cell(code: str) -> dict:
        return {
            "id": "call_abc123",
            "name": "execute_python",
            "arguments": json.dumps({"code": code}),
        }

    def test_inline_return_as_the_final_statement_closes_the_turn(self) -> None:
        inline = MODULE_CHAIN["_inline_return_closes_the_turn"]
        self.assertTrue(
            inline(
                self._cell(
                    "import json\n"
                    "action = 1\n"
                    "data = None\n"
                    "reason = json.dumps({'test': 'x'})\n"
                    "return_result({'action': action, 'data': data, 'reason': reason})"
                )
            )
        )

    def test_a_cell_that_does_not_end_in_return_result_does_not_close_it(self) -> None:
        inline = MODULE_CHAIN["_inline_return_closes_the_turn"]
        # Computing the payload is not returning it. Run 20260802T124434Z spent
        # every round doing exactly this and released no action.
        self.assertFalse(inline(self._cell("reason = '{}'\nprint(reason)")))
        self.assertFalse(inline(self._cell("return_result({'a': 1})\nprint('after')")))
        self.assertFalse(inline(self._cell("not valid python (")))
        self.assertFalse(inline({"id": "x", "name": "execute_python"}))

    def test_the_separate_tool_route_still_binds_the_literal_decision(self) -> None:
        binds = MODULE_CHAIN["_last_model_call_returns_decision"]
        decision = {"action": 1, "data": None, "reason": "r"}
        call = {
            "outcome": {
                "kind": "success",
                "response": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "return_result",
                            "arguments": json.dumps({"result": decision}),
                        }
                    ]
                },
            }
        }
        self.assertTrue(binds([call], decision))
        self.assertFalse(binds([call], {**decision, "action": 2}))
