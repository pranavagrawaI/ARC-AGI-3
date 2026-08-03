from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stdout
from hashlib import sha256
from http.client import IncompleteRead
from io import StringIO
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError


CONTROL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTROL_ROOT))

from host.qualification import ProviderRequestProfile  # noqa: E402
from host.control_surface import normalize_quoted_null_message  # noqa: E402
from llm_repl_agent.agent import parse_decision_message  # noqa: E402
from llm_repl_agent.client import ChatResponse  # noqa: E402
from run_paired_probe import (  # noqa: E402
    PROVIDER_ROUTES,
    _control_client_for_live,
    _materialize_runtime_overlay,
    _provider_profile_report,
    main,
)


class ProviderRouteTests(unittest.TestCase):
    """The pin has to reach both arms, which are reached by different means.

    The control client is host-owned and in-process, so it takes the route
    directly. The candidate runs in a separate process against the frozen
    runtime, so its route arrives through an overlay launcher. A pin that
    reached only one arm would leave the gate exactly as uninformative as the
    unpinned run it replaces, while looking pinned.
    """

    def _profile(self) -> ProviderRequestProfile:
        return ProviderRequestProfile.responses(
            model="openai/google/gemma-4-31b-it",
            wire_model="google/gemma-4-31b-it",
            api_base="https://openrouter.ai/api/v1",
            max_output_tokens=1_024,
            reasoning_effort="medium",
            request_timeout_seconds=30.0,
        )

    def test_the_control_client_carries_the_route_it_was_given(self) -> None:
        for name, route in PROVIDER_ROUTES.items():
            client = _control_client_for_live(
                self._profile(), "test-secret", provider_route=route
            )
            payload = client._payload([{"role": "user", "content": "hi"}], [])
            with self.subTest(route=name):
                if route:
                    self.assertEqual(payload["provider"], route)
                else:
                    self.assertNotIn("provider", payload)

    def test_an_absent_route_leaves_the_request_exactly_as_it_was(self) -> None:
        unrouted = _control_client_for_live(self._profile(), "test-secret")
        self.assertNotIn(
            "provider", unrouted._payload([{"role": "user", "content": "hi"}], [])
        )

    def test_the_overlay_points_at_the_frozen_runtime_without_copying_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "pair"
            run_root.mkdir()
            overlay = _materialize_runtime_overlay(
                run_root, PROVIDER_ROUTES["pinned"]
            )

            config = json.loads((overlay / "transport-config.json").read_text())
            self.assertEqual(
                config["schema"], "0002-nooa-provider-runtime-overlay-v1"
            )
            self.assertEqual(config["provider"], PROVIDER_ROUTES["pinned"])
            base = Path(config["base_runtime_root"])
            self.assertTrue((base / "launcher.py").is_file())
            self.assertEqual(base, (CONTROL_ROOT / "nooa-runtime").resolve())

            # The worker is started as <runtime>/.venv/bin/python <runtime>/launcher.py,
            # so both have to resolve inside the overlay.
            self.assertTrue((overlay / "launcher.py").is_file())
            self.assertTrue((overlay / ".venv").exists())
            self.assertEqual(
                (overlay / "launcher.py").read_bytes(),
                (CONTROL_ROOT / "provider_runtime_launcher.py").read_bytes(),
            )
            # The frozen runtime is pointed at, never rewritten.
            self.assertNotEqual(
                (overlay / "launcher.py").read_bytes(),
                (base / "launcher.py").read_bytes(),
            )

    def test_the_overlay_launcher_refuses_a_route_it_does_not_know(self) -> None:
        sys.path.insert(0, str(CONTROL_ROOT))
        from provider_runtime_launcher import _load_overlay_config

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transport-config.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "0002-nooa-provider-runtime-overlay-v1",
                        "base_runtime_root": str(CONTROL_ROOT / "nooa-runtime"),
                        "provider": {"order": ["somewhere-else"]},
                    }
                )
            )
            with self.assertRaisesRegex(RuntimeError, "unknown provider route"):
                _load_overlay_config(path)


class PairedProbeRunnerTests(unittest.TestCase):
    def test_control_canonicalizes_exact_quoted_null_before_decision_parsing(self) -> None:
        response = ChatResponse(
            message={
                "role": "assistant",
                "content": '{"action":1,"data":"null","reason":"move"}',
            },
            usage={},
            response_body={"preserves": "original transport evidence"},
        )

        normalized = normalize_quoted_null_message(response)
        action, data, reason = parse_decision_message(
            normalized.message,
            available_actions=(1,),
            frame_width=2,
            frame_height=2,
        )

        self.assertEqual((action, data, reason), (1, None, "move"))
        self.assertEqual(normalized.response_body, response.response_body)

    def test_g9_rejects_a_one_action_budget(self) -> None:
        with self.assertRaisesRegex(SystemExit, "at least 2"):
            main(["--offline", "--max-actions", "1"])

    def test_actual_provider_crossings_must_match_the_frozen_wire_profile(
        self,
    ) -> None:
        profile = ProviderRequestProfile.responses(
            model="openai/provider/model",
            wire_model="provider/model",
            api_base="https://gateway.invalid/v1",
            max_output_tokens=4096,
            reasoning_effort="medium",
            request_timeout_seconds=90.0,
        )
        crossing = {
            "request": {
                "url": "https://gateway.invalid/v1/responses",
                "body": {
                    "model": "provider/model",
                    "max_output_tokens": 4096,
                    "reasoning": {"effort": "medium"},
                    "stream": False,
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [{"type": "function", "name": "python"}],
                    "truncation": "disabled",
                },
            }
        }

        matching = _provider_profile_report(
            profile,
            control_crossings=[crossing],
            candidate_crossings=[crossing],
        )
        drifted_crossing = deepcopy(crossing)
        drifted_crossing["request"]["body"]["model"] = "another-model"
        drifted = _provider_profile_report(
            profile,
            control_crossings=[crossing],
            candidate_crossings=[drifted_crossing],
        )

        self.assertTrue(matching["success"])
        self.assertEqual(
            set(matching["projections"]),
            {"0002-R-minimal", "0002-NOOA"},
        )
        self.assertFalse(drifted["success"])
        self.assertEqual(drifted["mismatches"][0]["arm"], "0002-NOOA")

    def test_control_responses_transport_records_a_redacted_actual_crossing(
        self,
    ) -> None:
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return json.dumps(
                    {
                        "id": "response-1",
                        "echo": "credential-not-recorded",
                        "signed_url": (
                            "https://user:password@gateway.invalid/object"
                            "?token=other-secret&sig=signature-value&public=yes"
                        ),
                        "output": [
                            {
                                "type": "function_call",
                                "call_id": "call-1",
                                "name": "python",
                                "arguments": '{"code":"print(1)"}',
                            }
                        ],
                        "usage": {"total_tokens": 3},
                    }
                ).encode()

        profile = ProviderRequestProfile.responses(
            model="openai/provider/model",
            wire_model="provider/model",
            api_base="https://gateway.invalid/v1",
            max_output_tokens=4096,
            reasoning_effort="medium",
            request_timeout_seconds=90.0,
        )
        client = _control_client_for_live(profile, "credential-not-recorded")
        with patch("host.responses_client.urlopen", return_value=Response()):
            response = client.complete(
                messages=[
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "observe"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
                            },
                        ],
                    },
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "python",
                            "description": "execute",
                            "parameters": {
                                "type": "object",
                                "properties": {"code": {"type": "string"}},
                                "required": ["code"],
                            },
                        },
                    }
                ],
            )

        self.assertEqual(response.message["tool_calls"][0]["function"]["name"], "python")
        crossing_text = json.dumps(client.actual_crossings)
        self.assertNotIn("data:image/", crossing_text)
        self.assertNotIn("credential-not-recorded", crossing_text)
        self.assertNotIn("other-secret", crossing_text)
        self.assertNotIn("signature-value", crossing_text)
        self.assertNotIn("user:password", crossing_text)
        self.assertNotIn(
            "credential-not-recorded",
            json.dumps(response.response_metadata),
        )
        image = client.actual_crossings[0]["request"]["body"]["input"][0]["content"][1]
        self.assertTrue(image["image_url"]["redacted"])
        self.assertEqual(
            client.actual_crossings[0]["request"]["body"]["max_output_tokens"],
            4096,
        )
        self.assertEqual(response.response_body["usage"]["total_tokens"], 3)
        self.assertEqual(len(client.actual_crossings[0]["request"]["body_sha256"]), 64)

    def test_provider_profile_omits_tool_controls_when_a_crossing_has_no_tools(
        self,
    ) -> None:
        profile = ProviderRequestProfile.responses(
            model="openai/provider/model",
            wire_model="provider/model",
            api_base="https://gateway.invalid/v1",
            max_output_tokens=4096,
            reasoning_effort="medium",
            request_timeout_seconds=90.0,
        )
        crossing = {
            "request": {
                "url": "https://gateway.invalid/v1/responses",
                "body": {
                    "model": "provider/model",
                    "input": [],
                    "max_output_tokens": 4096,
                    "reasoning": {"effort": "medium"},
                    "stream": False,
                    "truncation": "disabled",
                },
            }
        }

        report = _provider_profile_report(
            profile,
            control_crossings=[crossing],
            candidate_crossings=[crossing],
        )

        self.assertTrue(report["success"])

    def test_control_responses_retries_only_proven_pre_connect_failures(self) -> None:
        profile = ProviderRequestProfile.responses(
            model="openai/provider/model",
            wire_model="provider/model",
            api_base="https://gateway.invalid/v1",
            max_output_tokens=4096,
            reasoning_effort="medium",
            request_timeout_seconds=90.0,
        )
        client = _control_client_for_live(profile, "credential-not-recorded")
        request = {"messages": [{"role": "user", "content": "x"}], "tools": []}

        with patch("host.responses_client.urlopen", side_effect=TimeoutError("late")):
            with self.assertRaises(Exception) as timeout_error:
                client.complete(**request)
        http_failure = HTTPError(
            "https://gateway.invalid/v1/responses",
            503,
            "unavailable",
            {},
            BytesIO(b'{"error":"received"}'),
        )
        with patch("host.responses_client.urlopen", side_effect=http_failure):
            with self.assertRaises(Exception) as status_error:
                client.complete(**request)
        with patch(
            "host.responses_client.urlopen",
            side_effect=IncompleteRead(b"partial", 10),
        ):
            with self.assertRaises(Exception) as partial_error:
                client.complete(**request)
        with patch(
            "host.responses_client.urlopen",
            side_effect=URLError(ConnectionRefusedError("before connect")),
        ):
            with self.assertRaises(Exception) as connect_error:
                client.complete(**request)

        self.assertFalse(timeout_error.exception.retryable)
        self.assertFalse(status_error.exception.retryable)
        self.assertFalse(partial_error.exception.retryable)
        self.assertTrue(connect_error.exception.retryable)

    def test_pair_returns_nonzero_when_either_arm_hits_a_blocker(self) -> None:
        blocked = {
            "schema": "0002-nooa-g9-arm-v1",
            "arm": "0002-NOOA",
            "actions": 0,
            "levels_completed": 0,
            "state": "HOST_FAILURE",
            "stop_reason": "nooa_worker_failure",
            "clean_shutdown": True,
            "trace_complete": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "blocked-pair"
            with patch("run_paired_probe._run_candidate", return_value=blocked):
                with redirect_stdout(StringIO()):
                    status = main(
                        [
                            "--offline",
                            "--max-actions",
                            "2",
                            "--max-model-calls",
                            "4",
                            "--max-model-rounds-per-action",
                            "2",
                            "--run-root",
                            str(run_root),
                        ]
                    )

            summary = json.loads((run_root / "summary.json").read_text())
            self.assertEqual(status, 1)
            self.assertFalse(summary["operational_success"])
            self.assertFalse(summary["arm_verdicts"]["0002-NOOA"]["success"])

    def test_control_transport_projects_the_frozen_profile_without_hidden_retries(
        self,
    ) -> None:
        profile = ProviderRequestProfile.responses(
            model="openai/provider/model",
            wire_model="provider/model",
            api_base="https://gateway.invalid/v1",
            max_output_tokens=4096,
            reasoning_effort="medium",
            request_timeout_seconds=90.0,
        )

        client = _control_client_for_live(profile, "credential-not-recorded")
        payload = client._payload(
            [{"role": "system", "content": "system"}, {"role": "user", "content": "x"}],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "python",
                        "description": "execute",
                        "parameters": {
                            "type": "object",
                            "properties": {"code": {"type": "string"}},
                            "required": ["code"],
                        },
                    },
                }
            ],
        )

        self.assertIs(client.profile, profile)
        self.assertEqual(payload["model"], profile.wire_model)
        self.assertEqual(payload["max_output_tokens"], profile.max_output_tokens)
        self.assertEqual(payload["reasoning"], {"effort": "medium"})
        self.assertEqual(payload["parallel_tool_calls"], False)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("seed", payload)

    def test_offline_pair_uses_one_spec_and_runs_both_real_harnesses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "pair"
            with redirect_stdout(StringIO()):
                status = main(
                    [
                        "--offline",
                        "--max-actions",
                        "2",
                        "--max-model-calls",
                        "4",
                        "--max-model-rounds-per-action",
                        "2",
                        "--run-root",
                        str(run_root),
                    ]
                )

            self.assertEqual(status, 0)
            pair = json.loads((run_root / "summary.json").read_text())
            # Pinned is the default, because a gate that does not say which
            # endpoint served it is the gate that already passed a broken
            # substrate once.
            self.assertEqual(pair["provider_route_name"], "pinned")
            self.assertTrue(pair["route_can_qualify"])
            self.assertTrue(pair["qualifies_g9"])
            self.assertEqual(pair["mode"], "offline-fixture")
            self.assertEqual(pair["claim_boundary"], "interface usability only")
            self.assertTrue(pair["operational_success"])
            self.assertTrue(
                all(
                    verdict["checks"]["python_on_two_action_points"]
                    and verdict["checks"]["persistent_file_reuse"]
                    for verdict in pair["arm_verdicts"].values()
                )
            )
            self.assertEqual(
                [(run["arm"], run["actions"]) for run in pair["runs"]],
                [("0002-R-minimal", 2), ("0002-NOOA", 2)],
            )
            self.assertTrue((run_root / "control" / "arm-manifest.json").is_file())
            manifest = json.loads(
                (run_root / "control" / "arm-manifest.json").read_text()
            )
            self.assertEqual(manifest["source_arm_id"], "0002-R")
            self.assertEqual(manifest["arm_id"], "0002-R-minimal")
            self.assertIsNone(
                manifest["compatibility_adapter"]["provided_helper_sha256"]
            )
            self.assertTrue((run_root / "control" / "process.jsonl").is_file())
            self.assertTrue((run_root / "candidate" / "worker" / "process.jsonl").is_file())
            self.assertTrue((run_root / "control" / "trace.jsonl").is_file())
            self.assertTrue((run_root / "candidate" / "trace.jsonl").is_file())
            source_state = json.loads((run_root / "source-state.json").read_text())
            self.assertEqual(source_state["schema"], "0002-nooa-source-state-v1")
            self.assertEqual(len(source_state["git"]["revision"]), 40)
            self.assertIn("host_python", source_state["runtime_versions"])
            self.assertIn(
                "experiments/agents/0002-nooa/nooa-runtime/agent.py",
                source_state["source_sha256"],
            )
            self.assertEqual(len(pair["source_state_sha256"]), 64)

            frozen_spec = json.loads((run_root / "paired-run-spec.json").read_text())
            probe_instruction = frozen_spec["usability_probe_instruction"]
            probe_commitment = "NOTES.md=sha256:" + sha256(
                probe_instruction.encode("utf-8")
            ).hexdigest()
            self.assertIn(
                probe_commitment,
                frozen_spec["host_workspace_initialization"],
            )
            self.assertNotIn(
                "NOTES.md=empty",
                frozen_spec["host_workspace_initialization"],
            )
            candidate_requests = [
                json.loads(line)
                for line in (
                    run_root / "candidate" / "worker" / "ipc" / "requests.jsonl"
                ).read_text().splitlines()
            ]
            action_requests = [
                record for record in candidate_requests if record.get("kind") != "stop"
            ]
            self.assertEqual(len(action_requests), 2)
            self.assertTrue(
                all(record["paired_run_spec"] == frozen_spec for record in action_requests)
            )

    def test_an_open_route_runs_the_pair_but_cannot_qualify_the_gate(self) -> None:
        """Succeeding is not the same as being evidence.

        An open route is legitimate for shakedown, so the run still reports its
        operational result. It just cannot stand as G9 evidence, because it does
        not say which endpoint served it -- which is exactly how the previous
        gate passed a substrate that one provider refused outright.
        """

        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "pair"
            with redirect_stdout(StringIO()):
                status = main(
                    [
                        "--offline",
                        "--provider-route",
                        "open",
                        "--max-actions",
                        "2",
                        "--max-model-calls",
                        "4",
                        "--max-model-rounds-per-action",
                        "2",
                        "--run-root",
                        str(run_root),
                    ]
                )

            pair = json.loads((run_root / "summary.json").read_text())
            self.assertEqual(status, 0)
            self.assertTrue(pair["operational_success"])
            self.assertEqual(pair["provider_route_name"], "open")
            self.assertEqual(pair["provider_route"], {})
            self.assertFalse(pair["route_can_qualify"])
            self.assertFalse(pair["qualifies_g9"])


if __name__ == "__main__":
    unittest.main()
