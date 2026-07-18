from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.harness import Decision, Observation, Transition


AGENT_ROOT = Path(__file__).resolve().parents[1] / "0002-llm-repl-agent"
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from llm_repl_agent.agent import (
    EXPERIMENT_FIRST_SYSTEM_PROMPT,
    LlmAgentConfig,
    LlmReplAgent,
    ModelBudgetExceeded,
    SYSTEM_PROMPT,
)
from llm_repl_agent.client import ChatResponse, OpenAICompatibleChatClient
from llm_repl_agent.config import model_request_options, resolve_model_environment
from llm_repl_agent.conversation import RollingConversation
from llm_repl_agent.observations import (
    ColorPermutation,
    OBSERVATION_FILE_CONTRACT,
    pixels_to_png,
    representation_descriptor,
)
from llm_repl_agent.process_log import annotation_template, measurement_summary, ProcessLog
from llm_repl_agent.workspace import PythonResult, PythonWorkspace
from probe_model import PNG_DATA_URL, valid_python_tool_call


def observation(
    *,
    actions: tuple[int, ...] = (1, 2, 3, 4),
    value: int = 0,
    levels: int = 0,
    state: str = "NOT_FINISHED",
) -> Observation:
    pixels = np.full((1, 6, 8), value, dtype=np.int8)
    pixels.setflags(write=False)
    return Observation(
        pixels=pixels,
        available_actions=actions,
        levels_completed=levels,
        state=state,
        full_reset=False,
    )


class ScriptedClient:
    def __init__(self, responses: list[ChatResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def complete(self, *, messages, tools) -> ChatResponse:
        self.requests.append({"messages": list(messages), "tools": list(tools)})
        if not self.responses:
            raise AssertionError("scripted model exhausted")
        return self.responses.pop(0)


class FakeHttpResponse:
    def __init__(self, body: dict[str, object] | None = None) -> None:
        self.body = json.dumps(body or {
            "id": "chat-1",
            "choices": [{"message": {"role": "assistant", "content": "{}"}}],
            "usage": {"prompt_tokens": 3},
        }).encode()

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        pass


def response(content: str, *, response_id: str = "response") -> ChatResponse:
    return ChatResponse(
        message={"role": "assistant", "content": content},
        usage={"prompt_tokens": 10, "completion_tokens": 5},
        response_id=response_id,
    )


class ColorPermutationTests(unittest.TestCase):
    def test_seed_is_reproducible_and_changes_only_public_color_values(self) -> None:
        left = ColorPermutation.from_seed(17)
        right = ColorPermutation.from_seed(17)
        pixels = np.asarray([[[0, 1, 15, 23]]], dtype=np.int16)

        transformed = left.apply(pixels)

        self.assertEqual(left, right)
        self.assertEqual(int(transformed[0, 0, 3]), 23)
        self.assertEqual(
            transformed[0, 0, :3].tolist(),
            [left.mapping[0], left.mapping[1], left.mapping[15]],
        )
        self.assertFalse(transformed.flags.writeable)

    def test_render_descriptor_identifies_the_rendered_grid_layer(self) -> None:
        pixels = np.asarray([
            [[0, 0, 0], [0, 0, 0]],
            [[9, 9, 9], [9, 9, 9]],
        ], dtype=np.int8)
        png = pixels_to_png(pixels, scale=2)
        descriptor = representation_descriptor(pixels, png)
        with Image.open(BytesIO(png)) as image:
            rendered = np.asarray(image)
        with Image.open(BytesIO(pixels_to_png(pixels[-1:], scale=2))) as image:
            expected_last_layer = np.asarray(image)
        with Image.open(BytesIO(pixels_to_png(pixels[:1], scale=2))) as image:
            unexpected_first_layer = np.asarray(image)

        self.assertEqual(descriptor["rendered_image"]["source_grid_layer"], 1)
        self.assertEqual(descriptor["rendered_image"]["source_grid_scale"], 2)
        self.assertEqual(descriptor["rendered_image"]["dtype"], str(rendered.dtype))
        np.testing.assert_array_equal(rendered, expected_last_layer)
        self.assertFalse(np.array_equal(rendered, unexpected_first_layer))


class OpenAICompatibleClientTests(unittest.TestCase):
    def test_posts_stateless_tool_enabled_chat_request(self) -> None:
        client = OpenAICompatibleChatClient(
            base_url="http://model.invalid/v1",
            model="local-model",
            api_key="test-key",
            temperature=0.25,
            max_tokens=123,
            seed=9,
        )
        with patch("llm_repl_agent.client.urlopen", return_value=FakeHttpResponse()) as opened:
            result = client.complete(
                messages=[{"role": "user", "content": "hello"}],
                tools=[{"type": "function", "function": {"name": "python"}}],
            )

        request = opened.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "http://model.invalid/v1/chat/completions")
        self.assertEqual(request.headers["Authorization"], "Bearer test-key")
        self.assertEqual(body["model"], "local-model")
        self.assertEqual(body["seed"], 9)
        self.assertEqual(result.response_id, "chat-1")

    def test_posts_qwen_thinking_and_sampling_contract(self) -> None:
        options = model_request_options("qwen3.6")
        client = OpenAICompatibleChatClient(
            base_url="https://model.invalid/v1",
            model="qwen3.6-27b",
            api_key="test-key",
            **options,
            seed=9,
        )
        with patch("llm_repl_agent.client.urlopen", return_value=FakeHttpResponse()) as opened:
            client.complete(messages=[{"role": "user", "content": "hello"}], tools=[])

        body = json.loads(opened.call_args.args[0].data)
        self.assertNotIn("max_tokens", body)
        self.assertEqual(body["max_completion_tokens"], 4096)
        self.assertEqual(body["temperature"], 1.0)
        self.assertEqual(body["top_p"], 0.95)
        self.assertEqual(body["top_k"], 20)
        self.assertEqual(body["presence_penalty"], 0.0)
        self.assertIs(body["enable_thinking"], True)
        self.assertIs(body["preserve_thinking"], True)

    def test_posts_openrouter_reasoning_contract(self) -> None:
        options = model_request_options("openrouter-qwen3.6")
        client = OpenAICompatibleChatClient(
            base_url="https://openrouter.ai/api/v1",
            model="qwen/qwen3.6-27b",
            api_key="test-key",
            **options,
            seed=9,
        )
        with patch("llm_repl_agent.client.urlopen", return_value=FakeHttpResponse()) as opened:
            client.complete(messages=[{"role": "user", "content": "hello"}], tools=[])

        body = json.loads(opened.call_args.args[0].data)
        self.assertNotIn("max_tokens", body)
        self.assertEqual(body["max_completion_tokens"], 4096)
        self.assertEqual(body["temperature"], 1.0)
        self.assertEqual(body["top_p"], 0.95)
        self.assertEqual(body["top_k"], 20)
        self.assertEqual(body["presence_penalty"], 0.0)
        self.assertEqual(body["reasoning"], {"enabled": True, "exclude": False})
        self.assertNotIn("enable_thinking", body)
        self.assertNotIn("preserve_thinking", body)

    def test_openrouter_profile_rejects_deprecated_max_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_completion_tokens"):
            model_request_options("openrouter-qwen3.6", max_tokens=100)

    def test_normalizes_openrouter_reasoning_for_existing_trace_consumers(self) -> None:
        client = OpenAICompatibleChatClient(
            base_url="https://openrouter.ai/api/v1",
            model="qwen/qwen3.6-27b",
            api_key="test-key",
        )
        response_body = {
            "id": "chat-2",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning": "inspect, then act",
                    "reasoning_details": [{"type": "reasoning.text", "text": "inspect"}],
                },
            }],
        }

        with patch(
            "llm_repl_agent.client.urlopen",
            return_value=FakeHttpResponse(response_body),
        ):
            result = client.complete(messages=[{"role": "user", "content": "hello"}], tools=[])

        self.assertEqual(result.message["reasoning_content"], "inspect, then act")
        self.assertEqual(result.message["reasoning"], "inspect, then act")
        self.assertEqual(
            result.message["reasoning_details"],
            [{"type": "reasoning.text", "text": "inspect"}],
        )


class ModelEnvironmentTests(unittest.TestCase):
    def test_openrouter_configuration_is_explicit_and_uses_only_its_key(self) -> None:
        resolved = resolve_model_environment({
            "OPENROUTER_API_KEY": "openrouter-secret",
            "OPENROUTER_MODEL": "qwen/qwen3.6-27b",
            "QWEN_API_KEY": "qwen-secret",
            "QWEN_MODEL": "qwen3.6-27b",
            "OPENAI_BASE_URL": "https://dashscope.invalid/compatible-mode/v1",
        })

        self.assertEqual(resolved.model, "qwen/qwen3.6-27b")
        self.assertEqual(resolved.endpoint, "https://openrouter.ai/api/v1")
        self.assertEqual(resolved.profile, "openrouter-qwen3.6")
        self.assertEqual(resolved.api_key, "openrouter-secret")
        self.assertEqual(resolved.api_key_source, "OPENROUTER_API_KEY")
        self.assertNotIn("openrouter-secret", repr(resolved))

    def test_openrouter_never_falls_back_to_an_alibaba_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
            resolve_model_environment({
                "OPENROUTER_MODEL": "qwen/qwen3.6-27b",
                "QWEN_API_KEY": "qwen-secret",
                "OPENAI_API_KEY": "generic-secret",
            })

    def test_openrouter_auto_profile_rejects_an_unmatched_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "qwen/qwen3.6-27b"):
            resolve_model_environment({
                "OPENROUTER_MODEL": "some/other-model",
                "OPENROUTER_API_KEY": "openrouter-secret",
            })

    def test_openrouter_endpoint_rejects_an_alibaba_profile_and_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "OpenRouter endpoint"):
            resolve_model_environment(
                {
                    "QWEN_MODEL": "qwen3.6-27b",
                    "QWEN_API_KEY": "qwen-secret",
                },
                endpoint="https://openrouter.ai/api/v1",
            )

    def test_openrouter_key_requires_the_official_https_api_base(self) -> None:
        with self.assertRaisesRegex(ValueError, "official HTTPS"):
            resolve_model_environment(
                {
                    "OPENROUTER_MODEL": "qwen/qwen3.6-27b",
                    "OPENROUTER_API_KEY": "openrouter-secret",
                },
                endpoint="http://openrouter.ai/api/v1",
            )

    def test_manual_openrouter_profile_requires_the_supported_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "qwen/qwen3.6-27b"):
            resolve_model_environment(
                {"OPENROUTER_API_KEY": "openrouter-secret"},
                model="some/other-model",
                profile="openrouter-qwen3.6",
            )

    def test_qwen_names_take_precedence_without_exposing_key_in_repr(self) -> None:
        resolved = resolve_model_environment({
            "QWEN_API_KEY": "qwen-secret",
            "QWEN_MODEL": "qwen3.6-27b",
            "OPENAI_API_KEY": "generic-secret",
            "OPENAI_MODEL": "other-model",
            "OPENAI_BASE_URL": "https://workspace.invalid/compatible-mode/v1",
        })

        self.assertEqual(resolved.model, "qwen3.6-27b")
        self.assertEqual(resolved.profile, "qwen3.6")
        self.assertEqual(resolved.api_key, "qwen-secret")
        self.assertEqual(resolved.api_key_source, "QWEN_API_KEY")
        self.assertNotIn("qwen-secret", repr(resolved))

    def test_repeating_qwen_model_on_cli_does_not_change_profile(self) -> None:
        resolved = resolve_model_environment(
            {
                "QWEN_API_KEY": "qwen-secret",
                "QWEN_MODEL": "qwen3.6-27b",
                "OPENAI_BASE_URL": "https://workspace.invalid/compatible-mode/v1",
            },
            model="qwen3.6-27b",
        )

        self.assertEqual(resolved.profile, "qwen3.6")

    def test_remote_endpoint_requires_a_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "API key"):
            resolve_model_environment({
                "OPENAI_MODEL": "remote-model",
                "OPENAI_BASE_URL": "https://model.invalid/v1",
            })

    def test_local_generic_endpoint_uses_empty_compatibility_key(self) -> None:
        resolved = resolve_model_environment({"OPENAI_MODEL": "local-model"})

        self.assertEqual(resolved.endpoint, "http://127.0.0.1:8000/v1")
        self.assertEqual(resolved.api_key, "EMPTY")
        self.assertEqual(resolved.profile, "generic")

    def test_hosted_keys_are_never_sent_to_a_local_endpoint(self) -> None:
        resolved = resolve_model_environment(
            {
                "QWEN_API_KEY": "hosted-secret",
                "QWEN_MODEL": "qwen3.6-27b",
            },
            endpoint="http://127.0.0.1:8000/v1",
        )

        self.assertEqual(resolved.api_key, "EMPTY")
        self.assertIsNone(resolved.api_key_source)


class ModelProbeTests(unittest.TestCase):
    def test_embedded_image_meets_qwen_minimum_dimensions(self) -> None:
        encoded = PNG_DATA_URL.split(",", 1)[1]
        png = base64.b64decode(encoded)
        width, height = struct.unpack(">II", png[16:24])

        self.assertGreater(width, 10)
        self.assertGreater(height, 10)

    def test_requires_parseable_python_code_arguments(self) -> None:
        valid = [{
            "function": {
                "name": "python",
                "arguments": json.dumps({"code": "print(6 * 7)"}),
            }
        }]
        malformed = [{
            "function": {"name": "python", "arguments": "not-json"}
        }]

        self.assertTrue(valid_python_tool_call(valid))
        self.assertFalse(valid_python_tool_call(malformed))


class RollingConversationTests(unittest.TestCase):
    def test_evicts_complete_old_turns_but_never_the_system_message(self) -> None:
        conversation = RollingConversation(
            system_message={"role": "system", "content": "stable"},
            max_turns=2,
        )
        conversation.append(step=0, messages=[{"role": "user", "content": "zero"}])
        conversation.append(step=1, messages=[{"role": "user", "content": "one"}])

        evicted = conversation.append(
            step=2,
            messages=[{"role": "user", "content": "two"}],
        )

        self.assertEqual(evicted, (0,))
        self.assertEqual(conversation.retained_steps, (1, 2))
        self.assertEqual(conversation.messages()[0]["content"], "stable")
        self.assertNotIn("zero", [item.get("content") for item in conversation.messages()])


class PythonWorkspaceTests(unittest.TestCase):
    def test_model_text_is_valid_bounded_json_with_truncation_metadata(self) -> None:
        result = PythonResult(
            ok=True,
            stdout="x" * 10_000,
            stderr="",
        )

        model_text = result.model_text()
        payload = json.loads(model_text)

        self.assertLessEqual(len(model_text), 4_000)
        self.assertTrue(payload["truncated"])
        self.assertGreater(payload["original_chars"], payload["limit"])
        self.assertEqual(payload["limit"], 4_000)
        self.assertTrue(payload["stdout"].startswith("x"))

    def test_namespace_and_files_persist_without_repo_or_home_mounts(self) -> None:
        with PythonWorkspace(timeout_seconds=5) as workspace:
            first = workspace.execute(
                "from pathlib import Path\n"
                "answer = 6 * 7\n"
                "Path('notes.txt').write_text(str(answer))\n"
            )
            evidence = workspace.evidence_state()
            second = workspace.execute(
                "from pathlib import Path\n"
                "print(answer, Path('notes.txt').read_text(), Path('/home').exists())\n"
            )
            network = workspace.execute(
                "import socket\n"
                "try:\n"
                "    socket.socket()\n"
                "except OSError as error:\n"
                "    print(type(error).__name__, error.errno)\n"
            )
            environment = workspace.execute(
                "import importlib.util, os\n"
                "print('ARC_API_KEY' in os.environ, 'QWEN_API_KEY' in os.environ, "
                "os.environ.get('HOME'), "
                "importlib.util.find_spec('requests'), "
                "importlib.util.find_spec('arc_agi'))\n"
            )
            scientific_python = workspace.execute(
                "import numpy as np\n"
                "from PIL import Image\n"
                "values = np.arange(6).reshape(2, 3)\n"
                "image = Image.fromarray(values.astype(np.uint8))\n"
                "try:\n"
                "    open(np.__file__, 'a').close()\n"
                "except OSError as error:\n"
                "    read_only = type(error).__name__\n"
                "print(values.sum(), image.size, read_only)\n"
            )

            self.assertTrue(first.ok, first.traceback)
            self.assertTrue(second.ok, second.traceback)
            self.assertEqual(second.stdout.strip(), "42 42 False")
            self.assertTrue(network.ok, network.traceback)
            self.assertEqual(network.stdout.strip(), "PermissionError 1")
            self.assertTrue(environment.ok, environment.traceback)
            self.assertEqual(environment.stdout.strip(), "False False /workspace None None")
            self.assertTrue(scientific_python.ok, scientific_python.traceback)
            self.assertEqual(
                scientific_python.stdout.strip(),
                "15 (3, 2) OSError",
            )
            runtime = workspace.runtime_info()
            self.assertEqual(runtime["executable"], "/runtime/bin/python3")
            self.assertIn("numpy", runtime["top_level_modules"])
            self.assertIn("PIL", runtime["top_level_modules"])
            self.assertNotIn("requests", runtime["top_level_modules"])
            self.assertEqual(set(runtime["allowed_distributions"]), {"numpy", "pillow"})
            self.assertEqual((workspace.root / "notes.txt").read_text(), "42")
            self.assertEqual(evidence["notes.txt"]["text"], "42")

    def test_snapshot_preserves_symlinks_instead_of_reading_host_targets(self) -> None:
        with TemporaryDirectory() as directory, PythonWorkspace(timeout_seconds=5) as workspace:
            created = workspace.execute(
                "from pathlib import Path\nPath('outside-link').symlink_to('/etc/passwd')\n"
            )
            destination = Path(directory) / "snapshot"

            workspace.snapshot(destination)

            self.assertTrue(created.ok, created.traceback)
            self.assertTrue((destination / "outside-link").is_symlink())
            self.assertEqual((destination / "outside-link").readlink(), Path("/etc/passwd"))
            self.assertEqual(
                workspace.evidence_state()["outside-link"],
                {"kind": "symlink", "target": "/etc/passwd"},
            )


class LlmReplAgentTests(unittest.TestCase):
    def test_final_deliberation_round_requires_an_action_without_python(self) -> None:
        def python_response(call_id: str) -> ChatResponse:
            return ChatResponse(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "python",
                            "arguments": json.dumps({"code": "print('inspect')"}),
                        },
                    }],
                },
                usage={},
                response_id=call_id,
            )

        client = ScriptedClient([
            python_response("call-1"),
            python_response("call-2"),
            response('{"action": 1, "data": null, "reason": "probe"}'),
        ])
        with TemporaryDirectory() as directory, PythonWorkspace(timeout_seconds=5) as workspace:
            log_path = Path(directory) / "process.jsonl"
            with ProcessLog(log_path) as process_log:
                agent = LlmReplAgent(
                    client=client,
                    workspace=workspace,
                    process_log=process_log,
                    config=LlmAgentConfig(max_model_rounds_per_action=3),
                )

                decision = agent.act(observation())

            events = [json.loads(line) for line in log_path.read_text().splitlines()]

        self.assertEqual(decision.action, 1)
        self.assertEqual([len(request["tools"]) for request in client.requests], [1, 1, 0])
        self.assertIn(
            "final deliberation round",
            client.requests[-1]["messages"][-1]["content"].lower(),
        )
        boundary = next(event for event in events if event["type"] == "action_required")
        self.assertEqual(boundary["action_step"], 1)
        self.assertEqual(boundary["model_round"], 3)

    def test_final_deliberation_round_repairs_an_invalid_action_without_python(self) -> None:
        python_response = ChatResponse(
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
            response_id="call-1",
        )
        client = ScriptedClient([
            python_response,
            response('{"action": 99, "data": null, "reason": "invalid"}'),
            response('{"action": 1, "data": null, "reason": "probe"}'),
        ])
        with TemporaryDirectory() as directory, PythonWorkspace(timeout_seconds=5) as workspace:
            log_path = Path(directory) / "process.jsonl"
            with ProcessLog(log_path) as process_log:
                agent = LlmReplAgent(
                    client=client,
                    workspace=workspace,
                    process_log=process_log,
                    config=LlmAgentConfig(
                        max_model_rounds_per_action=2,
                        max_decision_repairs=1,
                    ),
                )

                decision = agent.act(observation())

            events = [json.loads(line) for line in log_path.read_text().splitlines()]

        self.assertEqual(decision.action, 1)
        self.assertEqual([len(request["tools"]) for request in client.requests], [1, 0, 0])
        repairs = [event for event in events if event["type"] == "decision_repair"]
        self.assertEqual(len(repairs), 1)
        self.assertEqual(repairs[0]["model_round"], 2)

    def test_final_deliberation_round_rejects_unadvertised_python_without_executing(self) -> None:
        unadvertised_python = ChatResponse(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-unadvertised",
                    "type": "function",
                    "function": {
                        "name": "python",
                        "arguments": json.dumps({
                            "code": "open('should-not-exist.txt', 'w').write('executed')"
                        }),
                    },
                }],
            },
            usage={},
            response_id="call-unadvertised",
        )
        client = ScriptedClient([
            unadvertised_python,
            response('{"action": 1, "data": null, "reason": "probe"}'),
        ])
        with TemporaryDirectory() as directory, PythonWorkspace(timeout_seconds=5) as workspace:
            log_path = Path(directory) / "process.jsonl"
            with ProcessLog(log_path) as process_log:
                agent = LlmReplAgent(
                    client=client,
                    workspace=workspace,
                    process_log=process_log,
                    config=LlmAgentConfig(
                        max_model_rounds_per_action=1,
                        max_decision_repairs=1,
                    ),
                )

                decision = agent.act(observation())

            events = [json.loads(line) for line in log_path.read_text().splitlines()]
            file_was_created = (workspace.root / "should-not-exist.txt").exists()

        self.assertEqual(decision.action, 1)
        self.assertEqual([len(request["tools"]) for request in client.requests], [0, 0])
        self.assertFalse(file_was_created)
        self.assertFalse(any(event["type"] == "python_tool" for event in events))
        repair = next(event for event in events if event["type"] == "decision_repair")
        self.assertIn("Python is unavailable", repair["error"])

    def test_observation_prompt_and_file_have_an_unambiguous_shared_contract(self) -> None:
        client = ScriptedClient([
            response('{"action": 1, "data": null, "reason": "probe"}')
        ])
        with TemporaryDirectory() as directory, PythonWorkspace(timeout_seconds=5) as workspace:
            with ProcessLog(Path(directory) / "process.jsonl") as process_log:
                agent = LlmReplAgent(
                    client=client,
                    workspace=workspace,
                    process_log=process_log,
                )

                agent.act(observation())

            prompt_text = client.requests[0]["messages"][1]["content"][0]["text"]
            prompt_record = json.loads(prompt_text)
            file_record = json.loads(
                (workspace.root / "current_observation.json").read_text()
            )
            workspace_guide = (workspace.root / "WORKSPACE.md").read_text()
            with Image.open(workspace.root / "current_frame.png") as rendered:
                rendered_mode = rendered.mode
                rendered_array = np.asarray(rendered)
                rendered_shape = list(rendered_array.shape)
                rendered_dtype = str(rendered_array.dtype)

        self.assertNotIn("current", prompt_record)
        file_metadata = {
            key: value
            for key, value in file_record.items()
            if key not in {"pixels", "step"}
        }
        self.assertEqual(prompt_record["current_metadata"], file_metadata)
        self.assertNotIn("pixels", prompt_record["current_metadata"])
        self.assertEqual(file_record["step"], prompt_record["step"])
        self.assertEqual(np.asarray(file_record["pixels"]).shape, (1, 6, 8))
        representations = prompt_record["representations"]
        self.assertEqual(representations["grid"], {
            "path": "current_observation.json",
            "json_pointer": "/pixels",
            "encoding": "nested_integer_array",
            "values": "integer_color_ids",
            "json_value_type": "integer",
            "axes": ["layer", "row", "column"],
            "shape": [1, 6, 8],
        })
        self.assertIsInstance(file_record["pixels"][0][0][0], int)
        self.assertEqual(representations["rendered_image"], {
            "path": "current_frame.png",
            "encoding": "rgb",
            "axes": ["row", "column", "channel"],
            "shape": [48, 64, 3],
            "dtype": rendered_dtype,
            "source_grid_layer": 0,
            "source_grid_scale": 8,
        })
        self.assertEqual(rendered_mode, "RGB")
        self.assertEqual(representations["rendered_image"]["shape"], rendered_shape)
        self.assertEqual(representations["rendered_image"]["dtype"], rendered_dtype)
        self.assertIn(OBSERVATION_FILE_CONTRACT, SYSTEM_PROMPT)
        self.assertIn(OBSERVATION_FILE_CONTRACT, workspace_guide)

    def test_drop_stale_python_context_keeps_only_latest_consumable_exchange(self) -> None:
        def tool_response(call_id: str, code: str) -> ChatResponse:
            return ChatResponse(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "python",
                            "arguments": json.dumps({"code": code}),
                        },
                    }],
                },
                usage={},
                response_id=call_id,
            )

        client = ScriptedClient([
            tool_response("call-1", "saved = 41\nprint('first-result')"),
            tool_response("call-2", "print('second-result', saved + 1)"),
            tool_response("call-3", "print('third-result')"),
            response('{"action": 1, "data": null, "reason": "probe"}'),
        ])
        with TemporaryDirectory() as directory, PythonWorkspace(timeout_seconds=5) as workspace:
            log_path = Path(directory) / "process.jsonl"
            with ProcessLog(log_path) as process_log:
                agent = LlmReplAgent(
                    client=client,
                    workspace=workspace,
                    process_log=process_log,
                    config=LlmAgentConfig(drop_stale_python_context=True),
                )

                decision = agent.act(observation())

            events = [json.loads(line) for line in log_path.read_text().splitlines()]

        self.assertEqual(decision.action, 1)
        self.assertEqual([len(request["messages"]) for request in client.requests], [2, 4, 5, 5])
        deliberation = []
        for request in client.requests:
            observation_text = request["messages"][1]["content"][0]["text"]
            observation_payload = json.loads(observation_text)
            self.assertIn("representations", observation_payload)
            deliberation.append(observation_payload["deliberation"])
        self.assertEqual(deliberation, [
            {
                "max_model_rounds": 10,
                "model_round": 1,
                "python_calls_this_action": 0,
                "remaining_rounds": 10,
            },
            {
                "max_model_rounds": 10,
                "model_round": 2,
                "python_calls_this_action": 1,
                "remaining_rounds": 9,
            },
            {
                "max_model_rounds": 10,
                "model_round": 3,
                "python_calls_this_action": 2,
                "remaining_rounds": 8,
            },
            {
                "max_model_rounds": 10,
                "model_round": 4,
                "python_calls_this_action": 3,
                "remaining_rounds": 7,
            },
        ])
        third_request = json.dumps(client.requests[2]["messages"])
        fourth_request = json.dumps(client.requests[3]["messages"])
        self.assertNotIn("first-result", third_request)
        self.assertIn("second-result", third_request)
        self.assertIn("42", third_request)
        self.assertNotIn("second-result", fourth_request)
        self.assertIn("third-result", fourth_request)
        self.assertIn("Earlier Python exchanges omitted", third_request)
        persisted = json.dumps(agent.conversation.messages())
        self.assertNotIn("Python exchanges omitted", persisted)
        self.assertNotIn("third-result", persisted)
        self.assertEqual(len(agent.conversation.messages()), 3)
        compactions = [event for event in events if event["type"] == "context_compaction"]
        self.assertEqual([event["phase"] for event in compactions], [
            "within_action",
            "within_action",
            "action_complete",
        ])

    def test_deliberation_counts_executed_python_calls_and_resets_per_action(self) -> None:
        multi_tool_response = ChatResponse(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "python",
                            "arguments": json.dumps({"code": "print('first')"}),
                        },
                    },
                    {
                        "id": "call-malformed",
                        "type": "function",
                        "function": {"name": "python", "arguments": "not-json"},
                    },
                    {
                        "id": "call-2",
                        "type": "function",
                        "function": {
                            "name": "python",
                            "arguments": json.dumps({"code": "print('second')"}),
                        },
                    },
                ],
            },
            usage={},
        )
        client = ScriptedClient([
            multi_tool_response,
            response('{"action": 1, "data": null, "reason": "first probe"}'),
            response('{"action": 2, "data": null, "reason": "second probe"}'),
        ])
        before = observation(value=0)
        after = observation(value=1)
        with TemporaryDirectory() as directory, PythonWorkspace(timeout_seconds=5) as workspace:
            with ProcessLog(Path(directory) / "process.jsonl") as process_log:
                agent = LlmReplAgent(
                    client=client,
                    workspace=workspace,
                    process_log=process_log,
                    config=LlmAgentConfig(drop_stale_python_context=True),
                )

                first_decision = agent.act(before)
                agent.observe(Transition(before, first_decision, after))
                second_decision = agent.act(after)

        first_request = json.loads(client.requests[0]["messages"][1]["content"][0]["text"])
        after_tools = json.loads(client.requests[1]["messages"][1]["content"][0]["text"])
        second_action_request = json.loads(
            client.requests[2]["messages"][-1]["content"][0]["text"]
        )
        self.assertEqual(first_decision.action, 1)
        self.assertEqual(second_decision.action, 2)
        self.assertEqual(first_request["deliberation"]["python_calls_this_action"], 0)
        self.assertEqual(after_tools["deliberation"], {
            "max_model_rounds": 10,
            "model_round": 2,
            "python_calls_this_action": 2,
            "remaining_rounds": 9,
        })
        self.assertEqual(second_action_request["deliberation"], {
            "max_model_rounds": 10,
            "model_round": 1,
            "python_calls_this_action": 0,
            "remaining_rounds": 10,
        })

    def test_python_compaction_preserves_intervening_decision_repair(self) -> None:
        def tool_response(call_id: str, result: str) -> ChatResponse:
            return ChatResponse(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "python",
                            "arguments": json.dumps({"code": f"print('{result}')"}),
                        },
                    }],
                },
                usage={},
            )

        client = ScriptedClient([
            tool_response("call-1", "first-result"),
            response('{"action": 99, "data": null, "reason": "invalid"}'),
            tool_response("call-2", "second-result"),
            response('{"action": 1, "data": null, "reason": "probe"}'),
        ])
        with TemporaryDirectory() as directory, PythonWorkspace(timeout_seconds=5) as workspace:
            with ProcessLog(Path(directory) / "process.jsonl") as process_log:
                agent = LlmReplAgent(
                    client=client,
                    workspace=workspace,
                    process_log=process_log,
                    config=LlmAgentConfig(
                        drop_stale_python_context=True,
                        max_decision_repairs=1,
                    ),
                )

                decision = agent.act(observation())

        final_request = json.dumps(client.requests[-1]["messages"])
        deliberation = [
            json.loads(request["messages"][1]["content"][0]["text"])["deliberation"]
            for request in client.requests
        ]
        self.assertEqual(decision.action, 1)
        self.assertEqual(
            [item["python_calls_this_action"] for item in deliberation],
            [0, 1, 1, 2],
        )
        self.assertEqual([item["model_round"] for item in deliberation], [1, 2, 3, 4])
        self.assertNotIn("first-result", final_request)
        self.assertIn("second-result", final_request)
        self.assertIn("Your proposed action was invalid", final_request)
        self.assertIn('\\"action\\": 99', final_request)

    def test_selected_system_prompt_is_sent_in_the_first_model_request(self) -> None:
        client = ScriptedClient([
            response('{"action": 1, "data": null, "reason": "probe early"}')
        ])
        with TemporaryDirectory() as directory, PythonWorkspace(timeout_seconds=5) as workspace:
            with ProcessLog(Path(directory) / "process.jsonl") as process_log:
                agent = LlmReplAgent(
                    client=client,
                    workspace=workspace,
                    process_log=process_log,
                    system_prompt=EXPERIMENT_FIRST_SYSTEM_PROMPT,
                )

                agent.act(observation())

        self.assertEqual(
            client.requests[0]["messages"][0],
            {"role": "system", "content": EXPERIMENT_FIRST_SYSTEM_PROMPT},
        )

    def test_python_tool_then_legal_action_repair_and_transition_logging(self) -> None:
        tool_response = ChatResponse(
            message={
                "role": "assistant",
                "content": None,
                "reasoning_content": "I should inspect the workspace first.",
                "reasoning_details": [{
                    "type": "reasoning.text",
                    "text": "I should inspect the workspace first.",
                }],
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "python",
                        "arguments": json.dumps({
                            "code": "from pathlib import Path\n"
                            "probe = 21 * 2\n"
                            "Path('notes.txt').write_text(str(probe))\n"
                            "print('x' * 5000)"
                        }),
                    },
                }],
            },
            usage={"prompt_tokens": 4},
            response_id="tool-response",
        )
        client = ScriptedClient([
            tool_response,
            response('{"action": 99, "data": null, "reason": "try"}', response_id="bad"),
            response('{"action": 1, "data": null, "reason": "probe"}', response_id="good"),
        ])
        before = observation(value=0)
        after = observation(value=1)
        with TemporaryDirectory() as directory, PythonWorkspace(timeout_seconds=5) as workspace:
            log_path = Path(directory) / "process.jsonl"
            with ProcessLog(log_path) as process_log:
                agent = LlmReplAgent(
                    client=client,
                    workspace=workspace,
                    process_log=process_log,
                    config=LlmAgentConfig(max_decision_repairs=1),
                )

                decision = agent.act(before)
                agent.observe(Transition(before, decision, after))

            events = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertEqual(decision.action, 1)
            self.assertEqual(decision.reasoning["decision_repairs"], 1)
            self.assertEqual(agent.usage, {"prompt_tokens": 24, "completion_tokens": 10})
            self.assertEqual(len(client.requests), 3)
            continued_assistant = client.requests[1]["messages"][-2]
            self.assertEqual(
                continued_assistant["reasoning_content"],
                "I should inspect the workspace first.",
            )
            self.assertEqual(
                continued_assistant["reasoning_details"],
                [{
                    "type": "reasoning.text",
                    "text": "I should inspect the workspace first.",
                }],
            )
            python_tool = next(item for item in events if item["type"] == "python_tool")
            self.assertTrue(python_tool["output_truncated"])
            self.assertEqual(python_tool["output_chars"], len(python_tool["output"]))
            self.assertEqual(json.loads(python_tool["output"])["truncated"], True)
            self.assertEqual(
                client.requests[1]["messages"][-1]["content"],
                python_tool["output"],
            )
            workspace_change = next(item for item in events if item["type"] == "workspace_change")
            self.assertEqual(workspace_change["phase"], "policy")
            self.assertEqual(workspace_change["changes"][0]["path"], "notes.txt")
            self.assertTrue(any(item["type"] == "decision_repair" for item in events))
            transition = next(item for item in events if item["type"] == "transition")
            self.assertEqual(transition["changed_pixels"], 48)
            self.assertEqual(
                (workspace.root / "public_history.jsonl").read_text().count("\n"),
                1,
            )

    def test_model_call_budget_refuses_a_second_request(self) -> None:
        tool_response = ChatResponse(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "python",
                        "arguments": json.dumps({"code": "print(1)"}),
                    },
                }],
            },
            usage={"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        )
        client = ScriptedClient([tool_response, response("{}")])
        with TemporaryDirectory() as directory, PythonWorkspace(timeout_seconds=5) as workspace:
            log_path = Path(directory) / "process.jsonl"
            with ProcessLog(log_path) as process_log:
                agent = LlmReplAgent(
                    client=client,
                    workspace=workspace,
                    process_log=process_log,
                    config=LlmAgentConfig(max_model_calls=1),
                )

                with self.assertRaisesRegex(ModelBudgetExceeded, "model-call budget"):
                    agent.act(observation())

            events = [json.loads(line) for line in log_path.read_text().splitlines()]

        self.assertEqual(len(client.requests), 1)
        self.assertEqual(agent.model_calls, 1)
        self.assertTrue(any(item["type"] == "model_budget_exhausted" for item in events))

    def test_invalid_tool_call_uses_the_bounded_result_envelope(self) -> None:
        invalid_tool = ChatResponse(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{}"},
                }],
            },
            usage={},
        )
        client = ScriptedClient([
            invalid_tool,
            response('{"action": 1, "data": null, "reason": "recover"}'),
        ])
        with TemporaryDirectory() as directory, PythonWorkspace(timeout_seconds=5) as workspace:
            with ProcessLog(Path(directory) / "process.jsonl") as process_log:
                agent = LlmReplAgent(
                    client=client,
                    workspace=workspace,
                    process_log=process_log,
                )

                agent.act(observation())

        tool_result = json.loads(client.requests[1]["messages"][-1]["content"])
        self.assertFalse(tool_result["ok"])
        self.assertFalse(tool_result["truncated"])
        self.assertEqual(tool_result["limit"], 4_000)

    def test_reported_token_budget_refuses_the_next_request(self) -> None:
        tool_response = ChatResponse(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "python",
                        "arguments": json.dumps({"code": "print(1)"}),
                    },
                }],
            },
            usage={"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        )
        client = ScriptedClient([tool_response, response("{}")])
        with TemporaryDirectory() as directory, PythonWorkspace(timeout_seconds=5) as workspace:
            with ProcessLog(Path(directory) / "process.jsonl") as process_log:
                agent = LlmReplAgent(
                    client=client,
                    workspace=workspace,
                    process_log=process_log,
                    config=LlmAgentConfig(max_reported_tokens=7),
                )

                with self.assertRaisesRegex(ModelBudgetExceeded, "reported-token budget"):
                    agent.act(observation())

        self.assertEqual(len(client.requests), 1)

    def test_reported_token_budget_handles_mixed_usage_schemas(self) -> None:
        first = ChatResponse(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "python",
                        "arguments": json.dumps({"code": "print(1)"}),
                    },
                }],
            },
            usage={"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        )
        second = ChatResponse(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "python",
                        "arguments": json.dumps({"code": "print(2)"}),
                    },
                }],
            },
            usage={"prompt_tokens": 2, "completion_tokens": 2},
        )
        client = ScriptedClient([first, second, response("{}")])
        with TemporaryDirectory() as directory, PythonWorkspace(timeout_seconds=5) as workspace:
            with ProcessLog(Path(directory) / "process.jsonl") as process_log:
                agent = LlmReplAgent(
                    client=client,
                    workspace=workspace,
                    process_log=process_log,
                    config=LlmAgentConfig(max_reported_tokens=11),
                )

                with self.assertRaisesRegex(ModelBudgetExceeded, "11/11"):
                    agent.act(observation())

        self.assertEqual(len(client.requests), 2)

    def test_coordinate_action_requires_in_bounds_integer_data(self) -> None:
        client = ScriptedClient([
            response('{"action": 6, "data": {"x": 99, "y": 1}, "reason": "bad"}'),
            response('{"action": 6, "data": {"x": 7, "y": 5}, "reason": "fixed"}'),
        ])
        with TemporaryDirectory() as directory, PythonWorkspace(timeout_seconds=5) as workspace:
            with ProcessLog(Path(directory) / "process.jsonl") as process_log:
                agent = LlmReplAgent(
                    client=client,
                    workspace=workspace,
                    process_log=process_log,
                    config=LlmAgentConfig(max_decision_repairs=1),
                )
                decision = agent.act(observation(actions=(6,)))

        self.assertEqual(decision.action, 6)
        self.assertEqual(decision.data, {"x": 7, "y": 5})

    def test_retrospective_has_no_tool_and_cannot_mutate_policy_workspace(self) -> None:
        client = ScriptedClient([
            response('{"action": 1, "data": null, "reason": "act"}'),
            response(
                '{"summary": "done", "predictive_description": "", '
                '"reusable_procedure": "", "final_plan": "", "uncertainties": []}'
            ),
        ])
        before = observation(value=0)
        after = observation(value=1, state="WIN")
        with TemporaryDirectory() as directory, PythonWorkspace(timeout_seconds=5) as workspace:
            with ProcessLog(Path(directory) / "process.jsonl") as process_log:
                agent = LlmReplAgent(
                    client=client,
                    workspace=workspace,
                    process_log=process_log,
                )
                decision = agent.act(before)
                agent.observe(Transition(before, decision, after))
                policy_state = workspace.evidence_state()

                report = agent.finalize(after)
                retrospective_state = workspace.evidence_state()

        self.assertEqual(report["summary"], "done")
        self.assertEqual(client.requests[-1]["tools"], [])
        self.assertEqual(retrospective_state, policy_state)

    def test_policy_prompt_contains_no_game_or_benchmark_identifier(self) -> None:
        for prompt in (SYSTEM_PROMPT, EXPERIMENT_FIRST_SYSTEM_PROMPT):
            lowered = prompt.lower()
            self.assertNotIn("arc-agi", lowered)
            self.assertNotIn("ls20", lowered)
            self.assertNotIn("benchmark", lowered)

    def test_experiment_first_prompt_prefers_interaction_over_description(self) -> None:
        lowered = EXPERIMENT_FIRST_SYSTEM_PROMPT.lower()
        self.assertIn("wing it", lowered)
        self.assertIn("quick\nlegal probes", lowered)
        self.assertIn("not for exhaustive description before acting", lowered)


class ProcessAnnotationTests(unittest.TestCase):
    def test_annotation_rubric_is_post_run_and_action_indexed(self) -> None:
        template = annotation_template(actions=3)

        self.assertEqual(template["representation_latency_actions"], None)
        self.assertEqual(len(template["discoveries"]), 7)
        self.assertEqual([row["action"] for row in template["wasted_actions"]], [1, 2, 3])

    def test_completed_annotation_derives_latency_stability_and_waste_counts(self) -> None:
        template = annotation_template(actions=2)
        template["human_compact_description"] = "move the marker to the matching target"
        for record in template["discoveries"].values():
            record["status"] = "not_observed"
        compact = template["discoveries"]["compact_predictive_world_model"]
        compact.update({
            "status": "observed",
            "first_action": 2,
            "description": "marker motion predicts directional action effects",
            "evidence_event_indices": [4],
        })
        template["representation_latency_actions"] = 2
        template["representation_snapshots"] = [{
            "action": 2,
            "description": "marker motion predicts directional action effects",
            "evidence_event_indices": [4],
        }]
        template["representation_stability"].update({
            "status": "stable",
            "unsupported_mutation_count": 0,
        })
        template["wasted_actions"][0]["classification"] = "useful"
        template["wasted_actions"][1].update({
            "classification": "wasted",
            "category": "premature_commitment",
            "evidence_event_indices": [5],
        })

        summary = measurement_summary(template, actions=2, event_indices={1, 2, 3, 4, 5})

        self.assertEqual(summary["representation_latency_actions"], 2)
        self.assertEqual(summary["wasted_action_count"], 1)
        self.assertEqual(summary["wasted_action_counts"]["premature_commitment"], 1)

    def test_measurement_rejects_missing_action_rows_and_evidence(self) -> None:
        template = annotation_template(actions=3)
        template["human_compact_description"] = "description"
        for record in template["discoveries"].values():
            record["status"] = "not_observed"
        template["representation_stability"]["status"] = "not_reached"
        template["wasted_actions"] = [
            {"action": 1, "classification": "useful", "category": None}
        ]

        with self.assertRaisesRegex(ValueError, "each run action exactly once"):
            measurement_summary(template, actions=3, event_indices={1})


if __name__ == "__main__":
    unittest.main()
