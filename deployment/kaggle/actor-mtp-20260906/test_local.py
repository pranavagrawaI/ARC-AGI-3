"""Offline contract and replay checks; fixtures are not an Actor corpus."""

import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
from build import build
from corpus import validate
from runner import inspect_response
import runner
import corpus
from seal_capture import freeze, CAPTURE


def fixture():
    request = {
        "model": "test-only",
        "messages": [
            {"role": "user", "content": "fixture"},
            {"role": "tool", "tool_call_id": "x", "content": "fixture result"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "action",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"],
                    },
                },
            }
        ],
        "max_tokens": 8192,
        "temperature": 0,
        "top_p": 1,
        "tool_choice": {"type": "function", "function": {"name": "action"}},
    }
    return {
        "schema": "actor-replay-v2",
        "generation_config": {},
        "records": [
            {
                "id": k,
                "kind": k,
                "provenance": "TEST ONLY; never benchmark",
                "request": request,
            }
            for k in ["early", "long-history", "tool-continuation", "forced-action"]
        ],
    }


class Checks(unittest.TestCase):
    def test_capture_identity_and_omissions(self):
        value = json.loads(freeze())
        self.assertEqual(len(value["records"]), 14)
        self.assertEqual(
            [
                r["provenance"]["attempt"]
                for r in value["records"]
                if r["captured_quality"]["tool_schema_errors"]
            ],
            [4, 12, 13],
        )
        for record in value["records"]:
            for field in ("max_tokens", "temperature", "top_p"):
                self.assertNotIn(field, record["request"])
        value["records"][0]["request"]["temperature"] = 0
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            validate(value)

    def test_full_runner_keeps_invalid_results_and_runs_both_arms(self):
        value = json.loads(freeze())
        captured = [
            json.loads((CAPTURE / f"{i:04d}-response.json").read_text())
            for i in range(14)
        ]
        transmitted, fixed = [], []

        def fake_post(request):
            if request.get("max_tokens") == 64:
                return captured[0], 0.1
            index = len(transmitted) % 14
            original = value["records"][index]["request"]
            self.assertEqual(
                {
                    k: v
                    for k, v in request.items()
                    if k
                    not in (
                        "model",
                        "return_meta_info",
                        "return_token_ids",
                        "return_prompt_token_ids",
                    )
                },
                {k: v for k, v in original.items() if k != "model"},
            )
            transmitted.append(request)
            response = json.loads(json.dumps(captured[index]))
            response["choices"][0]["prompt_token_ids"] = [
                1,
                2,
                3,
            ]  # Test fixture, not tokenization evidence.
            return response, 0.1

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "generation_config.json").write_text(
                json.dumps(value["generation_config"])
            )
            runtime = SimpleNamespace(
                discover_model=lambda *a: model,
                verify_model=lambda *a: {},
                sglang_command=lambda *a, **kw: ["fixture-server"],
                R8_L3B_OWNER="fixture",
                process_identity=lambda *a: {},
                wait_for_server=lambda *a: None,
                wait_for_sglang_warmup=lambda *a: None,
                qualify_lifecycle=lambda *a, **kw: {"status": "passed"},
            )
            modules = {
                "corpus_contract": corpus,
                "frozen_runtime": runtime,
                "frozen_engine_metric": SimpleNamespace(
                    generate=lambda *a: fixed.append(a)
                ),
            }
            original_read = Path.read_text

            def read(path, *a, **kw):
                return (
                    json.dumps(value)
                    if str(path) == "/tmp/actor-corpus.json"
                    else original_read(path, *a, **kw)
                )

            gpu = SimpleNamespace(
                cuda=SimpleNamespace(
                    get_device_name=lambda: "RTX PRO 6000",
                    get_device_capability=lambda: (12, 0),
                    device_count=lambda: 1,
                )
            )
            process = SimpleNamespace(pid=12345, poll=lambda: None)
            with (
                patch.object(runner, "OUT", root / "out"),
                patch.dict("sys.modules", {"torch": gpu}),
                patch.object(
                    runner, "load_module", side_effect=lambda name, path: modules[name]
                ),
                patch.object(Path, "read_text", read),
                patch.object(runner, "post", side_effect=fake_post),
                patch.object(
                    runner,
                    "get",
                    side_effect=lambda route: json.dumps(
                        {
                            "sampling_defaults": "model",
                            "speculative_algorithm": "EAGLE"
                            if len(transmitted) >= 14
                            else None,
                        }
                    ),
                ),
                patch.object(runner.subprocess, "Popen", return_value=process),
                patch.object(runner.signal, "signal"),
                patch.object(runner.signal, "alarm"),
            ):
                runner.main()
            results = json.loads((root / "out/results.json").read_text())
            self.assertEqual(len(transmitted), 28)
            self.assertEqual(len(fixed), 4)
            self.assertEqual(
                sum(bool(r["quality"]["tool_schema_errors"]) for r in results), 6
            )
            self.assertEqual(
                json.loads((root / "out/VERDICT.json").read_text())["status"],
                "MEASURED_REVIEW_REQUIRED",
            )

    def test_unsealed_stops_before_gpu(self):
        n = json.loads(build()["actor_mtp.ipynb"])
        with self.assertRaisesRegex(RuntimeError, "NOT SEALED"):
            exec("".join(n["cells"][1]["source"]), {})

    def test_full_request_required(self):
        v = fixture()
        v["records"][0]["request"] = {"usage": {"prompt_tokens": 100}}
        with self.assertRaises(ValueError):
            validate(v)

    def test_sealed_build_keeps_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "fixture.json"
            p.write_text(json.dumps(fixture()))
            outputs = build(p)
        self.assertTrue(json.loads(outputs["manifest.json"])["upload_ready"])
        self.assertEqual(
            json.loads(outputs["kernel-metadata.json"])["competition_sources"],
            ["arc-prize-2026-arc-agi-3"],
        )

    def test_forced_tool_and_schema(self):
        request = fixture()["records"][0]["request"]
        response = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "action",
                                    "arguments": '{"code":"pass"}',
                                }
                            }
                        ]
                    },
                }
            ]
        }
        self.assertEqual(inspect_response(response, request)["tool_schema_errors"], [])
        response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
            "{}"
        )
        self.assertTrue(inspect_response(response, request)["tool_schema_errors"])
        response["choices"][0]["message"]["tool_calls"] = []
        self.assertTrue(inspect_response(response, request)["tool_schema_errors"])


if __name__ == "__main__":
    unittest.main()
