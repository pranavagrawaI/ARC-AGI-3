from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIXTURES = (
    ROOT
    / "analysis/deployment-probes/20260901T-successor-concurrency-candidate-v1"
    / "successor-request-fixtures.json"
)


def load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe() -> ModuleType:
    return load(HERE / "successor_concurrency_probe.py", "successor_concurrency_probe_test")


def builder() -> ModuleType:
    return load(HERE / "build_notebook.py", "successor_concurrency_builder_test")


def response_for(request: dict[str, Any]) -> dict[str, Any]:
    schema = request["tools"][0]["function"]["parameters"]
    actions = schema["properties"]["action"]["enum"]
    action = actions[0]
    arguments: dict[str, Any] = {"action": action}
    if action == "CLICK":
        arguments.update(row=0, col=0)
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "reasoning_content": "bounded fixture response",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "take_action",
                                "arguments": json.dumps(arguments),
                            }
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
    }


def test_fixture_file_is_exact_v6_successor_set() -> None:
    module = probe()
    fixtures = module.load_fixtures(FIXTURES)
    assert [fixture["name"] for fixture in fixtures] == [
        "keyboard-16",
        "click-32",
        "mixed-40",
        "space-64",
    ]


def test_validation_rejects_action_outside_frozen_schema() -> None:
    module = probe()
    fixture = module.load_fixtures(FIXTURES)[0]
    response = response_for(fixture["request"])
    response["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ] = '{"action":"CLICK","row":0,"col":0}'
    try:
        module.validate_response(response, fixture)
    except RuntimeError as error:
        assert "not legal" in str(error)
    else:
        raise AssertionError("illegal action passed validation")


def test_validation_rejects_out_of_bounds_click() -> None:
    module = probe()
    fixture = module.load_fixtures(FIXTURES)[1]
    response = response_for(fixture["request"])
    response["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ] = '{"action":"CLICK","row":64,"col":0}'
    try:
        module.validate_response(response, fixture)
    except RuntimeError as error:
        assert "outside the frozen schema" in str(error)
    else:
        raise AssertionError("out-of-bounds CLICK passed validation")


def test_c4_condition_reaches_four_client_requests_and_preserves_all_evidence(
    tmp_path: Path,
) -> None:
    module = probe()
    fixtures = module.load_fixtures(FIXTURES)
    lock = threading.Lock()
    active = 0
    observed_peak = 0

    def fake_post(url: str, payload: bytes, timeout_s: int) -> dict[str, Any]:
        nonlocal active, observed_peak
        request = json.loads(payload)
        with lock:
            active += 1
            observed_peak = max(observed_peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        body = json.dumps(response_for(request)).encode()
        return {"status": 200, "reason": "OK", "headers": {}, "body": body}

    module.post_bytes = fake_post
    result = module.run_condition(
        concurrency=4,
        fixtures=fixtures,
        model_name="served-model",
        endpoint="http://127.0.0.1:1/v1/chat/completions",
        output_dir=tmp_path,
        timeout_s=5,
    )
    assert result["client_inflight_peak"] == 4
    assert observed_peak == 4
    assert result["request_count"] == 4
    assert len(list((tmp_path / "c4").glob("*.request.bin"))) == 4
    assert len(list((tmp_path / "c4").glob("*.response.bin"))) == 4


def test_c1_c2_c4_send_the_same_frozen_request_set(tmp_path: Path) -> None:
    module = probe()
    fixtures = module.load_fixtures(FIXTURES)

    def fake_post(url: str, payload: bytes, timeout_s: int) -> dict[str, Any]:
        request = json.loads(payload)
        time.sleep(0.01)
        return {
            "status": 200,
            "reason": "OK",
            "headers": {},
            "body": json.dumps(response_for(request)).encode(),
        }

    module.post_bytes = fake_post
    hashes = []
    for concurrency in (1, 2, 4):
        result = module.run_condition(
            concurrency=concurrency,
            fixtures=fixtures,
            model_name="served-model",
            endpoint="http://127.0.0.1:1/v1/chat/completions",
            output_dir=tmp_path,
            timeout_s=5,
        )
        hashes.append(
            {
                record["fixture"]: record["request_sha256"]
                for record in result["records"]
            }
        )
    assert hashes[0] == hashes[1] == hashes[2]


def test_built_notebook_has_no_arc_or_submission_path() -> None:
    module = builder()
    notebook = module.build()
    module.validate(notebook)
    metadata = json.loads((HERE / "kernel-metadata.json").read_text())
    assert metadata["competition_sources"] == []
    assert metadata["enable_internet"] is False
    raw = json.dumps(notebook)
    assert "arc-prize-2026-arc-agi-3" not in raw
    assert "submission.parquet" not in raw
